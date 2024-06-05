import numpy as np
from scipy.stats import multivariate_normal
from filters import createFilters
from scipy import ndimage
np.seterr(divide='ignore')


IM_DIM = 96
NUM_PIXELS = IM_DIM**2
step = 4

numTrain = 160
numPoints = 15


class FaceTree:

    def __init__(self, edges, root):
        self.root = root
        self.tree = self.calculateTree(edges)
        self.filters = createFilters()


    def getGrid(self):
        x = range(step//2,IM_DIM,step)
        y = range(step//2,IM_DIM,step)
        yv, xv = np.meshgrid(x, y)
        return np.stack((xv, yv), axis=2).reshape(-1, 2).T


    def calculateTree(self, edges):
        G = {}
        tree = {}
        for i in range(numPoints):
            G[i] = []
            tree[i] = []
        for i,j in edges:
            G[i].append(j)
            G[j].append(i)
        
        visited = set()

        def recurse(v):
            visited.add(v)
            for w in G[v]:
                if w not in visited:
                    tree[v].append(w)
                    recurse(w)

        recurse(self.root)
        return tree
    

    def calculateSpatialParams(self, keypoints):
        self.spatialMeans = np.zeros((numPoints, 2))
        self.spatialCovs = np.zeros((numPoints, 2, 2))

        for v in self.tree:
            for w in self.tree[v]:
                diff = keypoints[:, v] - keypoints[:, w]
                self.spatialMeans[w] = np.mean(diff, axis=0)
                self.spatialCovs[w] = np.cov(diff.T, bias=True)

        self.absMeans = np.zeros((numPoints, 2))
        self.absCovs = np.zeros((numPoints, 2, 2))
        for i in range(numPoints):
            self.absMeans[i] = np.mean(keypoints[:, i], axis=0)
            self.absCovs[i] = np.cov(keypoints[:, i].T, bias=True)

    
    def filterResponse(self, im):
        response = np.zeros((IM_DIM, IM_DIM, 27))
        for i in range(27):
            response[:,:,i] = ndimage.correlate(im, self.filters[i], mode='nearest')
        return response / np.sqrt(np.sum(np.square(response), axis=2)).reshape(IM_DIM,IM_DIM,1)
    

    def calculateAppearanceParams(self, keypoints, images):
        keypointResponses = np.zeros((numTrain, numPoints, 27))
        for i in range(numTrain):
            response = self.filterResponse(images[i])
            for j in range(numPoints):
                loc = np.int64(keypoints[i, j])
                keypointResponses[i,j] = response[loc[0], loc[1]]

        self.appearanceMeans = np.mean(keypointResponses, axis=0)
        self.appearanceCovs = np.zeros((numPoints, 27, 27))
        variances = np.var(keypointResponses, axis=0)
        for i in range(numPoints):
            self.appearanceCovs[i] = variances[i] * np.identity(27)


    def deformationCost(self, edgeIndex, loc1, loc2):
        return -np.log(multivariate_normal.pdf(loc1-loc2.T, self.spatialMeans[edgeIndex], self.spatialCovs[edgeIndex]))


    def appearanceCost(self, nodeIndex, loc, response):
        iconicIndex = response[loc[0], loc[1]]
        return -np.log(multivariate_normal.pdf(iconicIndex, self.appearanceMeans[nodeIndex], self.appearanceCovs[nodeIndex]))
    

    def absoluteCost(self, nodeIndex, loc):
        return -np.log(multivariate_normal.pdf(loc.T, self.absMeans[nodeIndex], self.absCovs[nodeIndex]))
    
    
    def findChildLocations(self, response):
        B = np.zeros((numPoints, IM_DIM, IM_DIM)) + np.inf
        D = np.zeros((numPoints, IM_DIM, IM_DIM, 2)).astype(np.int64)
        grid = self.getGrid()

        def recurse(v):
            for w in self.tree[v]:
                recurse(w)
            if v != self.root:
                for a in range(step//2,IM_DIM,step):
                    for b in range(step//2,IM_DIM,step):
                        parentLoc = np.array([a,b])
                        cost = self.appearanceCost(v, grid, response) + self.deformationCost(v, parentLoc, grid) + self.absoluteCost(v, grid)
                        for w in self.tree[v]:
                            cost += B[w, grid[0], grid[1]]
                        B[v,a,b] = np.min(cost)
                        D[v,a,b] = grid[:, np.argmin(cost)]

        recurse(self.root)
        return B, D
    

    def findRootLocation(self, B, response):
        v = self.root
        grid = self.getGrid()
        cost = self.appearanceCost(v, grid, response) + self.absoluteCost(v, grid)
        for w in self.tree[v]:
            cost += B[w, grid[0], grid[1]]
        return grid[:, np.argmin(cost)]
    

    def backtrack(self, rootLoc, D):
        locations = np.zeros((numPoints, 2))

        def recurse(v, loc):
            locations[v] = loc
            for w in self.tree[v]:
                optLoc = D[w, loc[0], loc[1]]
                recurse(w, optLoc)
    
        recurse(self.root, rootLoc)
        return locations


    def train(self, images, keypoints):
        self.calculateSpatialParams(keypoints)
        self.calculateAppearanceParams(keypoints, images)


    def predict(self, image):
        response = self.filterResponse(image)
        B,D = self.findChildLocations(response)
        rootLoc = self.findRootLocation(B, response)
        locations = self.backtrack(rootLoc, D)
        return locations
    
