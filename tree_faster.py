import numpy as np
from scipy.stats import multivariate_normal
from filters import createFilters
from scipy import ndimage
from tqdm import tqdm
import multiprocessing as mp

np.seterr(divide='ignore')

step = 4
numPoints = 15

class FaceTree:
    def __init__(self, edges, root):
        self.root = root
        self.tree = self.calculateTree(edges)
        self.filters = createFilters()

    def getGrid(self, im_shape):
        x = np.arange(step // 2, im_shape[0], step)
        y = np.arange(step // 2, im_shape[1], step)
        yv, xv = np.meshgrid(y, x)
        return np.stack((xv, yv), axis=2).reshape(-1, 2).T

    def calculateTree(self, edges):
        G = {i: [] for i in range(numPoints)}
        tree = {i: [] for i in range(numPoints)}
        for i, j in edges:
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
                
                median_diff = np.median(diff, axis=0)
                mad_diff = np.median(np.abs(diff - median_diff), axis=0)
                
                inliers = (np.abs(diff - median_diff) <= 3 * mad_diff).all(axis=1)
                filtered_diff = diff[inliers]
                
                self.spatialMeans[w] = np.mean(filtered_diff, axis=0)
                self.spatialCovs[w] = np.cov(filtered_diff.T, bias=True)

    def filterResponse(self, im):
        response = np.zeros((im.shape[0], im.shape[1], 27))
        for i in range(27):
            response[:, :, i] = ndimage.correlate(im, self.filters[i], mode='nearest')
        norm_factor = np.sqrt(np.sum(np.square(response), axis=2, keepdims=True))
        return response / norm_factor

    def calculateAppearanceParams(self, keypoints, images):
        num_images = len(images)
        keypointResponses = np.zeros((num_images, numPoints, 27))
        responses = np.array([self.filterResponse(image) for image in images])
        keypoints_int = np.int64(keypoints)
        
        for i in range(num_images):
            keypointResponses[i] = responses[i, keypoints_int[i, :, 0], keypoints_int[i, :, 1]]
        
        self.appearanceMeans = np.zeros((numPoints, 27))
        self.appearanceCovs = np.zeros((numPoints, 27, 27))
        
        for i in range(numPoints):
            response = keypointResponses[:, i, :]
            median_response = np.median(response, axis=0)
            mad_response = np.median(np.abs(response - median_response), axis=0)
            
            inliers = (np.abs(response - median_response) <= 3 * mad_response).all(axis=1)
            filtered_response = response[inliers]
            
            self.appearanceMeans[i] = np.mean(filtered_response, axis=0)
            variances = np.var(filtered_response, axis=0)
            self.appearanceCovs[i] = variances * np.identity(27)

    def deformationCost(self, edgeIndex, loc1, loc2):
        return -np.log(multivariate_normal.pdf(loc1-loc2.T, self.spatialMeans[edgeIndex], self.spatialCovs[edgeIndex]))

    def appearanceCost(self, nodeIndex, loc, response):
        iconicIndex = response[loc[0], loc[1]]
        return -np.log(multivariate_normal.pdf(iconicIndex, self.appearanceMeans[nodeIndex], self.appearanceCovs[nodeIndex]))

    def findChildLocations(self, response):
        B = np.full((numPoints, response.shape[0], response.shape[1]), np.inf)
        D = np.zeros((numPoints, response.shape[0], response.shape[1], 2), dtype=np.int64)
        grid = self.getGrid(response.shape)

        def recurse(v):
            for w in self.tree[v]:
                recurse(w)
            if v != self.root:
                parentLocs = np.array([[a, b] for a in range(step//2, response.shape[0], step) for b in range(step//2, response.shape[1], step)])
                parentLocs_flat = parentLocs.reshape(-1, 2)
                cost = np.zeros((len(parentLocs_flat), grid.shape[1]))
                for idx, pl in enumerate(parentLocs_flat):
                    cost[idx] = self.appearanceCost(v, grid, response) + self.deformationCost(v, pl, grid)
                for w in self.tree[v]:
                    cost += B[w, grid[0], grid[1]]
                min_indices = np.argmin(cost, axis=1)
                B[v, parentLocs[:, 0], parentLocs[:, 1]] = cost[np.arange(len(parentLocs_flat)), min_indices]
                D[v, parentLocs[:, 0], parentLocs[:, 1]] = grid[:, min_indices].T
        
        recurse(self.root)
        return B, D

    def findRootLocation(self, B, response):
        v = self.root
        grid = self.getGrid(response.shape)
        cost = self.appearanceCost(v, grid, response)
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
        B, D = self.findChildLocations(response)
        rootLoc = self.findRootLocation(B, response)
        locations = self.backtrack(rootLoc, D)
        return locations

    def evaluate(self, images, keypoints):
        nums = len(images)
        print("Evaluating on", nums, "images")
        with mp.Pool(mp.cpu_count()) as pool:
            errors = pool.map(self._evaluate_single, zip(images, keypoints))
        return errors

    def _evaluate_single(self, args):
        image, keypoint = args
        locations = self.predict(image)
        return locations - keypoint
    
    def saveWeights(self, filename):
        np.savez(filename, sm=self.spatialMeans, sc=self.spatialCovs, am=self.appearanceMeans, ac=self.appearanceCovs)

    def loadWeights(self, filename):
        file = np.load(filename)
        self.spatialMeans = file['sm']
        self.spatialCovs = file['sc']
        self.appearanceMeans = file['am']
        self.appearanceCovs = file['ac']

