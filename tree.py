import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

IM_DIM = 96
NUM_PIXELS = IM_DIM**2

numImages = 200
numPoints = 15


class FaceTree:

    def __init__(self, edges, root):
        self.tree = self.calculateTree(edges, root)
        self.root = root


    def calculateTree(self, edges, root):
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

        recurse(root)
        return tree
    

    def calculateSpatialParams(self, keypoints):
        self.spatialMeans = np.zeros((numPoints, 2))
        self.spatialCovs = np.zeros((numPoints, 2, 2))

        for v in self.tree:
            for w in self.tree[v]:
                diff = keypoints[:, v] - keypoints[:, w]
                self.spatialMeans[w] = np.mean(diff, axis=0)
                self.spatialCovs[w] = np.cov(diff.T, bias=True)


    def deformationCost(self, edgeIndex, loc1, loc2):
        return -np.log(multivariate_normal.pdf(loc1-loc2, self.spatialMeans[edgeIndex], self.spatialCovs[edgeIndex]))


