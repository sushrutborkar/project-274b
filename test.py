import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from data import processFacesData, calculateEdges, visualizeFace, visualizeFaceGraph
from tree import FaceTree

IM_DIM = 96
NUM_PIXELS = IM_DIM**2

numImages = 200
numPoints = 15

keypoints, images = processFacesData()
edges = calculateEdges(keypoints)
visualizeFace(20, keypoints, images)
visualizeFaceGraph(20, keypoints, images, edges)

t = FaceTree(edges, 13)

def treeVisit(v):
    print(v)
    for w in t.tree[v]:
        treeVisit(w)

print(t.tree)
t.calculateSpatialParams(keypoints)