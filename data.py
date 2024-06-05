import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from kruskal import kruskalMST

IM_DIM = 96
NUM_PIXELS = IM_DIM**2

numImages = 200
numPoints = 15


def processFacesData():
    f = open("training.csv", 'r')
    keypoints = np.zeros((numImages, numPoints, 2))
    images = np.zeros((numImages, IM_DIM, IM_DIM))
    np.random.seed(5)
    per = np.random.permutation(numImages)

    for count, line in enumerate(f, -1):
        if count < 0:
            continue
        if count >= numImages:
            break
        l = line.rstrip().split(',')
        keypoints[per[count]] = np.flip(np.array(l[:30]).reshape(15,2), axis=1)
        im = np.array(l[-1].split(' ')).reshape(96, 96) 
        im = np.float64(im) / 255.0
        images[per[count]] = im

    return keypoints, images


def visualizeFace(keypoints, image):
    plt.imshow(image, cmap = 'gray')
    plt.plot(keypoints[:, 1], keypoints[:, 0], 'o', color='r')
    plt.show()


def visualizeFaceGraph(keypoints, image, edges):
    plt.imshow(image, cmap = 'gray')
    for i,j in edges:
        plt.plot([keypoints[i, 1], keypoints[j, 1]], [keypoints[i, 0], keypoints[j, 0]], color='lime')
    plt.plot(keypoints[:, 1], keypoints[:, 0], 'o', color='r')
    plt.show()


def calculateEdges(keypoints):
    scores = np.zeros((numPoints, numPoints)) + np.inf

    for i in range(numPoints):
        for j in range(i+1, numPoints):
            diff = keypoints[:, i] - keypoints[:, j]
            optMean = np.mean(diff, axis=0)
            optCov = np.cov(diff.T, bias=True)
            prob = multivariate_normal.pdf(diff, optMean, optCov)
            scores[i,j] = -np.sum(np.log(prob))

    edges = kruskalMST(scores)
    return edges
