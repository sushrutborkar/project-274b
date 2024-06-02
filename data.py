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

    for count, line in enumerate(f, -1):
        if count < 0:
            continue
        if count >= numImages:
            break
        l = line.rstrip().split(',')
        keypoints[count] = np.array(l[:30]).reshape(15,2)
        images[count] = np.array(l[-1].split(' ')).reshape(IM_DIM, IM_DIM)
        images[count] /= 255

    return keypoints, images


def visualizeFace(num, keypoints, images):
    plt.imshow(images[num], cmap = 'gray')
    plt.plot(keypoints[num, :, 0], keypoints[num, :, 1], 'o', color='r')
    plt.show()


def visualizeFaceGraph(num, keypoints, images, edges):
    plt.imshow(images[num], cmap = 'gray')
    for i,j in edges:
        plt.plot([keypoints[num, i, 0], keypoints[num, j, 0]], [keypoints[num, i, 1], keypoints[num, j, 1]], color='lime')
    plt.plot(keypoints[num, :, 0], keypoints[num, :, 1], 'o', color='r')
    plt.show()


def calculateEdges(keypoints):
    scores = np.zeros((numPoints, numPoints))

    for i in range(numPoints):
        for j in range(i+1, numPoints):
            diff = keypoints[:, i] - keypoints[:, j]
            optMean = np.mean(diff, axis=0)
            optCov = np.cov(diff.T, bias=True)
            prob = multivariate_normal.pdf(diff, optMean, optCov)
            scores[i,j] = -np.sum(np.log(prob))

    scores = np.where(scores==0, np.inf, scores)
    edges = kruskalMST(scores)
    return edges





