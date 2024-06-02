import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

IM_DIM = 96
NUM_PIXELS = IM_DIM**2

numImages = 200
numPoints = 15


def rotate(theta, i, j):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return np.int64(R @ np.array([i,j]))


def firstDerFilters(ss):
    sz = 16
    im1 = np.zeros((sz,sz))
    im2 = np.zeros((sz,sz))
    mid = sz // 2
    for i in range(-mid,mid):
        for j in range(-mid,mid):
            im1[i+mid,j+mid] = -j/ss * multivariate_normal.pdf([i,j], np.zeros(2), ss*np.identity(2))
            im2[i+mid,j+mid] = -i/ss * multivariate_normal.pdf([i,j], np.zeros(2), ss*np.identity(2))
    return [im1, im2]

def secondDerFilters(ss):
    sz = 16
    im1 = np.zeros((sz,sz))
    im2 = np.zeros((sz,sz))
    im3 = np.zeros((sz,sz))
    mid = sz // 2
    for i in range(-mid,mid):
        for j in range(-mid,mid):
            m,n = i,j
            im1[i+mid,j+mid] = ((n/ss)**2 - 1/ss) * multivariate_normal.pdf([m, n], np.zeros(2), ss*np.identity(2))
            m,n = rotate(np.pi/3, m, n)
            im2[i+mid,j+mid] = ((n/ss)**2 - 1/ss) * multivariate_normal.pdf([m, n], np.zeros(2), ss*np.identity(2))
            m,n = rotate(np.pi/3, m, n)
            im3[i+mid,j+mid] = ((n/ss)**2 - 1/ss) * multivariate_normal.pdf([m, n], np.zeros(2), ss*np.identity(2))
    return im3


ims = secondDerFilters(4)
#for im in ims:
plt.imshow(ims, cmap='gray')
plt.show()
