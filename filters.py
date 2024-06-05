import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

sz = 16


def rotate(theta, i, j):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R @ np.array([i,j])


def firstDerFilter(ss, theta):
    im = np.zeros((sz,sz))
    mid = sz // 2
    for i in range(-mid,mid):
        for j in range(-mid,mid):
            m,n = rotate(theta, i, j)
            im[i+mid,j+mid] = -n/ss * multivariate_normal.pdf([m, n], np.zeros(2), ss*np.identity(2))
    return im / np.sqrt(np.sum(np.square(im)))

def secondDerFilter(ss, theta):
    im = np.zeros((sz,sz))
    mid = sz // 2
    for i in range(-mid,mid):
        for j in range(-mid,mid):
            m,n = rotate(theta, i, j)
            im[i+mid,j+mid] = ((n/ss)**2 - 1/ss) * multivariate_normal.pdf([m, n], np.zeros(2), ss*np.identity(2))
    return im / np.sqrt(np.sum(np.square(im)))

def thirdDerFilter(ss, theta):
    im = np.zeros((sz,sz))
    mid = sz // 2
    for i in range(-mid,mid):
        for j in range(-mid,mid):
            m,n = rotate(theta, i, j)
            im[i+mid,j+mid] = (-(n/ss)**3 + 3*n/ss**2) * multivariate_normal.pdf([m, n], np.zeros(2), ss*np.identity(2))
    return im / np.sqrt(np.sum(np.square(im)))

def createFilters():
    filters = []
    for ss in [16, 32, 64]:
        for theta in [0, np.pi/2]:
            filters.append(firstDerFilter(ss, theta))
        for theta in [0, np.pi/3, 2*np.pi/3]:
            filters.append(secondDerFilter(ss, theta))
        for theta in [0, np.pi/4, 2*np.pi/4, 3*np.pi/4]:
            filters.append(thirdDerFilter(ss, theta))
    return np.array(filters).reshape(27, sz, sz)

def visualizeFilters(filters):
    fig = plt.figure()
    for i in range(27):
        plt.subplot(3, 9, i+1)
        plt.imshow(filters[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

#filters = createFilters()
#visualizeFilters(filters)