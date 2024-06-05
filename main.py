from data import *
from tree import FaceTree

numImages = 200
numTrain = 160
numTest = 40
numPoints = 15

# load data
keypoints, images = processFacesData()
trainImages = images[:numTrain]
testImages = images[numTrain:]
trainKeypoints = keypoints[:numTrain]
testKeypoints = keypoints[numTrain:]
edges = calculateEdges(trainKeypoints)
print("finished data processing")

# visualize data
'''
imNum = 12
visualizeFace(keypoints[imNum], images[imNum])
visualizeFaceGraph(keypoints[imNum], images[imNum], edges)
'''

# train
t = FaceTree(edges, 13)
t.train(trainImages, trainKeypoints)
print("finished training")

# predict on training image
im = trainImages[3]
locations = t.predict(im)
visualizeFaceGraph(locations, im, edges)

# predict on test image
im = testImages[4]
locations = t.predict(im)
visualizeFaceGraph(locations, im, edges)


