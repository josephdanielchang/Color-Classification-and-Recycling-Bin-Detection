'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import math

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.theta_MLE = np.load('pixel_classification/GNB_training_parameters/theta_MLE.npy')
    self.mu_MLE = np.load('pixel_classification/GNB_training_parameters/mu_MLE.npy')
    self.sigma_MLE = np.load('pixel_classification/GNB_training_parameters/sigma_MLE.npy')
    pass

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    # YOUR CODE HERE

    # import training parameters
    theta_MLE = self.theta_MLE
    mu_MLE = self.mu_MLE
    sigma_MLE = self.sigma_MLE

    K = 3  # classes: RGB

    # Gaussian Naive Bayes Testing
    sizeX = X.shape

    y = np.zeros((K,1))
    y_optimal = np.zeros((sizeX[0],1))

    for i in range(sizeX[0]):   # rows of X
      for k in range(K):        # classes: 1,2,3 k=0,1,2
        sum1 = 0
        for l in range(sizeX[1]):
          sum1 += math.log(sigma_MLE[k,l]**2) + (((X[i,l] - mu_MLE[k,l])**2)/sigma_MLE[k,l]**2)
        y[k] = math.log(1/theta_MLE[k]**2) + sum1
      y_optimal[i] = 1 + np.argmin(y)

    # print(y_optimal)

    return y_optimal

