'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
import math

if __name__ == '__main__':

  # load hsv training data
  binblue1 = np.load('data/training_hsv_binblue/binblue20.npy')
  binblue2 = np.load('data/training_hsv_binblue/binblue40.npy')
  binblue3 = np.load('data/training_hsv_binblue/binblue60.npy')

  brown1 = np.load('data/training_hsv_brown/brown20.npy')
  brown2 = np.load('data/training_hsv_brown/brown40.npy')
  brown3 = np.load('data/training_hsv_brown/brown60.npy')

  green1 = np.load('data/training_hsv_green/green20.npy')
  green2 = np.load('data/training_hsv_green/green40.npy')
  green3 = np.load('data/training_hsv_green/green60.npy')

  gray1 = np.load('data/training_hsv_gray/gray20.npy')
  gray2 = np.load('data/training_hsv_gray/gray40.npy')
  gray3 = np.load('data/training_hsv_gray/gray60.npy')

  darkgray1 = np.load('data/training_hsv_darkgray/darkgray20.npy')
  darkgray2 = np.load('data/training_hsv_darkgray/darkgray40.npy')
  darkgray3 = np.load('data/training_hsv_darkgray/darkgray60.npy')

  # skyblue1 = np.load('data/training_hsv_skyblue/skyblue20.npy')
  # skyblue2 = np.load('data/training_hsv_skyblue/skyblue40.npy')
  # skyblue3 = np.load('data/training_hsv_skyblue/skyblue60.npy')

  X1 = np.vstack((binblue1, binblue2, binblue3))
  X2 = np.vstack((darkgray1, darkgray2, darkgray3))
  X3 = np.vstack((brown1, brown2, brown3))
  X4 = np.vstack((green1, green2, green3))
  X5 = np.vstack((gray1, gray2, gray3))
  # X6 = np.vstack((skyblue1, skyblue2, skyblue3))

  y1,y2,y3,y4,y5 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0], 3), np.full(X4.shape[0], 4), np.full(X5.shape[0], 5)
  # print('X1:', X1.shape, 'X2:', X2.shape, 'X3:', X3.shape, 'X4:', X4.shape, 'X5:', X5.shape, 'X6:', X6.shape,
  #       'y1:', y1.shape, 'y2:', y2.shape, 'y3:', y3.shape, 'y4:', y4.shape, 'y5:', y5.shape, 'y6:', y6.shape)

  # convert to uint8 due to memory
  X1 = np.uint8(X1)
  X2 = np.uint8(X2)
  X3 = np.uint8(X3)
  X4 = np.uint8(X4)
  X5 = np.uint8(X5)
  # X6 = np.uint8(X6)
  y1 = np.uint8(y1)
  y2 = np.uint8(y2)
  y3 = np.uint8(y3)
  y4 = np.uint8(y4)
  y5 = np.uint8(y5)
  # y6 = np.uint8(y6)

  X, y = np.concatenate((X1,X2,X3,X4,X5)), np.concatenate((y1,y2,y3,y4,y5))
  print('X:', X.shape, 'y:', y.shape)

  # Gaussian Naive Bayes Training
  sizeX = X.shape
  sizey = y.shape
  K = 5                          # classes: binblue, brown, green, gray, darkgray
  theta_MLE = np.zeros((K, 1))
  mu_MLE = np.zeros((K, 3))
  sigma_MLE = np.zeros((K, 3))
  theta_sum = np.zeros((K, 1))
  mu_sum = 0
  sigma_sum = 0

  # training parameter: theta
  for k in range(K):              # classes: 1,2,3,4,5 k=0,1,2,3,4
    # print(k)
    for i in range(sizeX[0]):     # rows of X
      if y[i] == k+1:
        theta_sum[k] += 1
    theta_MLE[k] = (1/sizeX[0])*theta_sum[k]
  # print(theta_MLE)
  np.save('GNB_training_parameters/theta_MLE.npy', theta_MLE)

  # training parameter: mu
  for k in range(K):              # classes: 1,2,3,4,5 k=0,1,2,3,4
    # print(k)
    for l in range(sizeX[1]):     # rows of X
      mu_sum = 0
      for i in range(sizeX[0]):
        if y[i] == k+1:
          mu_sum += X[i,l]
        mu_MLE[k,l] = mu_sum/theta_sum[k]
  # print(mu_MLE)
  np.save('GNB_training_parameters/mu_MLE.npy', mu_MLE)

  # training parameter: sigma
  for k in range(K):              # classes: 1,2,3,4,5 k=0,1,2,3,4
    # print(k)
    for l in range(sizeX[1]):     # rows of X
      sigma_sum = 0
      for i in range(sizeX[0]):
        if y[i] == k+1:           # when yi is specific class
          sigma_sum += (X[i,l] - mu_MLE[k,l])**2
      sigma_MLE[k,l] = math.sqrt(sigma_sum/theta_sum[k])
  # print(sigma_MLE)
  np.save('GNB_training_parameters/sigma_MLE.npy', sigma_MLE)





