'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
import math

def read_pixels(folder, verbose = False):
  '''
    Reads 3-D pixel value of the top left corner of each image in folder
    and returns an n x 3 matrix X containing the pixel values
  '''
  n = len(next(os.walk(folder))[2]) # number of files
  X = np.empty([n, 3])
  i = 0

  if verbose:
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))

  for filename in os.listdir(folder):
    # read image
    # img = plt.imread(os.path.join(folder,filename), 0)
    img = cv2.imread(os.path.join(folder,filename))
    # convert from BGR (opencv convention) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # store pixel rgb value
    X[i] = img[0,0].astype(np.float64)/255
    i += 1

    # display
    if verbose:
      h.set_data(img)
      ax.set_title(filename)
      fig.canvas.flush_events()
      plt.show()

  return X


if __name__ == '__main__':
  folder = 'data/training'
  X1 = read_pixels(folder+'/red', verbose = True)  # red pixels
  X2 = read_pixels(folder+'/green')                # green pixels
  X3 = read_pixels(folder+'/blue')                 # blue pixels
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))

  # Gaussian Naive Bayes Training
  sizeX = X.shape
  sizey = y.shape
  K = 3               # classes: RGB
  theta_MLE = np.zeros((K, 1))
  mu_MLE = np.zeros((K, 3))
  sigma_MLE = np.zeros((K, 3))
  theta_sum = np.zeros((K, 1))
  mu_sum = 0
  sigma_sum = 0

  # training parameter: theta
  for k in range(K):              # classes: 1,2,3, k=0,1,2
    for i in range(sizeX[0]):     # rows of X
      if y[i] == k+1:
        theta_sum[k] += 1
    theta_MLE[k] = (1/sizeX[0])*theta_sum[k]
  # print(theta_MLE)

  # training parameter: mu
  for k in range(K):              # classes: 1,2,3, k=0,1,2
    for l in range(sizeX[1]):     # rows of X
      mu_sum = 0
      for i in range(sizeX[0]):
        if y[i] == k+1:
          mu_sum += X[i,l]
        mu_MLE[k,l] = mu_sum/theta_sum[k]
  # print(mu_MLE)

  # training parameter: sigma
  for k in range(K):              # classes: 1,2,3, k=0,1,2
    for l in range(sizeX[1]):     # rows of X
      sigma_sum = 0
      for i in range(sizeX[0]):
        if y[i] == k+1:           # when yi is specific R,G,B
          sigma_sum += (X[i,l] - mu_MLE[k,l])**2
      sigma_MLE[k,l] = math.sqrt(sigma_sum/theta_sum[k])
  # print(sigma_MLE)

  # save training parameters for testing
  np.save('GNB_training_parameters/theta_MLE.npy', theta_MLE)
  np.save('GNB_training_parameters/mu_MLE.npy', mu_MLE)
  np.save('GNB_training_parameters/sigma_MLE.npy', sigma_MLE)





