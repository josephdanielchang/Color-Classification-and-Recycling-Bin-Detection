'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
import math
import matplotlib.pyplot as plt

class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.theta_MLE = np.load('bin_detection/GNB_training_parameters/theta_MLE.npy')
		self.mu_MLE = np.load('bin_detection/GNB_training_parameters/mu_MLE.npy')
		self.sigma_MLE = np.load('bin_detection/GNB_training_parameters/sigma_MLE.npy')
		pass

	def segment_image(self, img):
		'''
           Obtain a segmented image using a color classifier,
           e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
           call other functions in this class if needed

           Inputs:
              img - original image
           Outputs:
              mask_img - a k-nary image with 1,2,3,4,5,6 if the pixel is of a specific color
        '''
		# YOUR CODE HERE

		theta_MLE = self.theta_MLE
		mu_MLE = self.mu_MLE
		sigma_MLE = self.sigma_MLE

		K = 5  # classes: # binblue, darkgray, green, brown, gray

		X = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # for autograder
		# X = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)  # for local
		rows = X.shape[0]
		cols = X.shape[1]
		channels = X.shape[2]  # 3 for hsv
		# print('sizeX: ', X.shape)

		# Gaussian Naive Bayes Testing
		X = np.reshape(X,(rows*cols,channels))
		y = np.zeros((rows * cols, K))

		for k in range(K):  # classes: 1,2,3,4,5 k=0,1,2,3,4
			sum1 = 0
			for l in range(channels):
				sum1 += math.log((sigma_MLE[k,l])**2) + (((X[:,l] - mu_MLE[k,l])**2)/(sigma_MLE[k,l]**2))
			y[:,k] = math.log(1/(theta_MLE[k]**2)) + sum1

		y_optimal = 1 + np.argmin(y, axis=1)
		y_optimal = np.reshape(y_optimal,(rows,cols))
		mask_img = y_optimal

		# # plot segmentation before thresholding (comment this out for submission as autograder doesn't take plots)
		# mask_rgb = np.zeros((rows, cols, 3))
		# # for 5 color classes
		# mask_rgb[mask_img == 1] = [255,0,0]	   # binblue
		# mask_rgb[mask_img == 2] = [50,50,50]     # darkgray
		# mask_rgb[mask_img == 3] = [35,102,11]    # green
		# mask_rgb[mask_img == 4] = [33,67,101]    # brown
		# mask_rgb[mask_img == 5] = [150,150,150]  # gray
		# mask_rgb = np.uint8(mask_rgb)
		# cv2.imshow('image', mask_rgb)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - mask image with values 1-6 where each number represents a color and 1 is the color to detect
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		# YOUR CODE HERE

		img = np.uint8(img)

		# threshold mask to only keep binblue pixels
		img[img == 1] = 255
		img[img == 2] = 0
		img[img == 3] = 0
		img[img == 4] = 0
		img[img == 5] = 0
		# img[img == 6] = 0

		# # plot segmentation after thresholding (comment this out for submission as autograder doesn't take plots)
		# cv2.imshow('image', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# morphological operations
		kernel5 = np.ones((5, 5), np.uint8)
		kernel3 = np.ones((3, 3), np.uint8)
		kernel1 = np.ones((1, 1), np.uint8)
		img = cv2.erode(img, kernel3, iterations=1)
		img = cv2.erode(img, kernel5, iterations=1)
		img = cv2.dilate(img, kernel5, iterations=1)
		img = cv2.dilate(img, kernel5, iterations=1)
		img = cv2.dilate(img, kernel1, iterations=1)

		# # plot segmentation after morphological operations (comment this out for submission as autograder doesn't take plots)
		# cv2.imshow('image', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# find bounding boxes
		sizeImg = img.shape
		areaImg = sizeImg[0] * sizeImg[1]

		contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

		boxes = []
		for i in range(np.shape(contours)[0]):
			if (cv2.contourArea(contours[i]) > areaImg*0.005):
				x, y, lengthX, lengthY = cv2.boundingRect(contours[i])
				if lengthY < 2.5 * lengthX and lengthY > 1 * lengthX:
					boxes.append([x, y, x + lengthX, y + lengthY])

		# print('estimated boxes: ', boxes)

		return boxes

