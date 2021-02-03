'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':  # since there's nothing above, this function can only run when executed as main script

    # color classes to train: binblue, brown, green, gray, darkgray, skyblue
    colorSelected_H = []
    colorSelected_S = []
    colorSelected_V = []

    for i in range(41, 61):  # there's 60 training images, change indices in increments of 20 to not run out of memory

        # print('labeling image:', i)

        # read the first training image
        folder = 'data/training'
        filename = '%04i.jpg' % (i,)
        img = cv2.imread(os.path.join(folder, filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # images are in BGR format, convert to RGB
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # images are in BGR format, convert to HSV

        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

        # get the image mask from hsv
        mask = my_roi.get_mask(img_hsv)

        # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img_hsv[mask, :].shape[0])

        ax1.imshow(img_rgb)
        ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)

        plt.show(block=True)

        # Save roipoly labeled data

        sizeImg = img_hsv.shape     # (500,281,3)
        sizeMask = mask.shape       # (500,281)

        # vectorize img and mask
        length_mask = sizeMask[0] * sizeMask[1]
        length_img = sizeImg[0] * sizeImg[1]
        maskVector = mask.reshape(length_mask, 1)   # (140500,3)
        imgVector = img_hsv.reshape(length_img, 3)  # (140500,1)
        # print(imgVector.shape)
        # print(maskVector.shape)

        # append hsv pixels
        for j in range(length_mask):
            if maskVector[j, 0] == 1:  # mask true
                colorSelected_H.append(imgVector[j, 0])
                colorSelected_S.append(imgVector[j, 1])
                colorSelected_V.append(imgVector[j, 2])

    # convert to arrays
    colorSelected_H = np.asarray(colorSelected_H)
    colorSelected_S = np.asarray(colorSelected_S)
    colorSelected_V = np.asarray(colorSelected_V)

    # combine HSV into single matrix
    colorSelected_hsv = np.vstack((colorSelected_H,colorSelected_S,colorSelected_V)).transpose()
    # print(colorSelected_hsv.shape)

    # save training data with 'color + incrementOf20' of labeled data
    np.save('data/training_hsv_darkgray/darkgray60.npy', colorSelected_hsv)
