# ECE276A Project 1 - Joseph Chang

## Color Classification and Recycling Bin Detection

### Directory Structure

autograder_code.py<br />
├─── bin_detection/...<br />
│ ├─── data/...<br />
│ ├─── GNB_training_parameters/...<br />
│ ├─── roipoly/...<br />
├─── pixel_classification/...<br />
│ ├─── GNB_training_parameters/...<br />

### Main Files under bin_detection directory
* **data_collector_roipoly**: use to collect data of for various color classes including recycling bin blue using roipoly
* **/data**: contains training color data of binblue, brown, darkgray, gray, green, and skyblue collected using roipoly
* **generate_hsv_data**: Gaussian Naive Bayes training to calculate parameters theta, mu, and sigma for color classes
* **/GNB_training_parameters**: contains saved training parameters theta, mu, sigma
* **bin_detector.py**: contains functions for image segmentation and detecting bounding boxes around recycling bins
* **test_bin_detector.py**: segments image and detects bounding boxes

### Main Files under pixel_classification directory
* **/GNB_training_parameters**: contains saved training parameters theta, mu, sigma
* **generate_rgb_data**: Gaussian Naive Bayes training to calculate parameters theta, mu, and sigma for color classes
* **pixel_classifier.py**: contains function for classifying pixels as red, green ,or blue
* **test_pixel_classifier.py**: classifies pixels as red, green, or blue

### How to Run

Place test images in */bin_detection* directory. Change the *folder* variable's path in *test_bin_detector.py*. If test images are not .jpg, change *filesname.endswith(".jpg")* in the same test file. Then, run *test_bin_detector.py* to perform image segmentation, detection, and return the bounding boxes of recycling bins.

