# OMR
*Optical music recognition system written in Python which converts sheet music into a readable text file.

## Algorithms & Techniques
### Binarization:
* local Otsu
### Rotating:
* Hough transform
### Find Line width and Line space:
* run length coding
### Split lines:
* Projection
### Remove lines for scanned images:
* Line Removal with Projection
### Remove lines for hand written images:
* Transfer music symbols to another image with Line width and Line space
### Calculate staff line places:
* Projection
### Segmentation:
* find contours
### Classification:
* NN
### Extract features:
* HOG
### Centre of circles detection:
* Hough Circles
### Primitive detection:
* Morphological operations


## Prerequisites
* Python3
* Numpy
* Skimage
* Open CV
* Sklearn

## Running the tests
* Put your input images inside the input folder
* python main.py $path_of_input_folder $path_of_output_folder
* Output appers inside the output folder

## Testing
![Input Image](/images/04.PNG)
![Output](/images/output4.png)

## Accuracy
### NN Hog Accuracy detection
* 95.7%
### Overall Accuracy for scanned images:
* about 98%

## Contributors
* Dai Alaa
* Dina Alaa
* Mohamed Monsef
* Nerdeen Ahmad
