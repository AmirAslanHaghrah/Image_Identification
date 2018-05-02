# <Image_Identification>
# <Convert_ImagesDataSet_to_TensorflowCapableData>
# Copyright © <2018> <AmirAslan Haghrah>

# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from skimage.io import imread               # Image Library
import numpy as np


# Mirroring to train data set augmentation
Augmented_Gun_TrainSet_Images_Count = 170              # Number of Gun Images in Train DataSet
Augmented_NotGun_TrainSet_Images_Count = 170           # Number of NotGun Images in Train DataSet

ImageSize = 73728                           # Max Size of Images in Train DataSet, Sincethere is different size images in dataset. (rgb x height x width) (3 x 128 x 192)


# Matrix which contains whole Train Gun images.
Augmented_Gun_Images_train_data_2D = np.zeros(shape = (Augmented_Gun_TrainSet_Images_Count, ImageSize))
# Manipulate 'Gun_Images_train_data_2D' with all Gun images.
for n in range(Augmented_Gun_TrainSet_Images_Count):
    p = 0
    image = imread('data/Train/Gun/' + str(int(n / 2)) + '.jpg')     # Loading images one by one to image buffer
    
    if (n % 2 == 0):
        for c in range(3):      # Trace Image RGB
            for i in range(np.shape(image)[0]):         # Trace Image Height
                for j in range(np.shape(image)[1]):     # Trace Image Width
                    Augmented_Gun_Images_train_data_2D[n][p] = image[i][j][c]
                    p+= 1
    else:
        for c in range(3):      # Trace Image RGB
            for i in range(np.shape(image)[0]):         # Trace Image Height
                for j in range(np.shape(image)[1]):     # Trace Image Width
                    Augmented_Gun_Images_train_data_2D[n][p] = image[i][np.shape(image)[1] - j - 1][c]
                    p+= 1

# Save Manipulated Matrix for next uses. 
np.savetxt('packedData/Augmented_Gun_Images_train_data_2D.txt', Augmented_Gun_Images_train_data_2D, fmt='%d')


# Matrix which contains whole Train NotGun images.
Augmented_NotGun_Images_train_data_2D = np.zeros(shape = (Augmented_NotGun_TrainSet_Images_Count, ImageSize))
# Maniplute 'NotGun_Images_train_data_2D' with all NotGun images.
for n in range(Augmented_NotGun_TrainSet_Images_Count):
    p = 0
    image = imread('data/Train/NotGun/' + str(int(n / 2)) + '.jpg')      # Loading images one by one to image buffer 
    
    if (n % 2 == 0):
        for c in range(3):      # Trace Image RGB
            for i in range(np.shape(image)[0]):         # Trace Image Height
                for j in range(np.shape(image)[1]):     # Trace Image Width
                    Augmented_NotGun_Images_train_data_2D[n][p] = image[i][j][c]
                    p+= 1
    else:
        for c in range(3):      # Trace Image RGB
            for i in range(np.shape(image)[0]):         # Trace Image Height
                for j in range(np.shape(image)[1]):     # Trace Image Width
                    Augmented_NotGun_Images_train_data_2D[n][p] = image[i][np.shape(image)[1] - j - 1][c]
                    p+= 1

# Save Manipulated Matrix for next uses. 
np.savetxt('packedData/Augmented_NotGun_Images_train_data_2D.txt', Augmented_NotGun_Images_train_data_2D, fmt='%d')