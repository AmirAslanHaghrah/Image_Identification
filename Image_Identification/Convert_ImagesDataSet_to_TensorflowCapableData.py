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

Gun_TrainSet_Images_Count = 85              # Number of Gun Images in Train DataSet
NotGun_TrainSet_Images_Count = 85           # Number of NotGun Images in Train DataSet
Gun_TestSet_Images_Count = 15               # Number of Gun Images in Test DataSet
NotGun_TestSet_Images_Count = 15            # Number of NotGun Images in Test DataSet
ImageSize = 73728                           # Max Size of Images in Train DataSet, Sincethere is different size images in dataset. (rgb x height x width) (3 x 128 x 192)


# Matrix which contains whole Train Gun images.
Gun_Images_train_data_2D = np.zeros(shape = (Gun_TrainSet_Images_Count, ImageSize))
# Manipulate 'Gun_Images_train_data_2D' with all Gun images.
for n in range(Gun_TrainSet_Images_Count):
    p = 0
    image = imread('data/Train/Gun/' + str(n) + '.jpg')     # Loading images one by one to image buffer
    
    for c in range(3):      # Trace Image RGB
        for i in range(np.shape(image)[0]):         # Trace Image Height
            for j in range(np.shape(image)[1]):     # Trace Image Width
                Gun_Images_train_data_2D[n][p] = image[i][j][c]
                p+= 1

# Save Manipulated Matrix for next uses. 
np.savetxt('packedData/Gun_Images_train_data_2D.txt', Gun_Images_train_data_2D, fmt='%d')



# Matrix which contains whole Train NotGun images.
NotGun_Images_train_data_2D = np.zeros(shape = (NotGun_TrainSet_Images_Count, ImageSize))
# Maniplute 'NotGun_Images_train_data_2D' with all NotGun images.
for n in range(NotGun_TrainSet_Images_Count):
    p = 0
    image = imread('data/Train/NotGun/' + str(n) + '.jpg')      # Loading images one by one to image buffer 
    
    for c in range(3):      # Trace Image RGB
        for i in range(np.shape(image)[0]):         # Trace Image Height
            for j in range(np.shape(image)[1]):     # Trace Image Width
                NotGun_Images_train_data_2D[n][p] = image[i][j][c]
                p+= 1

# Save Manipulated Matrix for next uses. 
np.savetxt('packedData/NotGun_Images_train_data_2D.txt', NotGun_Images_train_data_2D, fmt='%d')


# Matrix which contains whole Test Gun images.
Gun_Images_test_data_2D = np.zeros(shape = (Gun_TestSet_Images_Count, ImageSize))
# Maniplute 'Gun_Images_test_data_2D' with all Gun images.
for n in range(Gun_TestSet_Images_Count):
    p = 0
    image = imread('data/Test/Gun/' + str(n) + '.jpg')     # Loading images one by one to image buffer
    
    for c in range(3):      # Trace Image RGB
        for i in range(np.shape(image)[0]):         # Trace Image Height
            for j in range(np.shape(image)[1]):     # Trace Image Width
                Gun_Images_test_data_2D[n][p] = image[i][j][c]
                p+= 1

# Save Manipulated Matrix for next uses. 
np.savetxt('packedData/Gun_Images_test_data_2D.txt', Gun_Images_test_data_2D, fmt='%d')




# Matrix which contains whole Test NotGun images.
NotGun_Images_test_data_2D = np.zeros(shape = (NotGun_TestSet_Images_Count, ImageSize))
# Maniplute 'NotGun_Images_test_data_2D' with all NotGun images.
for n in range(NotGun_TestSet_Images_Count):
    p = 0
    image = imread('data/Test/NotGun/' + str(n) + '.jpg')      # Loading images one by one to image buffer 
    
    for c in range(3):      # Trace Image RGB
        for i in range(np.shape(image)[0]):         # Trace Image Height
            for j in range(np.shape(image)[1]):     # Trace Image Width
                NotGun_Images_test_data_2D[n][p] = image[i][j][c]
                p+= 1

# Save Manipulated Matrix for next uses. 
np.savetxt('packedData/NotGun_Images_test_data_2D.txt', NotGun_Images_test_data_2D, fmt='%d')



# One Dimension Vector capable with tensorflow, 
#Train_data_1D = []

# Merge 'Gun_Images_data_2D' and 'NotGun_Images_data_2D' in a One Dimension Vector.
#for n in range(np.shape(Gun_Images_data_2D)[0]):
#    for i in range(np.shape(Gun_Images_data_2D)[1]):
#        Train_data_1D.append(Gun_Images_data_2D[n][i])

#for n in range(np.shape(NotGun_Images_data_2D)[0]):
#    for i in range(np.shape(NotGun_Images_data_2D)[1]):
#        Train_data_1D.append(NotGun_Images_data_2D[n][i])
