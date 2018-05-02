# <Image_Identification>
# <Seprate_Gun_NotGun_Images_from_Corel10000_DataSet>
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

from shutil import copyfile

# These varaibles use to have Consecutive Indexes in copied images file name.
GunTrainImageIndex = 0
NotGunTrainImageIndex = 0
GunTestImageIndex = 0
NotGunTestImageIndex = 0

# Gun images are in range [4601, 4700].
# Seprate 85 of image as train data and 15 of them as test image.
# Here I use non divisible to 7 indexes as train set and divisible to 7 ones as test set.
for n in range(100):
    if n % 7 != 0:
        copyfile("Corel10000/" + str(4601 + n) + ".jpg", "data/Train/Gun/" + str(GunTrainImageIndex) + ".jpg")
        GunTrainImageIndex+= 1
    else:
        copyfile("Corel10000/" + str(4601 + n) + ".jpg", "data/Test/Gun/" + str(GunTestImageIndex) + ".jpg")
        GunTestImageIndex+= 1

# Selection of 85 Not-Gun image as train set and 15 of them as test set as like as pervious discipline.
# Notice that because we have Guns in range [4601, 4700] one Gun image will be select as Not Gun that can manually replace with another image.
for n in range(100):
    # some random index to choose 100 images from 10000 images.
    index = 100 * n + 2
    if n % 7 != 0:
        copyfile("Corel10000/" + str(index) + ".jpg", "data/Train/NotGun/" + str(NotGunTrainImageIndex) + ".jpg")
        NotGunTrainImageIndex+= 1
    else:
        copyfile("Corel10000/" + str(index) + ".jpg", "data/Test/NotGun/" + str(NotGunTestImageIndex) + ".jpg")
        NotGunTestImageIndex+= 1
