import csv
import os
import numpy as np
from six.moves import urllib
import SimpleITK as sitk
import scipy.ndimage as ndimage
from copy import deepcopy
import math
import h5py

classes= []

# Open csv training file
# Append classes and urls in 2 arrays
with open('train.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        classes.append(row['class'])
classes = np.array(classes)

print(len(classes))

labels = np.unique(classes, return_counts = "True")
print (labels)

n = len(classes)

trainSize = 1000

imgDir = '/fast_data3/knee/lua/images/playImages'

def ZeroPadSlice (imgSlice, desX = 640, desY = 416):
    currX = imgSlice.shape[1]
    currY = imgSlice.shape[2]
    center = [ imgSlice.shape[1], imgSlice.shape[2]]
    if currX < desX :
        diffX = desX - currX
        result_img = np.zeros( [3, desX, currY] , dtype = imgSlice.dtype )
        leftStart = 0
        result_img[:, leftStart:leftStart + currX, : ] = imgSlice
        imgSlice = deepcopy(result_img)
    if currY < desY:
        diffY = desY - currY
        result_img = np.zeros( [3, imgSlice.shape[1], desY] , dtype = imgSlice.dtype )
        upStart = 0
        result_img[:,:,upStart:upStart+imgSlice.shape[2]] = imgSlice
        imgSlice = deepcopy(result_img)

    # Making things a little too specific for this dataset
    # 640 is max, so currX>desX never happens
    # only take care of Y dimension which varies from 425 to 500
    if currY > desY :
        diffY = currY - desY
        chopUp = diffY/2
        chopDown = math.ceil(diffY/2)
        imgSlice = imgSlice[:, :, chopUp:-chopDown]
    return imgSlice

arrDimX = []; arrDimY = []
# read these 2D images
# all so far seem to be of same dimX and dimY
def ReadNormalizeScaleImage(i):
# for i in range(trainSize):
    scale=[1,1,1]
    try:
        sitkImg = sitk.ReadImage(os.path.join(imgDir, 'train_' + str(i) + '.jpg'))
        # Original values between 0 and 255. Maybe do normalization? Not LCN though.
        img = sitk.GetArrayFromImage(sitk.Normalize(sitkImg))
        img = img - img.min()
        img = img/img.max() # get values between 0 and 1
        # img = np.swapaxes(img, 0, 2) # sitk changes the dims. Now num_channels*x*y

        # Remove scaling for now
        # Scale varies from 0.15 to 1. Leads to such changes in image Sizes
        # Tiling was bad for knee mri, so try it later
        # # -----Scaling begins
        # raw_spacing = np.array(list(reversed(sitkImg.GetSpacing())))
        # raw_spacing = np.asarray(raw_spacing)
        #
        # # Adding extra spacing of 1 for the dimesnion along channel
        # raw_spacing_extra = np.array([1])
        # raw_spacing = np.concatenate((raw_spacing_extra, raw_spacing))
        #
        # spacing = raw_spacing/scale
        #
        # scaled_image = ndimage.zoom(img, spacing, order = 1)
        # scaled_image = scaled_image/np.max(scaled_image)
        # img = scaled_image
        # # -----Scaling ends

        # Axis swapping
        # input is 426 * 640 * 3
        # make 3* 640 * 426
        img = np.swapaxes(img, 0, 2)
        # some have 640*426*3
        # i.e. after swapping : 3*426*640
        if img.shape[1] < 600:
            img = np.swapaxes(img, 1, 2)

        # Do zero padding and make divisible by 16
        # No zero slices, all images have a class/label or they have unknown
        # Sizes vary slightly though. Crop eveything or zero pad?
        # Crop Y
        # Scale to 416*640
        img = ZeroPadSlice(img)
        return img
    except RuntimeError:
        print ('image ', i, ' not found')
        pass


# print(np.unique(classes)[0])

startPoint = 1
endPoint = 100
img_arr = np.array([ReadNormalizeScaleImage(i) for i in range(startPoint,endPoint)])
print(img_arr.shape)
# print(img_arr[3])

# Remove images and lables corresponding to None values for images not found in urls
labels = np.array([label for (label,img) in zip(classes[startPoint:endPoint],img_arr) if img is not None  ])
img_arr = np.array([img for img in img_arr if img is not None  ])

# Store as hdf5 file
h5_path_img = os.path.join(imgDir, 'hdf5s/trainImg.h5')
print (h5_path_img)
with h5py.File(h5_path_img ,'w') as hf:
    hf.create_dataset('imgs', data=img_arr)

# make csv file with required labels
np.savetxt("trainLabels.csv", labels, delimiter=",", fmt = '%s') # works for numbers

print(img_arr.shape)
print(labels.shape)
