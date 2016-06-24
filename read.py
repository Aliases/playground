import csv
import os
import numpy as np
from six.moves import urllib
import SimpleITK as sitk


classes= []

# Open csv training file
# Append classes and urls in 2 arrays
with open('train.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      classes.append(row['class'])

print(len(classes))

n = len(classes)

trainSize = 100

imgDir = '/fast_data3/knee/lua/images/playImages'

# image names are train_i.jpg

# read these 2D images
# all so far seem to be of same dimX and dimY
for i in range(trainSize):
  sitkImg = sitk.ReadImage(os.path.join(imgDir, 'train_' + str(i) + '.jpg'))
  imgArr = sitk.GetArrayFromImage(sitkImg)
  imgArr = np.swapaxes(imgArr, 0, 2)
  print(imgArr.shape) # 3*640*426
