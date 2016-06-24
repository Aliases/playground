
import csv
import os
import numpy as np
from six.moves import urllib


classes= []
urls =[]

# Open csv training file
# Append classes and urls in 2 arrays
with open('train.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      classes.append(row['class'])
      urls.append(row['image_url'])

print(len(classes), len(urls))
# print(classes[0], urls[0])

n = len(classes)
trainSize = n

trainSize = 100

work_dir = '/fast_data3/knee/lua/images/playImages'

# function to download an image with given url
def may_be_download(url, work_dir, name_ext):
  # make a directory for storing images if it doesn't exist
  if not os.path.exists(work_dir):
    os.mkdir(work_dir)

  # name with which to store the image
  filepath = os.path.join(work_dir, name_ext)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', url, statinfo.st_size, 'bytes.')
  return filepath

# Download as many images as desired using urls from array url
for i in range(n):
  a = may_be_download(urls[i], work_dir, 'train_' + str(i) + '.jpg')
