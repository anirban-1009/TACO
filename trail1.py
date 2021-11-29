from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import pylab
import json

dataset_path = './data'
anns_file_path = dataset_path + '/' + 'annotations.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
#User settings the things to tweek
image_filepath = 'batch_2/000003.JPG'
pylab.rcParams['figure.figsize'] = (28, 28)


#Obtain Exif orientation tag code(just the orientation)
for orientation in ExifTags.TAGS.keys():
    #print(type(orientation), orientation)
    print(ExifTags.TAGS[orientation])
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


#Loads the data set of coco
coco = COCO(anns_file_path)

"""#Find image id
img_id = -1
for img in imgs:
    if img['file_name'] == image_filepath:
        img_id = img['id']
        break

if img_id == -1:
    print('Incorrect file name')

else:
"""
#Load image
I = Image.open(dataset_path + '/' + image_filepath)

#Load and process image metadata
if I._getexif():
    exif = dict(I._getexif().items())
    #rotate portrait and upside down images if neessary
    if orientation in exif:
        if exif[orientation] == 3:
            I = I.rotate(180, expand=True)
        if exif[orientation] == 6:
            I = I.rotate(270, expand=True)
        if exif[orientation] == 8:
            I = I.rotate(90, expand=True)

    
#show image
fig, ax = plt.subplots(1)
plt.axis('off')
plt.imshow(I)

#Load mask ids
annIds = coco.getAnnIds(catIds=[], iscrowd=None)
anns_sel = coco.loadAnns(annIds)

#Show annotations
for ann in anns_sel:
    color = colorsys.hsv_to_rgb(np.random.random(),1,1)
    for seg in ann['segmentation']:
        poly = Polygon(np.array(seg).reshape((int(len(seg)/2),2)))
        p = PatchCollection([poly], facecolor='none', edgecolors=color,linewidths=2,alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=4)
        ax.add_collection(p)

    [x, y, w, h] = ann['bbox']
    rect = Rectangle((x,y),w,h,linewidth=5,edgecolor=color,
                    facecolor='none', alpha=0.7, linestyle ='--')

    ax.add_patch(rect)

plt.savefig('output.png')