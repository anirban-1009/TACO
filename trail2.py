from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab
import numpy as np
import matplotlib.pyplot as plt

dataset_path = './data'
anns_file_path = dataset_path + '/' + 'annotations.json'


#User settings
nr_img_2_display = 10
category_name = 'Bottle cap'
pylab.rcParams['figure.figsize'] = (14, 14)


#obtian Exif orientation tag code
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Oreintation':
        break

#Load dataset as a coco object 
coco = COCO(anns_file_path)

#Get image ids
imgIds = []
catIds = coco.getCatIds(catNms=[category_name])
if catIds:
    #Get all images containing an instance of chosen category
    imgIds = coco.getImgIds(catIds=catIds)
else:
    #Get all images containing an  instance of choosen super category
    catIds = coco.getCatIds(supNms=[category_name])
    for catId in catIds:
        imgIds+= (coco.getImgIds(catIds=catId))
    imgIds = list(set(imgIds))

nr_images_found = len(imgIds)
print('Number of images found: ',nr_images_found)

#Select N random images
random.shuffle(imgIds)
imgs = coco.loadImgs(imgIds[0:min(nr_img_2_display, nr_images_found)])

for img in imgs:
    image_path = dataset_path + '/' + img['file_name']
    #load image
    I = Image.open(image_path)

    #Load and process image metadata
    if I._getexif():
        exif = dict(I._getexif().items())
        #Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180, expand=True)
            if exif[orientation] == 6:
                I = I.rotate(270, expand=True)
            if exif[orientation] == 8:
                I = I.rotate(90, expand=True)

        #Show image
        fig, ax = plt.subplots(1)
        plt.axis('off')
        plt.imshow(I)

        #Load mask ids
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns_sel = coco.loadAnns(annIds)

        #Show annotations
        for ann in anns_sel:
            color = colorsys.hsv_to_rgb(np.random.random(), 1, 1)
            for seg in ann['segmentation']:
                poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
                p = PatchCollection([poly], facecolor=color, edgecolors=color, linewidths=0, alpha=0.4)
                ax.add_collection(p)
                p = PatchCollection([poly], facecolor='none', edgecolor=color, linewidths=2)
                ax.add_collection(p)

            [x, y, w, h] = ann['bbox']
            rect = Rectangle((x,y), w, h, linewidth=2, edgecolor=color,
                            facecolor='none', alpha=0.7, linestyle = '--')

            ax.add_patch(rect)

        plt.savefig('output.png')