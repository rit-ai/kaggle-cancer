import cv2
import numpy as np
import os
import re

PREPROCESS_TARGET_DIR = "./test"

"""
Params- img_name: String representing image to be checked
Returns- True if it is a night vision image, false otherwise

Why overcomplicate things? Computes the average color, then checks
to see if it is "very green", or twice as green as it is both blue
or red.
"""
def is_nv(img_name):
    img = cv2.imread(img_name)
    avg_color = np.average(np.average(img, axis=0), axis=0)

    if(avg_color[0] * 2 < avg_color[1] and avg_color[2] * 2 < avg_color[1]):
        return True
    else:
        return False

"""
imgs_dir- directory of images to preprocess
Perform our preprocessing steps on a given directory.
Currently, this is just making a seperate night vision folder
and moving all night vision images into it.
"""
def preprocess(imgs_dir):
    ext = re.compile(".+.jpg")
    if(not os.path.exists(imgs_dir + "/../nv")):
        os.makedirs(imgs_dir + "/../nv")

    for filename in os.listdir(imgs_dir):
        filename = imgs_dir + "/" + filename
        if(ext.match(filename)):
            if(is_nv(filename)):
                os.rename(filename, imgs_dir +"/../nv/" + filename)

#preprocess(PREPROCESS_TARGET_DIR)