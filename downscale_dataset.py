import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io

#percent by which the image is resized
scale_percent = 25

#input path
in_path  = r'data/or_dat_subset/'
out_path = r'data/1x/'

Bananas    = [f for f in listdir(os.path.join(in_path, 'Banana')) if isfile(join(os.path.join(in_path, 'Banana'), f))]
Carambolas = [f for f in listdir(os.path.join(in_path, 'Carambola')) if isfile(join(os.path.join(in_path, 'Carambola'), f))]


for idx, Banana_path in enumerate(Bananas):
    Banana_path = os.path.join(in_path, 'Banana', Banana_path)
    print(str(idx) + "  " + Banana_path)
    src = skimage.io.imread(Banana_path)

    #calculate the 25 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

        # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    cv2.imwrite(os.path.join(out_path, 'Banana', os.path.basename(Banana_path)), cv2.cvtColor(output, cv2.COLOR_BGR2RGB))



for idx, Carambola_path in enumerate(Carambolas):
    Carambola_path = os.path.join(in_path, 'Carambola', Carambola_path)
    print(str(idx) + "  " + Carambola_path)
    src = skimage.io.imread(Carambola_path)

    #calculate the 25 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

        # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    cv2.imwrite(os.path.join(out_path, 'Carambola', os.path.basename(Carambola_path)), cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


