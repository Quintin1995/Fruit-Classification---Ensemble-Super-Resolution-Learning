from PIL import Image
import os
from os.path import isfile, join
import numpy as np
from os import listdir
import cv2

data_path = r'data/4x/'
data_out_path = r'data/4x/Banana_new/'

bananen_onlyfiles   = [f for f in listdir(join(data_path, 'Banana')) if isfile(join(data_path, 'Banana', f))]
bananen    = np.asarray([np.array(Image.open(join(data_path, 'Banana', banaan))) for banaan in bananen_onlyfiles])

print(bananen.shape)

count = 0
for idx, banaan in enumerate(bananen):
    if banaan.shape != (256,320,3):
        count += 1
        bananen[idx] = cv2.resize(banaan, (320,256))

if True:
    for idx, banaan in enumerate(bananen):
        banaan = cv2.cvtColor(banaan, cv2.COLOR_BGR2RGB)
        cv2.imwrite(join(data_out_path, "banaan{0}.png".format(str(idx))), banaan)

if False:
    for idx, banaan in enumerate(bananen):
        if banaan.shape == (256,320,3):
            print(banaan.shape)
            print(idx)

print(count)
print(bananen.shape)