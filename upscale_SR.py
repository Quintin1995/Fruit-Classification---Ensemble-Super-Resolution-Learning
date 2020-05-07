import os
import cv2
from   ISR.models import RDN
import numpy as np
import matplotlib.pyplot as plt
from   os import listdir
from   os.path import isfile, join
import skimage.io
print("Starting upscaling")


in_path     = r'data/1x/'
out_path_2x = r'data/2x/'
out_path_4x = r'data/4x/'

rdn = RDN(weights='psnr-small')

Bananas    = [f for f in listdir(os.path.join(in_path, 'Banana')) if isfile(join(os.path.join(in_path, 'Banana'), f))]
Carambolas = [f for f in listdir(os.path.join(in_path, 'Carambola')) if isfile(join(os.path.join(in_path, 'Carambola'), f))]

for idx, Banana_path in enumerate(Bananas):
    Banana_path = os.path.join(in_path, 'Banana', Banana_path)
    print(str(idx) + "  " + Banana_path)
    
    src = skimage.io.imread(Banana_path)

    sr_img_2x = rdn.predict(src)

    cv2.imwrite(os.path.join(out_path_2x, 'Banana', os.path.basename(Banana_path)), cv2.cvtColor(sr_img_2x, cv2.COLOR_BGR2RGB))

    sr_img_4x = rdn.predict(sr_img_2x)

    cv2.imwrite(os.path.join(out_path_4x, 'Banana', os.path.basename(Banana_path)), cv2.cvtColor(sr_img_4x, cv2.COLOR_BGR2RGB))
    if(idx == 0):
        input()


for idx, Carambola_path in enumerate(Carambolas):
    Carambola_path = os.path.join(in_path, 'Carambola', Carambola_path)
    print(str(idx) + "  " + Carambola_path)
    
    src = skimage.io.imread(Carambola_path)

    sr_img_2x = rdn.predict(src)

    cv2.imwrite(os.path.join(out_path_2x, 'Carambola', os.path.basename(Carambola_path)), cv2.cvtColor(sr_img_2x, cv2.COLOR_BGR2RGB))

    sr_img_4x = rdn.predict(sr_img_2x)

    cv2.imwrite(os.path.join(out_path_4x, 'Carambola', os.path.basename(Carambola_path)), cv2.cvtColor(sr_img_4x, cv2.COLOR_BGR2RGB))



print("done")