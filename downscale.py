import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io


in_path = r'data/or_dat_subset/Banana/Banana01.png'

# src = cv2.imread(in_path)
src = skimage.io.imread(in_path)

#percent by which the image is resized
scale_percent = 25

#calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
output = cv2.resize(src, dsize)


plt.imshow(output)
plt.show()

cv2.imwrite('data/data_out_sandbox/cv2-resize-image-50.png', cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

for i in range(3):
    plt.imshow(src[:,:,i])
    plt.show()