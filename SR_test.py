import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ISR.models import RDN

#loading low resolution image
img = Image.open('data/sandbox_data/pik.png')
lr_img = np.array(img)



lr_img = lr_img[:,:,:3]

print(lr_img)
print(lr_img.shape)

plt.imshow(lr_img)
plt.show()

#show low resolution image
# for i in range(4):
#     plt.imshow(lr_img[:,:,i])
#     plt.show()

#set super resolution weights and network
rdn = RDN(weights='psnr-small')


for i in range(4):
    #ceate higher resolution image with the super resolution network.
    sr_img = rdn.predict(lr_img)
    Image.fromarray(sr_img)

    plt.imshow(sr_img)
    plt.show()

    lr_img = sr_img

print("Done")