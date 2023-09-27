import pydicom as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image

file_name = 'pics/pics/brain/brain_001.dcm'

# read and display dcm file
info = pd.dcmread(file_name)
img = info.pixel_array
plt.imshow(img)
plt.show()

# flip img horizontal
flip_img_hor = np.flipud(img)
plt.imshow(flip_img_hor)
plt.show()

# flip img vertical
flip_img_ver = np.fliplr(img)
plt.imshow(flip_img_ver)
plt.show()

# convert img from uint16 to uint8
cvimg = cv2.convertScaleAbs(img)
plt.imshow(cvimg)
plt.show()

# convert img from uint16 to uint8 with conversion of values (lowering file size)
cvimg2 = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))  # lowering colordepth in progress of converting 65535->255
plt.imshow(cvimg2)
plt.show()

# conver img from uint8 to float (0 -> 1)
cvimg3 = cv2.convertScaleAbs(img, alpha=(1.0 / 255.0))
plt.imshow(cvimg3)
plt.show()


# exercise 1.3.2

gif = []

# count elements in folder
# folder path
dir_path = r'C:\Users\michi\Documents\DIP\Lab_1\pics\pics\brain'
# Iterate directory
for file_name in os.listdir(dir_path):
    # check if current path is a file
    if file_name.startswith('brain'):
        file_path = f'pics/pics/brain/{file_name}'
        info = pd.dcmread(file_path)
        img = info.pixel_array.astype(float)
        img = Image.fromarray(img.astype(float))
        gif.append(img)

    else:
        pass

# save the array as gif
gif[0].save('brain.gif', save_all=True, append_images=gif[1:],
            optimize=True, duration=500)
