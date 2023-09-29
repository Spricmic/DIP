import matplotlib.pyplot as plt
from skimage import io
from PIL import Image

file_grey = 'pics/pics/lena_gray.gif'
file_color = 'pics/pics/lena_color.gif'

#read images with skimage
img_c_ski = io.imread(file_color)
img_g_ski = io.imread(file_grey)

#read image with pillow
img_c_pil = Image.open(file_color)
img_g_pil = Image.open(file_grey)


print('grey')
print(f'size: {img_g_ski.size}')
print(f'shape: {img_g_ski.shape}')
print(f'dtyp: {img_g_ski.dtype}')
print(img_g_ski)
print('******************')
print('color')
print(f'size: {img_c_ski.size}')
print(f'shape: {img_c_ski.shape}')
print(f'dtyp: {img_c_ski.dtype}')
print(img_c_ski)
print('\n\n\n')
print('grey')
print(f'size: {img_g_pil.size}')
# print(f'shape: {img_g_pil.shape}') no shape to this img
# print(f'dtyp: {img_g_pil.dtype}') no dtype to this img
print(img_g_pil)
print('******************')
print('color')
print(f'size: {img_c_pil.size}')
# print(f'shape: {img_c_pil.shape}') no shape to this img
# print(f'dtyp: {img_c_pil.dtype}') no dtype for this img
print(img_c_pil)


#show pictures from pillow using .show()
img_g_pil.show()
img_c_pil.show()


#read color image and refactor in RGB chanells
img_c_pil = Image.open(file_color).convert('RGB')

#split to 3 channels
r, g, b = img_c_pil.split()

#plot the new img as factor of its RGB value
r.show() # red values black white
g.show() # green values black white
b.show() # blue values black white

#convert the image to greyscale
bw_img = img_c_pil.convert('L')
bw_img.show()
