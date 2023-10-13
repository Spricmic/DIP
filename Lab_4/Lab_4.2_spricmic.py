import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, fft
from PIL import Image
from pathlib import Path

# ----------------------------------------------------- load image ---------------------------------------------
file_name = Path('pictures/MenInDesert.jpg')

# Open the image with Pillow
image = Image.open(file_name)

# Convert the image to color and grayscale NumPy arrays
color_pixels = np.asarray(image)
gray_pixels = np.asarray(image.convert('L'))

# summarize some details about the image
print(image.format)
print('numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)

# -------------------------------------------- generate the motion blur filter -----------------------------------------
nFilter = 91
angle = 30
my_filter = np.zeros((nFilter, nFilter))
my_filter[nFilter // 2, :] = 1.0 / nFilter
my_filter = scipy.ndimage.rotate(my_filter, angle, reshape=False)

nRows = gray_pixels.shape[0]
nCols = gray_pixels.shape[1]
nFFT = 1024

image_spectrum = scipy.fft.fft2(gray_pixels, (nFFT, nFFT))
filter_spectrum = scipy.fft.fft2(my_filter, (nFFT, nFFT))

modified_image_spectrum = image_spectrum * filter_spectrum
modified_image = scipy.fft.ifft2(modified_image_spectrum)
modified_image = np.real(modified_image)[nFilter:nRows + nFilter, nFilter:nCols + nFilter]

# --------------------------------------------------- reconstruct the image --------------------------------------------
# here goes your code ...


# get size of picture and filter
picture_size = modified_image.shape
filter_size = my_filter.shape

# get padding params P and Q
padd_para_p = picture_size[0] + filter_size[0]
padd_para_q = picture_size[1] + filter_size[1]

# use fft2 on filter and picture with padding.
fft_filter = np.array(fft.fft2(my_filter, (padd_para_p, padd_para_q)))
fft_picture = np.array(fft.fft2(modified_image, (padd_para_p, padd_para_q)))

# define wiener filter for image
k = 0.008
wiener_filter = np.conj(fft_filter) / (np.abs(fft_filter) ** 2 + k)
"""
good values for the wiener filter for this exaample lie in the range of 0.005 < x > 0.015.
the correction is not perfect. maybe the filter is not completly correct implemented. 
I could not find an error any more.
"""

deblurred_image_spectrum = wiener_filter * fft_picture
deblurred_imag = np.real(fft.ifft2(deblurred_image_spectrum))

# --------------------------------------------------------- display images ---------------------------------------------
fig = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_pixels, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Motion Blur Filter')
plt.imshow(my_filter, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Modified Image')
plt.imshow(modified_image, cmap='gray')

plt.subplot(2, 2, 4)
plt.title(f'Reconstructed Image with K ={k}')
# here goes your reconstructed image
plt.imshow(deblurred_imag[0:picture_size[0], 0:picture_size[1]], cmap='gray')
plt.show()
