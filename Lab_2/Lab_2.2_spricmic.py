import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from skimage import exposure

ctskull_path = 'pictures/ctSkull.tif'
xray_path = 'pictures/xRayChest.tif'

# read the tif images as np.array
ctskull_nparr = tf.imread(ctskull_path)
xray_nparr = tf.imread(xray_path)


# gamma correction function
def gamma_correction(pixel_array, gamma):
    new_pixel_array = pixel_array.astype(np.float16) / 255.0
    new_pixel_array = np.power(new_pixel_array, gamma)
    plt.imshow(new_pixel_array, cmap='gray')
    plt.title(f'self -> Gammavalue: {gamma}')
    plt.show()


def scikit_gamma(pixel_array, gamma):
    new_pixel_array = exposure.adjust_gamma(pixel_array, gamma)
    plt.imshow(new_pixel_array, cmap='gray')
    plt.title(f'scikit-image -> Gammavalue: {gamma}')
    plt.show()


if __name__ == '__main__':
    gamma_list = [0.125, 0.25, 0.5, 1, 2, 5, 10]
    for gamma_value in gamma_list:
        gamma_correction(ctskull_nparr, gamma_value)
        scikit_gamma(ctskull_nparr, gamma_value)
        gamma_correction(xray_nparr, gamma_value)
        scikit_gamma(xray_nparr, gamma_value)
