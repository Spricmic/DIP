import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import cv2
from scipy import signal

# constants that define which exercise should be executed
exercise_1_1 = False
exercise_1_2 = True
exercise_2 = False


# read picture path
bloodcells_path = 'pictures/bloodCells.tif'
xray_path = 'pictures/xRayChest.tif'

# convert picture to array
bloodcells_nparr = tf.imread(bloodcells_path)
xray_nparr = tf.imread(xray_path)


# 1.1.1 create two low pass filter
hm_filter = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
hm_filter = [[value * (1/16) for value in row] for row in hm_filter]

hg_filter = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]])
hg_filter = [[value * (1/16) for value in row] for row in hg_filter]


def apply_low_pass_filter(picture_array, filter):
    """
    apply the defined filter to a picture.
    use ignore boarder technic
    :param picture_array: the picture as np.array() to which the filter is applied
    :param filter: the filter matrix that should be applied
    :return: returns the new picture as np.array()
    """
    return signal.convolve2d(picture_array, filter)


def apply_high_pass_filter(picture_array, filter):
    """
    apply a high pass filter to an image by subtracting the low pass filter from the original picture
    :param picture_array: the picture as np.array() to which the filter is applied
    :param filter: the filter matrix that should be applied
    :return: returns the new picture as np.array()
    """
    low_pass_filtered = signal.convolve2d(picture_array, filter)
    #match the sizes of picture and lowpass filtered image. (lowpass is executed using ignore boarder
    # and is there for in size[-2, -2] smaller then the original picture)
    low_pass_filtered = low_pass_filtered[1:-1, 1:-1]
    return picture_array - low_pass_filtered


def plot_range(pixle_range, title):  # function to plot the 2-D Lists
    plt.bar(range(len(pixle_range)), pixle_range, width=1.0, color='b')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'{title}')
    plt.show()


def plot_img(image_array, title):
    plt.imshow(image_array)
    plt.title(f'{title}')
    plt.show()


def plot_img_uint8(image_array, title):
    #scale all chanells seperatly
    uint8_r = cv2.convertScaleAbs(image_array[:, :, 0], alpha=(255.0 / 65535.0))
    uint8_g = cv2.convertScaleAbs(image_array[:, :, 1], alpha=(255.0 / 65535.0))
    uint8_b = cv2.convertScaleAbs(image_array[:, :, 2], alpha=(255.0 / 65535.0))
    # recombine the chanells to a single image
    image_uint8 = cv2.merge([uint8_b, uint8_g, uint8_r])
    plt.imshow(image_uint8)
    plt.title(f'{title}')
    plt.show()


if __name__ == '__main__':
    if exercise_1_1:
        plot_img(bloodcells_nparr, 'original')
        plot_img(apply_low_pass_filter(bloodcells_nparr, hm_filter), 'hm_filtered, low pass')
        plot_img(apply_low_pass_filter(bloodcells_nparr, hg_filter), 'hg_filtered, low pass')
        plot_img(xray_nparr, 'original')
        plot_img(apply_low_pass_filter(xray_nparr, hm_filter), 'hm_filtered, low pass')
        plot_img(apply_low_pass_filter(xray_nparr, hg_filter), 'hg_filtered, low pass')

    if exercise_1_2:
        plot_img(bloodcells_nparr, 'original')
        plot_img(apply_high_pass_filter(bloodcells_nparr, hm_filter), 'hm_filtered, high pass')
        plot_img(apply_high_pass_filter(bloodcells_nparr, hg_filter), 'hg_filtered, high pass')
        plot_img_uint8(apply_high_pass_filter(bloodcells_nparr, hm_filter), 'hm_filtered, high pass, uint8')
        plot_img_uint8(apply_high_pass_filter(bloodcells_nparr, hg_filter), 'hg_filtered, high pass, uint8')
        plot_img(xray_nparr, 'original')
        plot_img(apply_high_pass_filter(xray_nparr, hm_filter), 'hm_filtered, high pass')
        plot_img(apply_high_pass_filter(xray_nparr, hg_filter), 'hg_filtered, high pass')
        plot_img_uint8(apply_high_pass_filter(xray_nparr, hm_filter), 'hm_filtered, high pass, uint8')
        plot_img_uint8(apply_high_pass_filter(xray_nparr, hg_filter), 'hg_filtered, high pass, uint8')

    if exercise_2:
        pass

