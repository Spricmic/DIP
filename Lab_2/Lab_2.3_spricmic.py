import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import cv2
from skimage import exposure, io

bloodcells_path = 'pictures/bloodCells.tif'
ctskull_path = 'pictures/ctSkull.tif'
xray_path = 'pictures/xRayChest.tif'

# read the tif images as np.array
bloodcells_nparr = tf.imread(bloodcells_path)
ctskull_nparr = tf.imread(ctskull_path)
xray_nparr = tf.imread(xray_path)

# pack tif files in itterable list
np_arr_pic = [bloodcells_nparr, ctskull_nparr, xray_nparr]


def create_histogram(tif_array):
    histogram_np, bins = np.histogram(tif_array, bins=256, range=(0, 256))  # return histogramlist for uint8 pictures
    return histogram_np  # return histogram as array


def create_density(histogram):  # creates densitifunction of histogram and equalises it to one
    density_count = 0  # keeps track of the summ
    density = []  # stores the density values
    for element in histogram:  #summs up all the values in the histogram
        density_count += element
        density.append(density_count)

    max_value = np.max(density)  # get highest value in list dor division
    index_count = 0
    for element in density:  # devides all values in list by the highest number
        density[index_count] = element / max_value
        index_count += 1
    return density


def plot_range(pixle_range, title):  # function to plot the 2-D Lists
    plt.bar(range(len(pixle_range)), pixle_range, width=1.0, color='b')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'{title}')
    plt.show()


def auto_equalization(histogram, cdf_equalized):
    """
    Function to automaticly adjust the histogram.
    this function spreads the pixel values more evenly apart to achiev higher contrast in the picture.
    :param list histogram: histogram of th greyscale picture.
    :param list cdf_equalized: density functino of the histogram equalized to y-values 0-1
    """
    if len(histogram) == len(cdf_equalized):
        equalized_histogram = [0] * len(histogram)  # init list with only 0 and same size as histogram
        index_counter = 0
        for value in histogram:
            new_index = int(np.floor(cdf_equalized[index_counter] * 255))
            equalized_histogram[new_index] = value
            index_counter += 1

        return equalized_histogram

    else:  # print this to see where the error was
        print(f'histogram and cdf have different lengths:\n'
              f'histogram: {len(histogram)}\n'
              f'cdf_equalizet: {len(cdf_equalized)}\n')


if __name__ == '__main__':
    histogram_cells = create_histogram(bloodcells_nparr)
    plot_range(histogram_cells, 'histogram_self')
    density_cells = create_density(histogram_cells)
    plot_range(density_cells, 'density_self')
    histogram_cells_equ = auto_equalization(histogram_cells, density_cells)
    plot_range(histogram_cells_equ, 'equalized_histogram_self')

    # scikit-image equalization and plot
    equalized_image = exposure.equalize_hist(bloodcells_nparr)
    equalized_hist, _ = np.histogram(equalized_image, bins=256, range=(0, 1))
    plot_range(equalized_hist, 'equalized_histogram_scikit')

    # cv2 equalization and plot
    equalized_image = cv2.imread('pictures/bloodCells.tif', cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(equalized_image)
    plot_range(equalized_hist, 'equalized_histogram_cv2')

