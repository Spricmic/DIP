import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

bloodcells_path = 'pictures/bloodCells.tif'
ctskull_path = 'pictures/ctSkull.tif'
xray_path = 'pictures/xRayChest.tif'

# read the tif images as np.array
bloodcells_nparr = tf.imread(bloodcells_path)
ctskull_nparr = tf.imread(ctskull_path)
xray_nparr = tf.imread(xray_path)

np_arr_pic = [bloodcells_nparr, ctskull_nparr, xray_nparr]


# compute the grey level histogram of a np.arrray
def comp_histogr(tif_array):
    max_pixel_value = np.max(tif_array)
    if max_pixel_value <= 255:
        histogram = [0] * 256  # empty list for uint8
    else:
        histogram = [0] * 65536  # empty list for uint16

    for pixel_value in tif_array.ravel():  # flaten the picture to 1D-array with .ravel then count the values
        histogram[pixel_value] += 1

    return histogram


# print the histogram on the console
def print_histogram(histogram, name): # name is to daclare the source of the histogram
    plt.bar(range(len(histogram)), histogram, width=1.0, color='b')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Grayscale Histogram from: {name}')
    plt.show()


if __name__ == '__main__':
    for element in np_arr_pic:
        print(f'shape: {element.shape}')
        print(f'max_value: {np.max(element)}')
        histogram_self = comp_histogr(element)
        histogram_np, bins = np.histogram(element, bins=256, range=(0, 256))
        print_histogram(histogram_self, 'self')
        print_histogram(histogram_np, 'np.histogram')
