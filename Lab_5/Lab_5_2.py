import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import segmentation
from scipy import ndimage
from PIL import Image


def plot_img(image_array, title):
    """
    function to plot a np.array as a picture
    :param image_array: np.array of the picture
    :param title: titel of the picture to be displayed
    :return: none (plotted picture)
    """
    image_array = np.abs(image_array)
    plt.imshow(image_array, cmap='gray')
    plt.title(f'{title}')
    plt.show()


def print_histogram(histogram, name):  # name is to declare the source of the histogram
    plt.bar(range(len(histogram)), histogram, width=0.5, color='b')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Grayscale Histogram from: {name}')
    plt.show()


# read picture path
squares_path = 'pics/bloodCells.png'
# Open the image with pillow
image = Image.open(squares_path)
# convert pictures to np.array
image_nparr = np.asarray(image)

#define threshold for th epicture
threshold = 75

# Apply global thresholding (https://www.geeksforgeeks.org/image-thresholding-in-python-opencv/)
_, binary_image = cv2.threshold(image_nparr, threshold, 255, cv2.THRESH_BINARY)

# plot_img(binary_image, "test shit")
# clear the boarder of cells.
img_clr_boarder = segmentation.clear_border(binary_image)

# fill the holes in the picture
img_filld = ndimage.binary_fill_holes(img_clr_boarder)
# plot_img(img_filld, "test")
#lable the array
lable_array, num_features = ndimage.label(img_filld)
# plot_img(lable_array, 'lable print')
print(f"there are {num_features} cells in this picture.")

#create the histogram to determain size of each cell.
cell_hist, bins = np.histogram(lable_array, bins=50, range=(1, 50))  # 256 used for unit8
print_histogram(cell_hist, 'cell histogram')

# Find unique frequencies and their occurrences
unique_frequencies, occurrences = np.unique(cell_hist, return_counts=True)

# Print the results
print("Frequency\tOccurrences")
for freq, count in zip(unique_frequencies, occurrences):
    print(f"{int(freq)}\t\t{int(count)}")

