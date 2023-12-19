import numpy as np
import matplotlib.pyplot as plt
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


# read picture path
squares_path = 'pics/squares.tif'
# Open the image with pillow
image = Image.open(squares_path)
# convert pictures to np.array
image_nparr = np.asarray(image)

# get size of the image
nRows = image_nparr.shape[0]
nCols = image_nparr.shape[1]

# define the kernals
ker5 = np.ones((5, 5))  # to delete all elements smaller than 5x5
ker6 = np.ones((6, 6))  # to delete all elements bigger than 5x5

eroded5 = ndimage.binary_erosion(image_nparr, structure=ker5)  # erroding by 5x5 to remove smaller
im5 = ndimage.binary_dilation(eroded5, structure=ker5)  # dillate remaining to original size
eroded6 = ndimage.binary_erosion(image_nparr, structure=ker6)  # erroding by 6x6 to remove bigger
im6 = ndimage.binary_dilation(eroded6, structure=ker6)  # dillate remaining to original size

# subtract from the squares >5 the squares >6 to get only squares in shape 5x5
only5 = np.bitwise_xor(im5, im6)

# Iterate over the entire image and detect segments
segment_array = []
input_image = only5
while np.sum(input_image) > 0:
    search = True
    segment = np.zeros((nRows, nCols), dtype=bool)
    for i in range(nRows):
        for j in range(nCols):
            if input_image[i, j] and search:  # find the seed pixel
                search = False
                segment[i, j] = 1
            if (input_image[i, j] and segment[i - 1, j]  # find connected pixels and
                    or input_image[i, j] and segment[i + 1, j]  # add them to the segment
                    or input_image[i, j] and segment[i, j - 1]
                    or input_image[i, j] and segment[i, j + 1]):
                segment[i, j] = 1
    segment_array.append(segment)  # append the found segment to the array
    input_image = np.bitwise_xor(input_image, segment)  # subtract the detected segment from the input

#plot_img(image_nparr, "nparray")
fig0 = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('original')
plt.imshow(image_nparr, cmap='gray')

#plot_img(image_nparr, "smaller or equal to 5")
plt.subplot(2, 2, 2)
plt.title('smaller or equal to 5')
plt.imshow(im5, cmap='gray')

#plot_img(image_nparr, "smaller or equal to 6")
plt.subplot(2, 2, 3)
plt.title('smaller or equal to 6')
plt.imshow(im6, cmap='gray')

#plot_img(image_nparr, "resulting image")
plt.subplot(2, 2, 4)
plt.title('resulting image')
plt.imshow(only5, cmap='gray')

plt.tight_layout()
for ax in fig0.get_axes():
    ax.axis('off')
plt.show()

