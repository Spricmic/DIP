import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy import fft
from scipy import signal

# constants that define which exercise should be executed
exercise_2_1 = False
exercise_2_2 = False
exercise_2_3 = False
exercise_2_4 = True

# read picture path
braincell_path = 'pics/pics/brainCells.tif'

# read pictures
braincell_picture = Image.open(braincell_path)

# convert pictures to np.array
braincell_nparr = np.asarray(braincell_picture)

# Convert to RGB
rgb_image = cv2.cvtColor(braincell_nparr, cv2.COLOR_BGR2RGB)

# Convert to CMY
cmy_image = 255 - rgb_image  # Invert RGB channels to get CMY

# Convert to HSI
hsi_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

if __name__ == '__main__':
    if exercise_2_1:
        # plot original picture for reference
        plt.imshow(braincell_nparr)
        plt.title("original picture for reference")
        plt.show()

        # Split the channels for each color space
        rgb_channels = cv2.split(rgb_image)
        cmy_channels = cv2.split(cmy_image)
        hsi_channels = cv2.split(hsi_image)

        # Display the grayscale channels for each color space
        for i, channel in enumerate(rgb_channels):
            plt.imshow(channel, cmap='gray')
            plt.title("RGB channels")
            plt.show()

        for i, channel in enumerate(cmy_channels):
            plt.imshow(channel, cmap='gray')
            plt.title("CMY channels")
            plt.show()

        for i, channel in enumerate(hsi_channels):
            plt.imshow(channel, cmap='gray')
            plt.title("HSI channels")
            plt.show()


    elif exercise_2_2:
        print(f'For the segmention task the Hue value of the HSI should be used.\n'
              f'In this the color are moste differented from each other.\n'
              f'This is due to that, that de saturation and intensity are their owen\n'
              f'channels.')


    elif exercise_2_3:
        # Define the lower and upper thresholds for segmentation (in HSI space)
        lower_threshold = np.array([0, 50, 50], dtype=np.uint8)  # define color range with this values
        upper_threshold = np.array([25, 255, 255], dtype=np.uint8)  # define color range with this values

        # Create a binary mask by thresholding the HSI image
        mask = cv2.inRange(hsi_image, lower_threshold, upper_threshold)

        # Apply the mask to the original image to segment the colors
        hsi_channels = cv2.split(hsi_image)
        segmented_image = []
        for channel in hsi_channels:
            segmented_channel = cv2.bitwise_and(channel, mask)
            segmented_image.append(segmented_channel)

        color_pic_nparray = np.stack(segmented_image, axis=-1)

        plt.imshow(color_pic_nparray)
        plt.title("HSI channels")
        plt.show()


    elif exercise_2_4:
        # Define the lower and upper thresholds for segmentation (in HSI space)
        lower_threshold = np.array([0, 50, 50], dtype=np.uint8)  # define color range with this values
        upper_threshold = np.array([25, 255, 255], dtype=np.uint8)  # define color range with this values

        # Create a binary mask by thresholding the HSI image
        mask = cv2.inRange(hsi_image, lower_threshold, upper_threshold)

        # Apply the mask to the original image to segment the colors
        hsi_channels = cv2.split(hsi_image)
        segmented_image = []
        for channel in hsi_channels:
            segmented_channel = cv2.bitwise_and(channel, mask)
            segmented_image.append(segmented_channel)

        color_pic_nparray = np.stack(segmented_image, axis=-1)

        grey_segment = color_pic_nparray[:,:,2]
        plt.imshow(grey_segment, cmap='gray')
        plt.title("greymap")
        plt.show()


    else:
        print('No exercise was selected!\n'
              'Set a exercise to True!')
