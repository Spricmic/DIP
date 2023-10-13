import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy import fft
from scipy import signal

# constants that define which exercise should be executed
exercise_1_1 = True
exercise_1_2 = False
exercise_1_3 = False
exercise_1_4 = False
exercise_1_5 = True
exercise_1_6 = True
exercise_1_7 = True
exercise_1_8 = True

# read picture path
men_in_desert_path = 'pictures/MenInDesert.jpg'

# convert picture to greyscale array
man_in_desert = Image.open(men_in_desert_path)
greyscale_desert = man_in_desert.convert('L')
man_in_desert_nparr = np.asarray(greyscale_desert)


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


def plot_fft(fft_array, title):
    """
    function to plot a picture on which fft has been used
    :param fft_array: np.array of the fft tranformed picture
    :param title: titel of the fft to be displayed
    :return: none (plotted fft as picture)
    """
    plt.imshow(np.abs(fft_array), cmap='gray')
    plt.title(f'{title}: linear scale')
    plt.show()
    plt.imshow(np.log(np.abs(fft_array)), cmap='gray')
    plt.title(f'{title}: log scale')
    plt.show()


if __name__ == '__main__':
    if exercise_1_1:
        print(f'Dimension Men in desert: {man_in_desert_nparr.shape}')
        plot_img(man_in_desert_nparr, 'Men in desert')
        """
        The picture has shape (552, 736, 3)
        """

    if exercise_1_2:
        fft_array = fft_greyscale(man_in_desert_nparr)
        """
        the transformed pictures have the same shape as the original.
        the datatype of the returned array is a np.array with corresponding complex numbers (complex128).
        if in the function fft2() the argument s is added the picture will be padded with 
        0 to match the defined size.
        """

    if exercise_1_3:
        fft_array = fft.fft2(man_in_desert_nparr)
        plot_fft(fft_array, 'fft image')

    if exercise_1_4:
        plot_img(man_in_desert_nparr, 'original')
        fft_array = fft.fft2(man_in_desert_nparr)
        retransformed_pic = fft.ifft2(fft_array)
        print(f'picture type: {type(retransformed_pic)}')
        print(f'data type: {type(retransformed_pic[0, 0])}')
        print(f'picture shape: {retransformed_pic.shape}')
        plot_img(retransformed_pic, 'retransformed')
        """
        The original data can be obtained by using np.abs() on the complex128 numbers.
        This will yield the magnitude of the complex number which than can be ploted.
        """

    if exercise_1_5:
        # definition of matrix for averaging filter
        averaging_filter = np.ones((9, 9))
        averaging_filter = [[value * (1 / 81) for value in row] for row in averaging_filter]
        averaging_filter = np.array(averaging_filter)
        # filter the given image
        filtered_img = signal.convolve2d(man_in_desert_nparr, averaging_filter)
        plot_img(filtered_img, 'averaging filterd image')

    if exercise_1_6:
        """
        this exercise also contains exercise 1.7 and 1.8
        """
        # create a padded array to place the
        size_picture = man_in_desert_nparr.shape
        size_filter = averaging_filter.shape
        padded_struct = np.zeros((np.min(size_picture) + np.min(size_filter), np.max(size_picture) + np.max(size_filter)))
        # create the filter with padded structure
        padded_filter = padded_struct.copy()
        padded_filter[0:size_filter[0], 0:size_filter[1]] = averaging_filter
        fft_filter = np.array(fft.fft2(padded_filter))
        # create picture with padded structure
        padded_picture = padded_struct.copy()
        padded_picture[0:size_picture[0], 0:size_picture[1]] = man_in_desert_nparr
        fft_picture = np.array(fft.fft2(padded_picture))
        # apply filter to picture
        filtered_pic = fft_picture * fft_filter

        retransformed_pic = fft.ifft2(filtered_pic)
        # Take the real part and scale it to the original size
        retransformed_pic = np.real(retransformed_pic)[0:size_picture[0], 0:size_picture[1]]
        plot_img(retransformed_pic, 'frequency filtered picture')

    if exercise_1_7:
        """
        image is the same expect for the boarder that was created while using the convolution method with zero padding.
        """
        pass

    if exercise_1_8:
        """
        applying the filter using the frequencye domain is much faster especialy for bigger pictures and
        filter kernels.
        interessting would also be the comparison between fft applied filter and using seperable spatial filters,
        as the separable spatial filters are faster then the non seperable filter kernels.
        """
        pass
