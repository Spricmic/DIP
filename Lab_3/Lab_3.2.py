import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import cv2
from scipy import signal

# constants that define which exercise should be executed
exercise_2_1 = True
exercise_2_2 = False

# read picture path
cellssandp_path = 'pictures/cellsSandP.tif'

# convert picture to array
cellssandp_nparr = tf.imread(cellssandp_path)


# definition of matrix for averaging filter
sharpening_filter_1 = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])
sharpening_filter_1 = [[value * (1/8) for value in row] for row in sharpening_filter_1]

sharpening_filter_2 = np.array([[1, 2, 1],
                                [2, 0, 2],
                                [1, 2, 1]])
sharpening_filter_2 = [[value * (1/12) for value in row] for row in sharpening_filter_2]


def apply_matrix_filter(picture_array, filter):
    """
    apply the defined filter to a picture.
    use ignore boarder technic
    :param picture_array: the picture as np.array() to which the filter is applied
    :param filter: the filter matrix that should be applied
    :return: returns the new picture as np.array()
    """
    return signal.convolve2d(picture_array, filter)


def apply_filter(picture_array, type_filter='median'):
    """
    applies a defined filter to the picture
    :param picture_array: picture which should be filtered
    :param type_filter: applies max, min or median filter to the picture
    :return: new np.array() of the picture
    """
    return_array = picture_array
    for row in range(1, return_array.shape[0] - 1):
        for col in range(1, return_array.shape[1] - 1):
            neighbors = []

            for i in range(-1, 2):
                for j in range(-1, 2):
                    new_row = row + i
                    new_col = col + j

                    # Check if the new indices are within the matrix bounds
                    if 0 <= new_row < return_array.shape[0] and 0 <= new_col < return_array.shape[1]:
                        neighbors.append(return_array[new_row, new_col])

            # Sort the neighbors and calculate the median
            if type_filter == 'median':
                new_value = np.median(sorted(neighbors))
            elif type_filter == 'max':
                new_value = np.max(sorted(neighbors))
            elif type_filter == 'min':
                new_value = np.min(sorted(neighbors))
            else:
                print('wrong argument passed to apply_filter():\n'
                      f'passed argument: {type_filter}.')
                break
            return_array[row, col] = new_value

    return return_array


def plot_img(image_array, title):
    plt.imshow(image_array, cmap='gray')
    plt.title(f'{title}')
    plt.show()


if __name__ == '__main__':
    if exercise_2_1:
        plot_img(cellssandp_nparr, 'original image')
        plot_img(apply_matrix_filter(cellssandp_nparr, sharpening_filter_1), 'averaging filter 1 applied')
        plot_img(apply_matrix_filter(cellssandp_nparr, sharpening_filter_2), 'averaging filter 2 applied')

    if exercise_2_2:
        plot_img(cellssandp_nparr, 'original image')
        plot_img(apply_filter(cellssandp_nparr, 'median'), 'median filtered')
        plot_img(apply_filter(cellssandp_nparr, 'max'), 'max filtered')
        plot_img(apply_filter(cellssandp_nparr, 'min'), 'min filtered')
        """
        Die Durchgänge mit dem averaging filter dauern wahrnehmbar länger.
        Dies könnte jedoch auch daran liegen, das für den averaging filter eine bibliothek verwendet wurde.
        Diese Ist möglicherweise in C geschriben was einen wahrnehmbaren untercshied in der durchlaufzeit führen würde.
        """

