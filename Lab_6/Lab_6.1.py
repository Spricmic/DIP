import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# constants that define which exercise should be executed
exercise_1_1 = False
exercise_1_2 = False
exercise_1_3 = False
exercise_1_4 = False

# read picture path
artery_path = 'pics/pics/arterie.tif'
ctskull_path = 'pics/pics/ctSkull.tif'

# read pictures
artery = Image.open(artery_path)
ctskull = Image.open(ctskull_path)

# convert pictures to np.array
artery_nparr = np.asarray(artery)
ctskull_nparr = np.asarray(ctskull)


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


def greyscale_to_color(greyscale_pic_nparray):
    """
    This function transforms assigness a grayscale picture color based on the value of a pixle.
    There is probably a faster way to apply this using a CDF.
    :param greyscale_pic_nparray: a greyscale picture converted to a np.array
    :return: a three dimensional matrix with the RGB value for each pixle
    """
    # Initialisiere leere RGB-Arrays mit den gleichen Dimensionen wie das Graustufenbild
    r_array = np.zeros_like(greyscale_pic_nparray)
    g_array = np.zeros_like(greyscale_pic_nparray)
    b_array = np.zeros_like(greyscale_pic_nparray)
    histogram, bins = np.histogram(greyscale_pic_nparray, bins=256, range=(0, 256))  # 256 für uint8


    colormap = np.array([[255, 0, 0],
                         [255, 0, 128],
                         [128, 0, 255],
                         [0, 64, 255],
                         [0, 255, 255],
                         [0, 255, 64],
                         [128, 255, 0],
                         [255, 255, 0]])

    # Berechne die Anzahl der Pixel in deinem Bild
    pixels_in_picture = greyscale_pic_nparray.size

    # Berechne, wie viele Pixel auf eine Farbe entfallen sollten
    pixels_per_color = pixels_in_picture / colormap.shape[0]

    colormap_count = 0
    pixel_count = 0
    histogram_index = 0
    for value in histogram:
        if value > 0:
            if pixel_count < pixels_per_color:
                row_count = 0
                for row in greyscale_pic_nparray:
                    column_count = 0
                    for pixel in row:
                        if pixel == histogram_index:
                            r_array[row_count][column_count] = colormap[colormap_count][0]
                            g_array[row_count][column_count] = colormap[colormap_count][1]
                            b_array[row_count][column_count] = colormap[colormap_count][2]
                            pixel_count += 1
                        else:
                            pass
                        column_count += 1
                    row_count += 1

            else:
                pixel_count = 0
                colormap_count += 1
            histogram_index += 1

        else:
            histogram_index += 1


    # Erstelle ein 3D-Array für das Farbbild und fülle es mit den RGB-Daten
    color_pic_nparray = np.stack([r_array, g_array, b_array], axis=-1)

    print('There is probably a faster way to apply this using a CDF.')
    print('So one has not to itterrate trough all the pixles.')
    return color_pic_nparray


if __name__ == '__main__':
    if exercise_1_1:
        greyscale_artery = artery.convert('L')
        greyscale_ctskull = ctskull.convert('L')
        greyscale_artery_nparr = np.asarray(greyscale_artery)
        greyscale_artery_ctskull = np.asarray(greyscale_ctskull)
        artery_hist, bins = np.histogram(greyscale_artery_nparr, bins=256, range=(0, 256))  # 256 used for unit8
        ctskull_hist, bins = np.histogram(greyscale_artery_ctskull, bins=256, range=(0, 256))  # 256 used for unit8

        # print histogram for artery
        print_histogram(artery_hist, 'histogram artery')
        plot_img(greyscale_artery_nparr, 'greyscale artery')  # plot greyscale picture
        # print histogram for ctSkull
        print_histogram(ctskull_hist, 'histogram ctSkull')
        plot_img(greyscale_artery_ctskull, 'greyscale ctSkull')  # plot greyscale picture

    elif exercise_1_2:
        print("""
        Define a color map:
        1 -> [255, 0, 0]  red
        2 -> [255, 0, 128]
        3 -> [128, 0, 255]
        4 -> [0, 64, 255]  blue
        5 -> [0, 255, 255]
        6 -> [0, 255, 64]
        7 -> [128, 255, 0]  green
        8 -> [255, 255, 0]
        """)

    elif exercise_1_3:
        print("""
        k = amount of colors = 8
        n = amount of pixles
        1 -> [255, 0, 0]    -> 0 =< n/k
        2 -> [255, 0, 128]  -> n/k =< 2n/k
        3 -> [128, 0, 255]  -> 2n/k =< 3n/k
        4 -> [0, 64, 255]   -> 3n/k =< 4n/k
        5 -> [0, 255, 255]  -> 4n/k =< 5n/k
        6 -> [0, 255, 64]   -> 5n/k =< 6n/k
        7 -> [128, 255, 0]  -> 6n/k =< 7n/k
        8 -> [255, 255, 0]  -> 7n/k =< k
        for each color assigne n/k pixles.
        switch to next color only when a specific greyscale was completly assigned.
        do NOT switch in the itteration after n/k was reached.
        with this the last color assigned will be underreprecented but the other values
        should appear in similar frequency.
        """)

    elif exercise_1_4:

        # read picture as greyscale and create histogram
        greyscale_artery = artery.convert('L')
        greyscale_ctskull = ctskull.convert('L')
        greyscale_artery_nparr = np.asarray(greyscale_artery)
        greyscale_ctskull_nparr = np.asarray(greyscale_ctskull)

        colored_artery = greyscale_to_color(greyscale_artery_nparr)
        colored_ctskull = greyscale_to_color(greyscale_ctskull_nparr)


        plt.imshow(colored_artery)
        plt.title("artery colored")
        plt.show()

        plt.imshow(colored_ctskull)
        plt.title("CTSkull colored")
        plt.show()

    else:
        print('No exercise was selected!\n'
              'Set a exercise to True!')
