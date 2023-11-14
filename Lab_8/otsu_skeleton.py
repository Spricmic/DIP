import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image


def basic_thresholding(image):
    t = np.mean(image.flatten()).astype(int)
    print(t)

    # Calculate the basic threshold

    # <-- your input

    bin_img = image > t
    return bin_img, t


def my_otsu(image):
    # creat histogram of the image
    image_hist, bins = np.histogram(image, bins=256, range=(0, 256))  # 256 used for unit8
    pixels_in_image = np.sum(image_hist)
    # dummy variables, replace with your own code:
    current_threshold = np.mean(image.flatten()).astype(int)
    best_threshold = current_threshold
    between_class_variance = 0
    max_between_class_variance = 0
    separability = 0
    mu_foreground = np.sum(image_hist)
    mu_backgorund = 0
    # until here
    for index, value in enumerate(image_hist):
        current_threshold = index
        if value != 0:
            pixels_in_forground = np.sum(image_hist[index:])
            weight_forground = pixels_in_forground / pixels_in_image
            weight_background = 1 - weight_forground
            mu_foreground = mu_foreground - value
            mu_backgorund = mu_backgorund + value
            between_class_variance = weight_background * weight_forground * (mu_backgorund - mu_foreground) ** 2

            if max_between_class_variance < between_class_variance:
                max_between_class_variance = between_class_variance
                best_threshold = current_threshold
    # link to video understandanding OTSU Methode: https://www.youtube.com/watch?v=jUUkMaNuHP8
    binary_image = image > best_threshold
    return binary_image, between_class_variance, best_threshold, separability


def main():
    # image = np.asarray(Image.open("polymersome_cells_10_36.png"))
    image = np.asarray(Image.open("thGonz.tif"))

    # image = np.asarray(Image.open("binary_test_image.png"))
    if len(image.shape) == 3:
        image = image[:, :, 0]

    binary_image, t = basic_thresholding(image)
    print("Basic Thresholding. Output Threshold: " + str(t))

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.title("binary image from basic threshholding")
    plt.show()

    binary_image, between_class_variance, threshold, separability = my_otsu(image)

    print("Otsu's Method. Output Threshold: " + str(threshold))
    print("Separability: " + str(separability))

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.title("binary image from otsu")
    plt.show()


if __name__ == "__main__":
    main()
