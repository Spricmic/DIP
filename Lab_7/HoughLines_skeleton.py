import os  # Operating System
import cv2 as cv  # OpenCV
import numpy as np  # numeric computations package
import matplotlib.pyplot as plt  # Bilder und Graphen wie in MALAB plotten
from skimage.feature.peak import peak_local_max
import matplotlib.cm as cm


def invert_lut(lut):
    indexFromValue = np.empty_like(lut)
    for index, value in enumerate(lut):
        indexFromValue[int(np.round(value))] = index
    return indexFromValue


class HoughLines():
    def __init__(self, img):
        self.maximaOnVotingSpace = None
        self.normalVectors = None
        self.imgDimensionX = img.shape[1]
        self.imgDimensionY = img.shape[0]
        self.maxRadius = int(np.sqrt(self.imgDimensionX ** 2 + self.imgDimensionY ** 2))
        self.minRadius = (-1) * self.maxRadius
        self.maxThetaDegrees = int(180)

        self.radiusFromIndex = np.arange(self.minRadius, self.maxRadius + 1)
        # print(self.radiusFromIndex)
        self.indexFromRadius = invert_lut(self.radiusFromIndex)
        self.angleFromIndex = np.arange(0, self.maxThetaDegrees)
        self.indexFromAngle = invert_lut(self.angleFromIndex)

        self.votingSpace = np.zeros((len(self.angleFromIndex), len(self.radiusFromIndex)), int)
        self.normalVectors = self.__compute_normal_vectors__(self.angleFromIndex)  # lines to test for match in point
        self.__vote_from_image__(img)
        self.maximaOnVotingSpace = self.__find_peaks__(self.votingSpace)  # hier werden die koordinaten der zu zeichnenden Linien gespeichert.

    def __compute_normal_vectors__(self, degrees):
        # print(degrees)
        normalVectors = np.empty((len(degrees), 2), dtype=float)
        for index, thetaDegrees in enumerate(degrees):
            thetaRadians = np.radians(thetaDegrees)  # Convert degrees to radians
            n_y = np.cos(thetaRadians)  # Calculate x-component of the normal vector
            n_x = np.sin(thetaRadians)  # Calculate y-component of the normal vector
            normalVectors[index, :] = np.array([n_x, n_y])
        # print(normalVectors)
        return normalVectors


    def __vote_from_pixel__(self, imageCoordinateXY: np.ndarray, greyValue: int):
        for angleDegrees in np.arange(0, self.maxThetaDegrees):
            thetaRadians = np.radians(angleDegrees)
            radius = int(
                round(imageCoordinateXY[0] * np.cos(thetaRadians) + imageCoordinateXY[1] * np.sin(thetaRadians)))
            radiusIndex = self.indexFromRadius[radius]
            angleIndex = self.indexFromAngle[angleDegrees]
            self.votingSpace[angleIndex, radiusIndex] += greyValue

    def __vote_from_image__(self, img):
        for x in np.arange(0, self.imgDimensionX):
            for y in np.arange(0, self.imgDimensionY):
                imageCoordinateXY = np.array([x, y])
                grayValue = img[y, x]
                if grayValue > 0:
                    self.__vote_from_pixel__(imageCoordinateXY, grayValue)

    def __normalize_voting_space__(self):
        self.votingSpace = 255 * self.votingSpace.astype(float) / np.amax(self.votingSpace)

    def __find_peaks__(self, votingSpace):
        # print(peak_local_max(votingSpace, min_distance=3, num_peaks=10, threshold_rel=0.4)[:, ::-1])
        return peak_local_max(votingSpace, min_distance=3, num_peaks=10, threshold_rel=0.4)[:, ::-1]

    def show_peaks_on_voting_space(self):
        drawingImage = self.votingSpace.copy()
        radius = int(self.maxRadius / 50)
        for peak in self.maximaOnVotingSpace:
            center = peak
            cv.circle(drawingImage, center, radius, color=(255, 100, 100), thickness=1)
        plt.figure()
        plt.imshow(drawingImage, cmap="jet")
        plt.title("Voting Space with Peak locations")
        plt.show()

    def show_voting_space(self):
        plt.figure()
        maxVote = np.amax(self.votingSpace)
        plt.imshow(self.votingSpace, cmap="jet", vmin=0, vmax=maxVote)
        plt.title("HoughtLines Voting Space: maxVote:{}".format(maxVote))
        plt.show()

    def __draw_found_line__(self, img, houghPeakCoordinates, halfLength=200, color=(255, 255, 255)):
            rho, theta = houghPeakCoordinates  # Extract theta and rho from the given peak
            a = np.cos(np.radians(theta))
            b = np.sin(np.radians(theta))
            # calculate the position of the point in the real picture
            x0 = a * self.radiusFromIndex[rho]
            y0 = b * self.radiusFromIndex[rho]

            line_direction = theta + 90
            # print(line_direction)
            a_part = np.cos(np.radians(line_direction))
            b_part = np.sin(np.radians(line_direction))

            # line point a
            x1 = round(x0 + halfLength * (a_part))
            y1 = round(y0 + halfLength * (b_part))
            # line point b
            x2 = round(x0 - halfLength * (a_part))
            y2 = round(y0 - halfLength * (b_part))

            startPoint = np.array([x1, y1])
            endPoint = np.array([x2, y2])
            print('start and endpoint:')
            print(startPoint)
            print(endPoint)
            cv.line(img, startPoint.astype(int), endPoint.astype(int), color=color, thickness=1)

    def show_found_lines(self, img):
        if img is not None:
            drawingImage = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = np.zeros((self.imgDimensionY, self.imgDimensionX), dtype=np.uint8)

        cmap = cm.get_cmap('jet')
        nr_lines = self.maximaOnVotingSpace.shape[0]
        for id, m in enumerate(self.maximaOnVotingSpace):
            color = tuple(255 * np.array(cmap(id / (nr_lines))[0:3]))
            print('peak location in theta roh space:')
            print(m)  # m sind die zu zeichnenden peaks im rho theta format.
            self.__draw_found_line__(drawingImage, m, color=color)

        plt.figure()
        plt.imshow(drawingImage)
        plt.title("Found Lines")
        plt.show()


def main():
    # load Image
    imgOriginal = plt.imread('handdrawing.tif')
    img = cv.pyrDown(cv.pyrDown(cv.pyrDown(cv.cvtColor(imgOriginal, cv.COLOR_RGB2GRAY)))).astype(float)
    # Invert Image such that bright pixels represent the lines
    img = np.amax(img) - img
    imgNormalized = ((img.astype(int) * 255) / np.amax(img)).astype(np.uint8)

    # Show image
    plt.figure()
    plt.imshow(imgNormalized, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title("Image to be analyzed with Hough-Lines Algorithm")
    plt.show()

    # HoughVoting
    houghLines = HoughLines(imgNormalized)
    houghLines.__normalize_voting_space__()
    houghLines.show_voting_space()
    houghLines.show_peaks_on_voting_space()
    houghLines.show_found_lines(imgNormalized)


if __name__ == "__main__":
    main()
