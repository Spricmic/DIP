import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
from numpy.linalg import eig

##### Aspect Ratio and Extent #####

# 1. Read image
img_bone = plt.imread("C:/Users/michi/Documents/ZHAW_git/Semester_5/DIP/DIP/Lab_9/handout_files/aspectRatio/bone.tif")
img_bone_gray = cv2.cvtColor(img_bone, cv2.COLOR_RGBA2GRAY)
img_bone_gray = img_bone_gray.astype(np.float64) / np.max(img_bone_gray)
plt.figure()
plt.imshow(img_bone_gray, cmap='gray')
plt.show()

# 2. Create a matrix A with the coordinates of the pixels that belong to the bone and whose centroid is in the origin
points = np.argwhere(img_bone_gray > 0.5)
# Note that the matrix points have the dimension (number of image points x 2)
#    points[:,0] ... y coordinates
#    points[:,1] ... x coordinates
# Change the order so that 
#    points[:,0] ... x coordinates
#    points[:,1] ... y coordinates
points = points[:, ::-1]  # coordinates of the pixels belonging to the bone[x,y]

sum_x = np.sum(points[:, 0])
sum_y = np.sum(points[:, 1])
pixel_weight = np.shape(points)[0]

centroid = np.round([sum_x / pixel_weight, sum_y / pixel_weight])
print(f'centroid-> x:{centroid[0]}  , y:{centroid[1]}')

# recenter image
recentered_image = points - centroid

# compute covariance matrix
covarianceMatrix = np.cov(recentered_image.T)
# print(covarianceMatrix)

# 3. Compute the eigen vector for the largest eigen value
D, V = eig(covarianceMatrix)
idx = np.argsort(D)[-1]  # Index of largest eigenvalue
eigV = V[:, idx]  # Eigen vector for largest eigen value
# print(eigenV)

# 4. Compute rotation that aligns the bone's dominant axis to the image's x-axis
height, width = img_bone_gray.shape
# 4.a) Method using the explicit rotation angle phi
phi = np.degrees(np.arctan2(eigV[1], eigV[0]))
rotMatAffine = cv2.getRotationMatrix2D(tuple(centroid), phi, 1)

# 5. Rotate the image
img_bone_rot = cv2.warpAffine(img_bone_gray, rotMatAffine, (width, height), cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

# 6. Determine the bounding box for the aligned bone
alignedBoneCoords = np.argwhere(img_bone_rot > 0.5)
y1, x1 = alignedBoneCoords.min(axis=0)
y2, x2 = alignedBoneCoords.max(axis=0) + 1

# 7. Draw the bonding box onto the image
img_bone_rot = cv2.rectangle(img_bone_rot.copy(), (x1, y1), (x2, y2), color=1)

height = y2 - y1 + 1
width = x2 - x1 + 1
print('Bounding Box: width={}, height={}'.format(width, height))

# 8. Plot the aligned object bone including bounding box
plt.figure()
plt.imshow(img_bone_rot, cmap='gray')
plt.title('Rotated Object with Boundig Box')
plt.show()


###### Texture and Co-Ocurrence matrix #####


# Computation of the Co-Ocurrence Matrix
def compute_co_occurrence_matrix(image):
    cooMat = np.zeros((256, 256))

    M, N = image.shape

    count = 1
    for ii in range(1, M - 1):
        for jj in range(1, N - 1):
            p = image[ii, jj]
            q_up = image[ii, jj + 1]
            q_right = image[ii + 1, jj]
            q_down = image[ii, jj - 1]
            q_left = image[ii - 1, jj]

            # Co-occurrence matrix berechnung
            cooMat[p, q_up] += 1
            cooMat[p, q_right] += 1
            cooMat[p, q_down] += 1
            cooMat[p, q_left] += 1

            count = count + 4

    cooMat = cooMat / count
    #print(cooMat)
    return cooMat


def compute_texture_features(cooMat):
    energy = np.sum(cooMat**2)
    contrast = np.sum((np.arange(256) - np.arange(256)[:, np.newaxis])**2 * cooMat)
    entropy = -np.sum(cooMat * np.log2(cooMat + 1e-10))  # kleine numer um division miit 0 zu verhindern
    homogeneity = np.sum(cooMat / (1 + np.abs(np.arange(256) - np.arange(256)[:, np.newaxis])))

    return energy, contrast, entropy, homogeneity


def plot_result(cooMat, energy, contrast, entropy, homogeneity):
    # Plotting the Co-Ocurrence matrix
    X = np.arange(cooMat.shape[1])
    Y = np.arange(cooMat.shape[0])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, cooMat, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    print("energy: %5.5f\ncontrast: %5.5f\nentropy: %5.5f\nhomogeneity: %5.5f" % (energy, contrast, entropy, homogeneity))


img_mc1 = plt.imread('C:/Users/michi/Documents/ZHAW_git/Semester_5/DIP/DIP/Lab_9/handout_files/musclecells/mc1.tif')
img_mc2 = plt.imread('C:/Users/michi/Documents/ZHAW_git/Semester_5/DIP/DIP/Lab_9/handout_files/musclecells/mc2.tif')
img_mc3 = plt.imread('C:/Users/michi/Documents/ZHAW_git/Semester_5/DIP/DIP/Lab_9/handout_files/musclecells/mc3.tif')
img_mc4 = plt.imread('C:/Users/michi/Documents/ZHAW_git/Semester_5/DIP/DIP/Lab_9/handout_files/musclecells/mc4.tif')

images = [img_mc1, img_mc2, img_mc3, img_mc3]

for image in images:
    print('\n\nNext cell:')
    cooMatrix = compute_co_occurrence_matrix(image)
    energy, contrast, entropy, homogeneity = compute_texture_features(cooMatrix)
    plot_result(cooMatrix, energy, contrast, entropy, homogeneity)

