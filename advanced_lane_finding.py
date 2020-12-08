# Import Packages
import glob  # OS dependencies to get file system details
import cv2
# importing some useful packages
import pickle
import matplotlib.pyplot as plt
import numpy as np


def show_images(images, gray=None, divider=2):
    """
    This is an utility function to show multiple images with different colour maps

    :param images - An images list
    :param gray - A flag to set default value for matplotlib imshow colour map. If the image
                  shape is 2( i.e binary image) then cmap value will be "gray"
    :return: Nothing
    """
    rows = (len(images) + 1) // divider
    plt.figure(figsize=(16, 16))
    for idx, img in enumerate(images):
        plt.subplot(rows, divider, idx + 1)
        # if the image is binary then it'll be printed as grayscale, otherwise colour map
        # will be ignored
        plt.imshow(img, cmap="gray" if len(img.shape) == 2 else gray)
        plt.xticks([])
        plt.yticks([])

    plt.show()


# Loading test images from test_image directory
camera_cal_imgs = [plt.imread(path) for path in glob.glob("camera_cal/*")]

# Visualize calibration images
show_images(camera_cal_imgs, divider=4)


def grayscale(img, opencv_read=False):
    """

    :param img:
    :param opencv_read:
    :return:
    """
    if opencv_read:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# prepare objects points
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and iamg points from all the images
objpoints = []
imgpoints = []


def find_and_draw_chessboard(img, idx, axs, pattern_size=(9, 6)):
    gray = grayscale(img)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # if found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        axs[idx].axis('off')
        axs[idx].imshow(img)


# Draw subplots dynamically
fig, axs = plt.subplots(5, 4, figsize=(16, 16))
axs = axs.ravel()

for idx, img in enumerate(camera_cal_imgs):
    find_and_draw_chessboard(img, idx, axs)

### Calibrating Camera

# cache an image to further reuse
sample_img = cv2.imread("camera_cal/calibration1.jpg")
# cache image size to further reuse
img_size = sample_img.shape[:2]

# Do Camera calibration given objects' points and images' points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the Camera calibration results for later use
dist_pickle = {"mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open("resources/calibration.p", "wb"))


### Undistort Images

def undistort(img, mtx, dist):
    """

    :param img:
    :param mtx:
    :param dist:
    :return:
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def undistort_image(img, cmatrix, distc):
    """

    :param sample_img:
    :param cmatrix:
    :param distc:
    :return:
    """
    udistord_img = undistort(img, cmatrix, distc)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=18)
    ax2.imshow(udistord_img)
    ax2.set_title('Undistorted Image', fontsize=18)
    # this can be used
    return udistord_img


res = undistort_image(sample_img, mtx, dist)
test_images = [plt.imread(path) for path in glob.glob("test_images/*")]

# undistord images
undistort_images = list(map(lambda img: undistort_image(img, mtx, dist), test_images))


### Perspective Transform

def corners_unwarp(img, src, dst):
    """

    :param img: input image
    :param src: source
    :param dst: destination
    :return:
    """
    M = cv2.getPerspectiveTransform(src, dst)  # magnitute
    Minv = cv2.getPerspectiveTransform(dst, src)
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped, Minv, M


#### Select source and destination from images

height, width = test_images[0].shape[:2]
# source points
p1 = (575, 465)
p2 = (705, 465)
p3 = (255, 685)
p4 = (1050, 685)
line_color = (0, 255, 0)  # Green

# destination points
pd1 = (450, 0)
pd2 = (width - 450, 0)
pd3 = (450, height)
pd4 = (width - 450, height)


def draw_polygon_on_image(img, line_color=(0, 255, 0)):
    """

    :param img:
    :return:
    """
    cv2.line(img, p1, p2, line_color, 3)
    cv2.line(img, p2, p4, line_color, 3)
    cv2.line(img, p4, p3, line_color, 3)
    cv2.line(img, p3, p1, line_color, 3)

    return img


#### Warped source images before warp

src_selected_images = list(map(lambda img: draw_polygon_on_image(img), test_images))
show_images(src_selected_images)


def visualize_warped_images(img, src, dst):
    unwarped, _, _ = corners_unwarp(img, src, dst)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    img = draw_polygon_on_image(img)
    ax1.imshow(img)
    ax1.set_title('Undistorted Image', fontsize=18)
    ax2.imshow(unwarped)
    ax2.set_title('Unwarped Image', fontsize=18)
    return unwarped


src = np.float32([p1, p2, p3, p4])
dst = np.float32([pd1, pd2, pd3, pd4])
#
warped_images = list(map(lambda img: visualize_warped_images(img, src, dst), undistort_images))


#### Colour channels

def apply_color_filter(uwimg):
    unwarp_R = uwimg[:, :, 0]
    unwarp_G = uwimg[:, :, 1]
    unwarp_B = uwimg[:, :, 2]
    unwarp_HSV = cv2.cvtColor(uwimg, cv2.COLOR_RGB2HSV)
    unwarp_H = unwarp_HSV[:, :, 0]
    unwarp_S = unwarp_HSV[:, :, 1]
    unwarp_V = unwarp_HSV[:, :, 2]
    fig, axs = plt.subplots(1, 6, figsize=(16, 16))
    axs = axs.ravel()
    axs[0].imshow(unwarp_R, cmap='gray')
    axs[0].set_title('RGB R-channel', fontsize=12)
    axs[1].imshow(unwarp_G, cmap='gray')
    axs[1].set_title('RGB G-Channel', fontsize=12)
    axs[2].imshow(unwarp_B, cmap='gray')
    axs[2].set_title('RGB B-channel', fontsize=12)
    axs[3].imshow(unwarp_H, cmap='gray')
    axs[3].set_title('HSV H-Channel', fontsize=12)
    axs[4].imshow(unwarp_S, cmap='gray')
    axs[4].set_title('HSV S-channel', fontsize=12)
    axs[5].imshow(unwarp_V, cmap='gray')
    axs[5].set_title('HSV V-Channel', fontsize=12)


for img in warped_images:
    apply_color_filter(img)


#### Thresolding

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = grayscale(img)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    dx = 1 if orient == 'x' else 0
    dy = 1 if orient == 'y' else 0

    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, None)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_sobel


def apply_sobel_threshold(unwarp_img, min_thresh=0, max_thresh=255):
    """

    :param unwarp_img:
    :param min_thresh:
    :param max_thresh:
    :return:
    """
    cv2.imshow("temp", unwarp_img)
    cv2.waitKey(0)
    abs_sobel = abs_sobel_thresh(unwarp_img, 'x', min_thresh, max_thresh)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    ax1.imshow(unwarp_img)
    ax1.set_title('Unwarped Image', fontsize=18)
    ax2.imshow(abs_sobel)
    cv2.imshow("temp", abs_sobel)
    cv2.waitKey(0)
    ax2.set_title('Sobel Absolute', fontsize=18)

min_thresh = 20
max_thresh = 100

for img in warped_images[:1]:
    apply_sobel_threshold(img)
