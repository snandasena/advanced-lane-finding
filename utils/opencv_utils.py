import cv2
import glob
import pickle

test_images = [cv2.imread(path) for path in glob.glob("../test_images/*")]

sample_img = test_images[0]
cv2.imshow("input", sample_img)
cv2.waitKey(0)
dist_pickle = pickle.load(open("../calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

p1 = (575, 465)
p2 = (705, 465)
p3 = (255, 685)
p4 = (1050, 685)
c = (0, 255, 0)

for idx, img in enumerate(test_images):
    cv2.line(img, p1, p2, c, 3)
    cv2.line(img, p2, p4, c, 3)
    cv2.line(img, p4, p3, c, 3)
    cv2.line(img, p3, p1, c, 3)

    cv2.imshow(str(idx), img)
    cv2.waitKey(0)

h, w = sample_img.shape[:2]
pd1 = (450, 0)
pd2 = (w - 450, 0)
pd3 = (450, h)
pd4 = (w - 450, h)


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


for idx, img in enumerate(test_images):
    res = undistort(sample_img, mtx, dist)
    cv2.line(res, pd1, pd2, c, 3)
    cv2.line(res, pd2, pd4, c, 3)
    cv2.line(res, pd4, pd3, c, 3)
    cv2.line(res, pd3, pd1, c, 3)

    cv2.imshow("res {}".format(idx), res)
    cv2.waitKey(0)

cv2.destroyAllWindows()
