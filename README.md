[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## Advanced Lane Finding

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Image processing pipeline
---
Following flow chart will be described how this image processing pipeline development thing happening advanced computer vision techniques.  

![](resources/image-processing-pipeline.svg)

Camera calibration
---

Camera resectioning determines which incoming light is associated with each pixel on the resulting image [[Wikipedia](https://en.wikipedia.org/wiki/Camera_resectioning)]. 

##### Find and draw chessboard

Before calibration images, chessboard was detected from provided chessboard iamges by using OpenCV `cv2.findChessboardCorners` function. Following are the detected chessboard results. Red reactangle highlighted were not detected as chessbords. 

![](resources/calibrated-imgs.png)

Here is the function that was used to find chessboard from images.

```python
# prepare objects points
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and iamg points from all the images
objpoints = []
imgpoints = []

def find_and_draw_chessboard(img, pattern_size= (9,6)):
    gray = grayscale(img)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # if found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    
    # if not found, return same input image
    return img
```       

**Following OpenCV function was used to calibrate above chessboards found images**

```python
# Do Camera calibration given objects' points and images' points
# mtx - camera calibrated matrix
# dist - distortion cofficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)

```
Above `mtx` and `dist` will be used to undistort images in the pipeline.

##### Distortion correction
**Image distortion** occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image; this transformation isn’t perfect. Distortion actually changes what the shape and size of these 3D objects appear to be. So, the first step in analyzing camera images, is to undo this distortion so that you can get correct and useful information out of them.

`cv2.undistort` can be used to corret distortion of images. Following are the sample images to show undistortion

![](resources/undistorted.png)
![](resources/undistort-2.png)

Perspective transformation
---
A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. Aside from creating a bird’s eye view representation of an image, a perspective transform can also be used for all kinds of different view points.

Perspective transfomation can be done using following steps.
##### Step 01: 
Select four source coordation points from given image and these points were used for all other as well with a generic adgusment.  
Selected **source** points:  
`p1 = (575, 465)` `p2 = (705, 465)` `p3 = (255, 685)` `p4 = (1050, 685)`   
Selected **destination** coordinates.  
`pd1 =(450, 0)` `pd2 = (width - 450, 0)` `pd3 = (450, height)` `pd4 = (width - 450, height)`  
Here height and width are recpectively number of rows and number of colomn in an raw image. Following is an image with drawan polygon by using **source** points.

![](resources/selected-points-image.png )

##### Step 02:
Apply OpenCV `cv2.getPerspectiveTransform` function to select a region as **bird-eye** view. Following function was used to warpe raw images.

```python
# define source and destination points for tranform
src = np.float32([p1, p2, p3, p4])
dst = np.float32([pd1, pd2, pd3, pd4])

def unwarp(img, source=src, desctination=dst):
    """
    This is used to select a region from a given undistortion image as bird eye perspective.
    
    :param img - Distortion corrected image
    :param src - source 
    :param dst - destination
    :return - warped image, transform matrix, and inverse
    """
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv
    
```    
Here is the sample result of above function.

![](resources/unwarped-img.png)

Working with colour spaces and channels
---
On road lane line images, we have to detect two major colors such that yellow and white. By seeing bellow color venn diagram we can get some idea about how other colors are making with three major colors. I have tested with four color spaces sush that RGB, HSV, HLS, and LAB. 

![](resources/color-venn-dia.png)

Following functions were used to extract color channels from color spaces.

```python
# These utilitis are used to extract different color channels from different color spaces.

# Extract R,G, and B color channels from RGB color space.
def extract_rgb_color_spaces(uwimg):
    unwarp_R = uwimg[:, :, 0]
    unwarp_G = uwimg[:, :, 1]
    unwarp_B = uwimg[:, :, 2]
    return unwarp_R,unwarp_G,unwarp_B

# Extract H,S, and V color channels from HSV color space.
def extract_hsv_color_spaces(uwimg):
    unwarp_HSV = cv2.cvtColor(uwimg, cv2.COLOR_RGB2HSV)
    unwarp_H = unwarp_HSV[:, :, 0]
    unwarp_S = unwarp_HSV[:, :, 1]
    unwarp_V = unwarp_HSV[:, :, 2]
    return unwarp_H,unwarp_S,unwarp_V

# Extract H,S, and L color channels from HLS color space.
def extract_hls_color_spaces(uwimg):
    unwarp_HLS = cv2.cvtColor(uwimg, cv2.COLOR_RGB2HLS)
    unwarp_HLS_H = unwarp_HLS[:, :, 0]
    unwarp_HLS_L = unwarp_HLS[:, :, 1]
    unwarp_HLS_S = unwarp_HLS[:, :, 2]
    
    return unwarp_HLS_H,unwarp_HLS_L,unwarp_HLS_S
    
# Extract L,A, and B color channels from LAB color space.
def extract_lab_color_spaces(uwimg):
    unwarped_LAB = cv2.cvtColor(uwimg, cv2.COLOR_RGB2Lab)
    unwarp_L = unwarped_LAB[:,:,0]
    unwarp_A = unwarped_LAB[:,:,1]
    unwarp_B = unwarped_LAB[:,:,2]
    return unwarp_L, unwarp_A,unwarp_B
    
```    
I have applied above functions to detect lane lines edges from road images. Following are the results that I got with each color filters. I have tested two different images to detect lane lines.

##### RGB color space results
![](resources/rgb-test-1.png)
![](resources/rgb-test-2.png)

##### HSV color space results
![](resources/hsv-test-1.png)
![](resources/hsv-test-2.png)

##### HLS color space results
![](resources/hsl-test-1.png)
![](resources/hsl-test-2.png)

##### LAB color space results
![](resources/lab-test-1.png)
![](resources/lab-test-2.png)

##### Color spaces conclutions
**LAB** color space **B** was perfomed well to detect yellow lane lines from the roads images. For white lane lines, we have few options to select color channels. **HLS** **L** was perfomed well to detect white lane lines from road images.

With the above conclutions, I select to only two color channels to apply gredient threshold filters. 

### Gradient threshold
When we are detecting edges from road images, there are ohter edges will be detected with Canny edge detection. We have to minimize those unnecessary edge detections. One of the improved version of Canny edge detection is **Sobel operator**[[Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator)].  Applying the Sobel operator to an image is a way of taking the derivative of the image in the x or y direction.

#### Sobel absolute threshold
Following function was used to apply sobel absolute filter

```python
def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    """
    This is used to create abinary image using a grayscaled image. 
    
    :param gray - Grayscaled or binary image
    :param orient - x or y derections of the image
    :param thresh_min - minimum value for binary threshold
    :param thresh_max - maximum value for binary threshold
    :return - A binary image after appying Sobel absolute threshold
    """
    # Apply the following steps to img
    # 1) Take the derivative in x or y given orient = 'x' or 'y'
    dx = 1 if orient=='x'  else  0
    dy = 1 if orient=='y'  else  0

    sobel = cv2.Sobel(gray, cv2.CV_64F,dx ,dy)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
        
    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_sobel

```
Following are the some of results after applying above filter.

![](resources/sobel-abs-test-3.png)
![](resources/sobel-abs-test-2.png)

#### Sobel direction of the gradient
The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients. For a gradient in both the **x** and **y** directions, the magnitude is the square root of the sum of the squares. Following function was used to calculate direction of the gradient.

```python
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    This is used to generate sobel direction of the warped gray images
    """
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    binary_output =  np.ones_like(absgraddir)
    
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 0

    # Return the binary image
    return binary_output
```
And following were a few results afater applying sobel gradient direction filter.
![](resources/sg-test1.png)
![](resources/sg-test2.png)

After applying above sobel gradient and sobel absolute filter, we can notice how both filters results are varing for two colors such that yellow and white. Now we can combine both filters and can see how combine filter is working for both yellow and white lane lines. Following function was used to combine filters.

```python
def combine_thresholds(unwarp_img, gray, mag_kernel, mag_thresh, dir_thresh, dir_kernel ):
    """
    This is used to combine sobel magnitude and sobel direction gradients.
    """
    gradx = abs_sobel_thresh(gray, orient='x', thresh_min=mag_thresh[0], thresh_max=mag_thresh[1])
    grady = abs_sobel_thresh(gray, orient='y', thresh_min=mag_thresh[0], thresh_max=mag_thresh[1])
    
    mag_binary = mag_threshold(gray, sobel_kernel=mag_kernel, mag_thresh=mag_thresh)
    
    dir_binary = dir_threshold(gray, sobel_kernel=dir_kernel, thresh=dir_thresh)
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # A combine binary image 
    return combined
```
Following are the few results for combine filter.

![](resources/combine-1.png)
![](resources/combine-2.png)

#### Normalize color channels
HLS L channel and LAB B was perfomed well to detect yellow and white lane lines, now we can nomalize both color channels to smooth edges. Following functions were used to nomalize color channels.

######  Nomalize HLS L color channel

```python
def hls_l_nomalize(img, thresh=(220, 255)):
    """
    This is used to nomalize HLS L color channel
    """
    # 1) Convert to HLS color space
    _,hls_l,_ = extract_hls_color_spaces(img)
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

```

######  Nomalize LAB B color channel

```python
def lab_b_nomalize(unwarped_img, thresh=(190,255)):
    """
    This is used to LAB B color channel
    """
    _,_,lab_b = extract_lab_color_spaces(unwarped_img)
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output
```

Following are the results of above filters respectively.

![](resources/thersh-1.png)
![](resources/thersh-2.png )

#### Image pipeline
```python
def pipeline(p_img):
    
    # Undistort images
    undistort_img = undistort(p_img)
    
    # Persfective transform
    img_unwarp,M, Minv = unwarp(undistort_img, src, dst)
    
    # HLS L-channel Threshold (using default parameters)
    img_hls_L = hls_l_nomalize(img_unwarp)

    # Lab B-channel Threshold (using default parameters)
    img_lab_B = lab_b_nomalize(img_unwarp)
    
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_lab_B)
    combined[(img_hls_L == 1) | (img_lab_B == 1)] = 1
    
    return combined, Minv
    
```
Following are the a few samples of pipeline output.

![](resources/pipeline-1.png)
![](resources/pipeline-2.png)

Above pipeline can be used to contine next steps image processing.


Detect lane boundaries
---
To detect lane boundaries we can peaks in a histogram. As we notice around the lane boundaries average white pixel density is higer than comparativelty other regions. Using following function we can generate a histogram data points from a binary image.

```python
def hist(img):
    """
    This is used to extract data points for a histogram
    """
    # Grab only the bottom half of the image
    bottom_half = img[img.shape[0]//2:,:]
    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram
```
Following are the image and it's relevant lane line boundaries histogram  

![](resources/his-img.png) ![](resources/histogram.png)


Pipeline's extracted binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. We can use that as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

