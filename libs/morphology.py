import numpy as np
from cv2 import imread,imwrite, dilate, erode
from cv2 import cvtColor, COLOR_BGR2HLS, calcHist
import cv2 as cv2
import random
from matplotlib import pyplot as plt
from skimage.measure import label



# --------------------------------- Zusatzaufgabe ---------------------------------------
def segment_util(img):
    """
    Given an input image, output the segmentation result
    Input:  
        img:        n x m x 3, values are within [0,255]
    Output:
        img_seg:    n x m
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Apply Canny edge detection
    edged = cv2.Canny(gray_blurred, 200, 300)

    # Perform a dilation and erosion to close gaps in between object edges
    dilated_edged = cv2.dilate(edged.copy(), None, iterations=5)
    eroded_edged = cv2.erode(dilated_edged.copy(), None, iterations=5)

    # Perform a circle Hough Transform to detect the coins
    circles = cv2.HoughCircles(eroded_edged, cv2.HOUGH_GRADIENT, 1, 110, param1=400, param2=20, minRadius=110, maxRadius=195)

    # Create a black image with the same dimensions as the original
    img_seg = np.zeros(img.shape, dtype=np.uint8)

    # Ensure at least some circles were found
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(img_seg, (x, y), r, (True, 255, 255), -1)  # -1 indicates to fill the circle

    return img_seg.astype(bool)

def close_hole_util(img):
    """
    Given the segmented image, use morphology techniques to close the holes
    Input:
        img:        n x m, values are within [0,1]
    Output:
        closed_img: n x m
    """
    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    # Apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed_img = cv2.erode(dilated_img, kernel, iterations=5)

    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed_img = cv2.dilate(closed_img, kernel, iterations=5)

    return closed_img

def instance_segmentation_util(img):
    """
    Given the closed segmentation image, output the instance segmentation result
    Input:  
        img:        n x m, values are within [0,255]
    Output:
        instance_seg_img:    n x m x 3, different coin instances have different colors
    """
    
    """# Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilated_img = cv2.dilate(img, kernel, iterations=2)

    # Apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100, 100))
    closed_img = cv2.erode(dilated_img, kernel, iterations=2)

    for _ in range(10):
        closed_img = cv2.GaussianBlur(closed_img, (51, 51), 10)
        # Re-threshold the image to make it binary again
        _, closed_img = cv2.threshold(closed_img, 127, 255, cv2.THRESH_BINARY)

    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed_img = cv2.dilate(closed_img, kernel, iterations=40)

    # Find connected components
    _, labeled_img = cv2.connectedComponents(closed_img)
    labeled_img += 1

    closed_img = cv2.merge((closed_img, closed_img, closed_img))
    watershed = cv2.watershed(closed_img, labeled_img)

    for segment in np.unique(watershed):
        closed_img[watershed == segment] = [255 * random.random(),
                                            255 * random.random(),
                                            255 * random.random()]

    instance_seg_img = closed_img * 255

    ### Better to use https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    ### to do the instance segmentation, but I thought I might try something different"""

    _, labeled_img = cv2.connectedComponents(img)
    labeled_img = labeled_img.astype(np.int32)
    labeled_img += 1

    img = cv2.merge((img, img, img))
    closed_img = cv2.convertScaleAbs(img)

    watershed = cv2.watershed(img, labeled_img)

    for segment in np.unique(watershed):
        closed_img[watershed == segment] = [255 * random.random(),
                                            255 * random.random(),
                                            255 * random.random()]

    instance_seg_img = closed_img * 255

    return instance_seg_img

def text_recog_util(text, letter_not):
    """
    Given the text and the character, recognise the character in the text
    Input:
        text:           n x m
        letter_not:     a x b
    Output:
        text_er_dil:    n x m
    """
    from scipy.ndimage import binary_erosion as erode
    from scipy.ndimage import binary_dilation as dilate
    ## TODO
    # Initial image thresholding
    # We apply binary inverse thresholding to get the binary representations of the images
    _, text_binary = cv2.threshold(text, 0.5, 1, cv2.THRESH_BINARY_INV)
    _, letter_binary = cv2.threshold(letter_not, 0.5, 1, cv2.THRESH_BINARY)

    # Perform binary erosion on the text_binary with the structuring element being letter_binary
    # Erosion will shrink the white regions (foreground) in the binary image
    eroded_image = erode(text_binary, letter_binary)

    # Perform binary dilation on the eroded image with the structuring element being letter_binary
    # Dilation will expand the white regions in the binary image
    dilated_image = dilate(eroded_image, letter_binary)

    # Our final image is the dilated image
    text_er_dil = dilated_image

    return text_er_dil