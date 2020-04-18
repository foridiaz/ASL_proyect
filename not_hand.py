
"""
Identificación nothing
"""

import numpy as np
import skimage.io as sk


# Function to transform from RGB to YCbCr
def ycbcr2rgb(im):
    """
    Transforms an RGB image to YCbCr color space
    :param im: Image
    :return: Image in new color space
    """
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def nothing_detection(image) -> bool:
    """
    The function analyzes an image and returns True if a hand is detected and False if it does not
    :param image: numpy array containing the image
    :return:
    Bool:
    True if a hand is detected, False otherwise
    """
    # Image is read and transformed to YCrCb first channel
    image = np.array(ycbcr2rgb(sk.imread(image)))[:, :, 0]
    # We define the area and intensity threshholds
    ArT = 6300
    InT = 40
    # We binarize the image using the intensity threshhold
    image[image <= InT] = 1
    image[image > InT] = 0
    # We get the area by performing the sum of the binarized image
    area = np.sum(image)
    # We determine if the image has a hand or not based on an arbitrary area value
    if area > ArT:
        return True
    else:
        return False


