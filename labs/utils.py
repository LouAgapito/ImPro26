import numpy as np
import copy
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import random
from PIL import Image
from typing import Union
from sklearn.feature_extraction.image import check_array, _extract_patches, \
_compute_n_patches, check_random_state


def extract_patches_2d(image, patch_size, *, max_patches=None,
                       random_state=None):
    """Reshape a 2D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = _extract_patches(image,
                                         patch_shape=(p_h, p_w, n_colors),
                                         extraction_step=patch_size[0])

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.squeeze()
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


def resize(img, scale: Union[float, int]) -> np.ndarray:
    """ Resize an image maintaining its proportions
    Args:
        fp (str): Path argument to image file
        scale (Union[float, int]): Percent as whole number of original image. eg. 53
    Returns:
        image (np.ndarray): Scaled image
    """
    _scale = lambda dim, s: int(dim * s / 100)
    height, width, channels = img.shape
    new_width: int = _scale(width, scale)
    new_height: int = _scale(height, scale)
    new_dim: tuple = (new_width, new_height)
    return cv2.resize(src=img, dsize=new_dim, interpolation=cv2.INTER_LINEAR)

def colourize(img):
    height, width = img.shape

    colors = []
    colors.append([])
    colors.append([])
    color = 1
    # Displaying distinct components with distinct colors
    coloured_img = Image.new("RGB", (width, height))
    coloured_data = coloured_img.load()

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 0:
                if img[i][j] not in colors[0]:
                    colors[0].append(img[i][j])
                    colors[1].append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

                ind = colors[0].index(img[i][j])
                coloured_data[j, i] = colors[1][ind]

    return coloured_img


def binarize(img_array, threshold=130):
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            if img_array[i][j] > threshold:
                img_array[i][j] = 0
            else:
                img_array[i][j] = 1
    return img_array


def order_points(pts):
    '''
    
    code adapted from
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    '''
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts, dst=None):
    '''
    
    code adapted from
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    '''
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    if dst is None:
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def draw_corners(image, corners_map):
    """Draw a point for each possible corner."""
    
    color_img = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)
    for each_corner in corners_map:
        cv2.circle(color_img, (each_corner[1], each_corner[0]), 1, (255,0,0), -1)
    return color_img

def load_image(path_name):
    image = cv2.imread(path_name)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_kernel(image, kernel):
    kernel_size = kernel.shape[0]

    padding_amount = int((kernel_size - 1) / 2)
    rows = image.shape[0] + 2 * padding_amount
    cols = image.shape[1] + 2 * padding_amount
    channels = image.shape[2]
    padded_image_placeholder = np.zeros((rows, cols, channels))
    padded_image_placeholder[padding_amount:rows-padding_amount, padding_amount:cols-padding_amount, :] = image

    filtered_image = np.zeros(image.shape)

    for each_channel in range(channels):
        padded_2d_image = padded_image_placeholder[:,:,each_channel]
        filtered_2d_image = filtered_image[:,:,each_channel]
        width = padded_2d_image.shape[0]
        height = padded_2d_image.shape[1]
        for i in range(width-kernel_size+1):
            for j in range(height-kernel_size+1):
                current_block = padded_2d_image[i:i+kernel_size, j:j+kernel_size]
                convoluted_value = np.sum(current_block * kernel)
                filtered_2d_image[i][j] = convoluted_value
        filtered_image[:,:,each_channel] = filtered_2d_image

    return filtered_image

def get_gaussian_filter(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    denom = 2 * np.pi * sigma * sigma
    samples = np.arange(-int(kernel_size/2), int(kernel_size/2) + 1)

    for i in range(len(samples)):
        for j in range(len(samples)):
            x = samples[i]
            y = samples[j]
            num = np.exp(-1*(((x*x) + (y*y)) / (2*sigma*sigma)))
            val = num / np.sqrt(denom)
            kernel[i][kernel_size - j - 1] = val
    kernel = kernel / kernel.sum()
    
    return kernel