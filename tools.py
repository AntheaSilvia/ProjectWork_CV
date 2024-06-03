import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIRST TASK

def array_of_images(path):
    """
    Load grayscale and color images from files in the specified directory based on filename suffixes.

    Parameters:
        path (str): The path to the directory containing the images.

    Returns:
        tuple: A tuple containing lists of grayscale images and color images.
    """
    bw_file_names = glob.glob(path + "/*C0*")
    bw_file_names.sort()
    bw_images = [cv2.imread(img) for img in bw_file_names]

    color_file_names = glob.glob(path + "/*C1*")
    color_file_names.sort()
    color_images = [cv2.imread(img) for img in color_file_names]

    return bw_images, color_images

def get_threshold(image):
    """
    Calculate the threshold for binarizing the given image.

    This function calculates the threshold as half of the median intensity value
    of the flattened image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        int: The threshold value.
    """
    flattened = image.flatten()
    median = int(np.median(flattened))
    threshold = int(median / 2)

    return threshold

def show_images(name1, name2, image1, image2):
    """
    Display two images side by side with their respective titles.

    Parameters:
        name1 (str): Title for the first image.
        name2 (str): Title for the second image.
        image1 (numpy.ndarray): First image to display.
        image2 (numpy.ndarray): Second image to display.

    Returns:
        None
    """
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(image1_rgb)
    ax1.axis('off')

    ax2.imshow(image2_rgb)
    ax2.axis('off')

    ax1.text(0, -20, name1, color='red', fontsize=12, fontweight='bold')
    ax2.text(0, -20, name2, color='red', fontsize=12, fontweight='bold')

def get_component(labels, label):
    """
    Extracts a specific component from a labeled image.

    Parameters:
        labels (numpy.ndarray): The labeled image where each connected component has a unique label.
        label (int): The label of the component to extract.

    Returns:
        numpy.ndarray: A binary image representing the specified component, where pixels belonging
                       to the component are set to 255 (white) and others to 0 (black).
    """
    component = np.zeros_like(labels, dtype=np.uint8)
    component[labels == label] = 255

    return component

def get_connected_component(image):
    """
    Find and return the biggest connected component in the given binary image.

    This function uses the connectedComponentsWithStats function from OpenCV to find
    connected components in the binary image. It then iterates through the components
    to find the largest one based on its area.

    Parameters:
        image (numpy.ndarray): The input binary image.

    Returns:
        numpy.ndarray: The binary image of the biggest connected component.
    """
    elements, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)

    max_area = float("-inf")
    biggest_component = None

    for i in range(1, elements):
        component = get_component(labels, i)
        component_area = cv2.countNonZero(component)

        if component_area > max_area:
            biggest_component = component
            max_area = component_area

    return biggest_component

def fill_element(image):
    """
    Fill the holes in the given binary image.

    This function identifies holes in the input binary image and fills them to obtain a
    connected and filled binary image.

    Parameters:
        image (numpy.ndarray): The input binary image.

    Returns:
        numpy.ndarray: The binary image with filled holes.
    """
    mask = get_connected_component(image)
    holes = np.where(mask == 0)

    if len(holes[0]) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    seed = (holes[0][0], holes[1][0])
    holes_mask_inverted = mask.copy()
    h_, w_ = mask.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
    cv2.floodFill(holes_mask_inverted, mask_, seedPoint=seed, newVal=255)
    holes_mask = cv2.bitwise_not(holes_mask_inverted)
    filled = mask + holes_mask

    return filled

def is_not_duplicate(m, measured, min_parallax):
    """
    Checks if a measurement is not a duplicate compared to the already measured ones.

    Parameters:
        m (tuple): The measurement to check.
        measured (list): List of measurements already taken.
        min_parallax (float): Minimum distance between measurements to not be considered duplicates.

    Returns:
        bool: True if 'm' is not a duplicate, False otherwise.
    """
    for c in measured:
        if np.array_equal(m[0], c[0]):
            continue

        distance = math.sqrt((c[1][0] - m[1][0]) ** 2 + (c[1][1] - m[1][1]) ** 2)
        if distance < min_parallax and m[2] <= c[2]:
            return False

    return True

def filter_duplicated_contours(contours, min_parallax):
    """
    Filters out duplicated contours based on the minimum parallax distance.

    Parameters:
        contours (list): List of contours.
        min_parallax (float): Minimum distance between contours to not be considered duplicates.

    Returns:
        list: Filtered list of contours.
    """
    measured = []

    for c in contours:
        if len(c) >= 5:
            centroid, _, _ = cv2.fitEllipse(c)
            area = cv2.contourArea(c)
            measured.append([c, centroid, area])

    filtered = []

    for m in measured:
        if is_not_duplicate(m, measured, min_parallax):
            filtered.append(m[0])

    return filtered

def draw_defect(image, component, thickness, scale, min_area, max_area, min_parallax):
    """
    Draws defects on the image based on the given component.

    Parameters:
        image (numpy.ndarray): The image on which to draw defects.
        component (numpy.ndarray): The component containing defects.
        thickness (int): Thickness of the drawn ellipse.
        scale (float): Scale factor for the ellipse axes.
        min_area (float): Minimum area of the defect to be drawn.
        max_area (float): Maximum area of the defect to be drawn.
        min_parallax (float): Minimum parallax distance between defects.

    Returns:
        int: Number of defects drawn.
    """
    contours, hierarchy = cv2.findContours(component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return

    filtered_contours = filter_duplicated_contours(contours, min_parallax)

    drawn = 0

    for c in filtered_contours:
        area = cv2.contourArea(c)
        if min_area < area and len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            scaled_axes = (ellipse[1][0] * scale, ellipse[1][1] * scale)
            if scaled_axes[0] * scaled_axes[1] * math.pi < max_area:
                scaled_ellipse = ellipse[0], scaled_axes, ellipse[2]
                cv2.ellipse(image, scaled_ellipse, (0, 0, 255), thickness)
                drawn += 1

    return drawn

def draw_fruit_outline(image, mask, thickness, color=(0, 255, 0)):
    """
    Draws the outline of the fruit on the image based on the given mask.

    Parameters:
        image (numpy.ndarray): The image on which to draw the outline.
        mask (numpy.ndarray): The mask representing the fruit.
        thickness (int): Thickness of the drawn contour.
        color (tuple): Color of the drawn contour.

    Returns:
        None
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, color, thickness)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SECOND TASK

def upload_samples(path):
    """
    Load images from files in the specified directory.

    Parameters:
        path (str): The path to the directory containing the images.

    Returns:
        list: A list containing the images loaded from the files in the directory.
    """

    samples_file_names = glob.glob(path + "/*")
    samples_file_names.sort()
    samples = [cv2.imread(img) for img in samples_file_names]

    return samples