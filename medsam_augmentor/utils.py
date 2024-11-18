import cv2
import numpy as np

def expand_bbox(bbox, expansion=50, image_shape=None):
    """
    Expand bounding box by specified pixels.
    
    Args:
        bbox (tuple): (x, y, w, h) bounding box
        expansion (int): Pixels to expand in each direction
        image_shape (tuple): Optional (height, width) of image for bounds checking
        
    Returns:
        tuple: Expanded (x, y, w, h) bounding box
    """
    x, y, w, h = bbox
    new_bbox = (
        max(0, x - expansion),
        max(0, y - expansion),
        w + 2 * expansion,
        h + 2 * expansion
    )
    
    if image_shape is not None:
        height, width = image_shape
        x, y, w, h = new_bbox
        x = min(x, width - w)
        y = min(y, height - h)
        new_bbox = (x, y, w, h)
        
    return new_bbox

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE enhancement to image.
    
    Args:
        image (numpy.ndarray): Input image
        clip_limit (float): CLAHE clip limit
        grid_size (tuple): Grid size for CLAHE
        
    Returns:
        numpy.ndarray: CLAHE enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image.astype(np.uint8))

def get_contour_mask(image, threshold=0):
    """
    Get binary mask of breast contour.
    
    Args:
        image (numpy.ndarray): Input mammogram image
        threshold (int): Threshold value for binarization
        
    Returns:
        numpy.ndarray: Binary mask
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def create_alpha_map(mask, max_alpha=0.9, blend_radius=30):
    """
    Create alpha map for gradient blending.
    
    Args:
        mask (numpy.ndarray): Binary mask
        max_alpha (float): Maximum alpha value
        blend_radius (int): Radius for blending
        
    Returns:
        numpy.ndarray: Alpha map for blending
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    alpha = np.zeros_like(dist)
    valid_dist = (dist > 0)
    alpha[valid_dist] = max_alpha * (1 - np.exp(-dist[valid_dist] / blend_radius))
    
    return alpha