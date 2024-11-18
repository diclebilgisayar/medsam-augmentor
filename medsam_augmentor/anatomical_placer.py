import cv2
import numpy as np

class AnatomicalPlacer:
    def __init__(self, config):
        """
        Initialize anatomical placement module.
        
        Args:
            config (dict): Configuration for anatomical placement
            - min_distance: Minimum distance between lesions
            - border_margin: Margin from breast border
        """
        self.config = config
        self.min_distance = config.get('min_distance', 50)
        self.border_margin = config.get('border_margin', 30)
        
    def _get_breast_mask(self, image):
        """
        Extract breast region mask using thresholding and morphology.
        
        Args:
            image (numpy.ndarray): Grayscale mammogram image
            
        Returns:
            numpy.ndarray: Binary mask of breast region
        """
        # Otsu thresholding to separate breast from background
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _get_valid_placement_area(self, breast_mask):
        """
        Get valid area for lesion placement.
        
        Args:
            breast_mask (numpy.ndarray): Binary mask of breast region
            
        Returns:
            numpy.ndarray: Binary mask of valid placement area
        """
        # Daha güvenli bir sınır marjı
        border_margin = self.border_margin * 3  # Kenardan daha fazla uzaklaş
        
        # Meme kontürünü elde et
        kernel = np.ones((border_margin, border_margin), np.uint8)
        
        # Kenarlardan uzaklaş
        eroded_mask = cv2.erode(breast_mask, kernel)
        
        # Memenin merkez bölgesini bul (dış kenardan daha fazla uzaklaş)
        center_kernel = np.ones((border_margin * 2, border_margin * 2), np.uint8)
        center_area = cv2.erode(eroded_mask, center_kernel)
        
        return center_area
    
    def _get_random_valid_position(self, valid_area):
        """
        Get random position within valid area.
        
        Args:
            valid_area (numpy.ndarray): Binary mask of valid placement area
            
        Returns:
            tuple: (y, x) coordinates
            
        Raises:
            ValueError: If no valid positions found
        """
        # Get all valid coordinates
        y_coords, x_coords = np.where(valid_area > 0)
        
        if len(y_coords) == 0:
            raise ValueError("No valid placement positions found in the valid area")
        
        # Random selection
        idx = np.random.randint(0, len(y_coords))
        return y_coords[idx], x_coords[idx]
    
    def _check_overlap(self, new_pos, existing_bbox, margin=20):
        """
        Check if new position would overlap with existing bbox.
        
        Args:
            new_pos (tuple): (x, y) coordinates of new position
            existing_bbox (tuple): (x, y, w, h) of existing bbox
            margin (int): Additional margin to prevent close placement
            
        Returns:
            bool: True if overlap exists
        """
        x, y = new_pos
        ex, ey, ew, eh = existing_bbox
        
        # Add margin to existing bbox
        ex -= margin
        ey -= margin
        ew += 2 * margin
        eh += 2 * margin
        
        # Check overlap
        return not (x > ex + ew or x + ew < ex or y > ey + eh or y + eh < ey)
    
    def get_placement_coordinates(self, image, existing_boxes=None):
        """
        Get valid coordinates for lesion placement.
        
        Args:
            image (numpy.ndarray): Input mammogram image
            existing_boxes (list): List of existing bbox tuples (x,y,w,h)
            
        Returns:
            tuple: (x, y) coordinates for placement
            
        Raises:
            ValueError: If no valid position found after maximum attempts
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            grayscale_image = image.copy()

        # Get breast mask and valid area
        breast_mask = self._get_breast_mask(grayscale_image)
        valid_area = self._get_valid_placement_area(breast_mask)
        
        # Maximum deneme sayısı
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            y, x = self._get_random_valid_position(valid_area)
            
            # Overlap kontrolü
            if existing_boxes:
                overlap = False
                for bbox in existing_boxes:
                    if self._check_overlap((x, y), bbox):
                        overlap = True
                        break
                if not overlap:
                    return (x, y)
            else:
                return (x, y)
                
            attempt += 1
            
        raise ValueError("Could not find valid non-overlapping position after maximum attempts")
    
    def validate_placement(self, coords, image_shape, existing_lesions=None):
        """
        Validate if placement coordinates are valid.
        
        Args:
            coords (tuple): (x, y) coordinates
            image_shape (tuple): Shape of the image
            existing_lesions (list): List of existing lesion coordinates
            
        Returns:
            bool: Whether placement is valid
        """
        x, y = coords
        height, width = image_shape[:2] if len(image_shape) > 2 else image_shape
        
        # Check image bounds
        if not (0 <= x < width and 0 <= y < height):
            return False
            
        # Check minimum distance from existing lesions
        if existing_lesions:
            for ex_x, ex_y in existing_lesions:
                distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
                if distance < self.min_distance:
                    return False
                    
        return True

    def __repr__(self):
        """String representation of the placer."""
        return f"AnatomicalPlacer(min_distance={self.min_distance}, border_margin={self.border_margin})"