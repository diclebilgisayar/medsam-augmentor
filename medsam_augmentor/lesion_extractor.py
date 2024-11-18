import numpy as np
import cv2
import os
from .utils import expand_bbox, apply_clahe
from segment_anything import sam_model_registry, SamPredictor

class LesionExtractor:
    def __init__(self, config):
        self.config = config
        self.sam = self._initialize_sam()
        self.predictor = SamPredictor(self.sam)

    def _initialize_sam(self):
        """Initialize and load SAM model."""
        models_dir = os.path.join(os.path.expanduser('~'), '.medsam_augmentor', 'models')
        sam_checkpoint = os.path.join(models_dir, "sam_vit_h_4b8939.pth")
        
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(
                f"SAM model not found at {sam_checkpoint}. "
                "Please run download_sam_model.py first."
            )
        
        model_type = self.config.get('model_type', "vit_h")
        device = self.config.get('device', 'cuda')
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        return sam
    
    def extract_lesion(self, image, mask=None):
        """
        Extract lesion from mammogram using SAM.
        
        Args:
            image (numpy.ndarray): Input mammogram (RGB)
            mask (numpy.ndarray, optional): Expert mask
            
        Returns:
            dict: Extracted lesion data including mask and enhanced region
        """
        # Ensure image is RGB
        if len(image.shape) != 3:
            raise ValueError("Input image must be RGB (3 channels)")

        # Convert mask to binary grayscale if provided
        if mask is not None:
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = (mask > 0).astype(np.uint8) * 255

        if mask is None:
            raise ValueError("Mask is required for lesion extraction")
            
        # Get bbox from mask
        bbox = self._get_bbox_from_mask(mask)
            
        # Expand bbox
        expanded_bbox = expand_bbox(bbox, 
                                  expansion=self.config.get('bbox_expansion', 50),
                                  image_shape=image.shape[:2])  # Only height and width
        
        # Extract region
        x, y, w, h = expanded_bbox
        region = image[y:y+h, x:x+w]
        
        # Convert to grayscale for enhancement
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        enhanced_region = apply_clahe(gray_region)
        
        # Convert back to RGB for consistency
        enhanced_region = cv2.cvtColor(enhanced_region, cv2.COLOR_GRAY2RGB)
        
        return {
            'region': enhanced_region,
            'mask': mask[y:y+h, x:x+w],  # Crop mask to match region
            'bbox': expanded_bbox
        }

    def _get_bbox_from_mask(self, mask):
        """Get bounding box from binary mask."""
        # Ensure mask is grayscale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Convert to binary
        mask_binary = (mask > 0).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            raise ValueError("No contours found in mask")
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)