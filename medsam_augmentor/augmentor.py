import yaml
import numpy as np
import torch
import cv2
import pydicom
import os
from .lesion_extractor import LesionExtractor
from .anatomical_placer import AnatomicalPlacer
from .gradient_blender import GradientBlender

class MEDSAMAugmentor:
    """
    A class for mammographic data augmentation using SAM model.
    
    This class provides functionality to:
    1. Load mammographic images
    2. Extract lesions using SAM
    3. Place lesions in anatomically valid positions
    4. Blend lesions seamlessly with gradient blending
    """
    
    def _get_model_path(self):
        """Get SAM model path dynamically."""
        home_dir = os.path.expanduser('~')
        models_dir = os.path.join(home_dir, '.medsam_augmentor', 'models')
        model_path = os.path.join(models_dir, 'sam_vit_h_4b8939.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SAM model not found at {model_path}. "
                "Please run download_sam_model.py first."
            )
        return model_path
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the MEDSAMAugmentor with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
        
        # Model yolunu dinamik olarak ayarla
        if self.config['lesion_extraction'].get('sam_checkpoint') == 'auto':
            self.config['lesion_extraction']['sam_checkpoint'] = self._get_model_path()
        
        # Setup device (auto-select CUDA if available)
        if self.config['lesion_extraction'].get('device', 'auto') == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
            self.config['lesion_extraction']['device'] = self.device
            
        # Initialize pipeline components
        try:
            self.extractor = LesionExtractor(self.config.get('lesion_extraction', {}))
            self.placer = AnatomicalPlacer(self.config.get('anatomical_placement', {}))
            self.blender = GradientBlender(self.config.get('gradient_blending', {}))
        except Exception as e:
            raise RuntimeError(f"Error initializing pipeline components: {e}")

    def load_mammogram(self, path, convert_rgb=True):
        """
        Load mammogram from various formats (DICOM, JPG, PNG).
        
        Args:
            path (str): Path to mammogram image
            convert_rgb (bool): Whether to convert grayscale to RGB
            
        Returns:
            numpy.ndarray: Loaded image
        """
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found at {path}")
            
        try:
            # Load DICOM files
            if path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(path)
                image = ds.pixel_array
            # Load standard image formats
            else:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                
            if image is None:
                raise ValueError(f"Could not load image from {path}")
            
            # Convert to RGB if requested
            if convert_rgb:
                image = cv2.cvtColor(cv2.equalizeHist(image), cv2.COLOR_GRAY2RGB)
                
            return image
            
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def _get_bbox_from_mask(self, mask):
        """
        Get bounding box from binary mask.
        
        Args:
            mask (numpy.ndarray): Binary mask
            
        Returns:
            tuple: (x, y, w, h) bounding box coordinates
        """
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

    def augment(self, image, mask=None, num_augmentations=5, preserve_labels=True):
        """
        Augment the mammogram with multiple lesions.
        
        Args:
            image (numpy.ndarray): Input mammogram image
            mask (numpy.ndarray, optional): Binary mask for lesion
            num_augmentations (int): Number of augmentations to generate
            preserve_labels (bool): Whether to preserve original annotations
            
        Returns:
            tuple: (augmented_images, bboxes, original_bbox)
        """
        try:
            # Get original lesion bbox if mask is provided
            original_bbox = None
            if mask is not None:
                original_bbox = self._get_bbox_from_mask(mask)
            
            # Extract lesion
            lesion_data = self.extractor.extract_lesion(image, mask)
            
            # Start with original image
            current_image = image.copy()
            bboxes = []
            existing_boxes = [original_bbox] if original_bbox else []
            
            # Add multiple lesions
            for _ in range(num_augmentations):
                try:
                    placement_coords = self.placer.get_placement_coordinates(
                        current_image, 
                        existing_boxes=existing_boxes
                    )
                    
                    augmented_image, bbox = self.blender.blend(
                        original_image=current_image,
                        lesion_data=lesion_data,
                        placement_coords=placement_coords
                    )
                    
                    current_image = augmented_image
                    bboxes.append(bbox)
                    existing_boxes.append(bbox)
                    
                except ValueError as e:
                    print(f"Warning: Skipping one augmentation due to: {str(e)}")
                    continue
            
            return [current_image], bboxes, original_bbox
            
        except Exception as e:
            raise ValueError(f"Augmentation failed: {e}")

    def save_results(self, augmented_images, bboxes, output_dir, original_image_path, original_image, original_bbox=None):
        """
        Save results in multiple formats.
        
        Args:
            augmented_images (list): List of augmented images
            bboxes (list): List of augmented bounding boxes
            output_dir (str): Directory to save results
            original_image_path (str): Path to original image for naming
            original_image: Original mammogram image
            original_bbox (tuple): Original lesion bbox (x,y,w,h)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Get original image name without extension
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            
            # Save bboxes information
            bbox_path = os.path.join(output_dir, f"{base_name}_bboxes.txt")
            with open(bbox_path, 'w') as f:
                if original_bbox:
                    f.write(f"original_lesion: {original_bbox[0]},{original_bbox[1]},{original_bbox[2]},{original_bbox[3]}\n")
                for i, bbox in enumerate(bboxes):
                    f.write(f"augmented_lesion_{i}: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
            
            # 1. Save original image
            original_path = os.path.join(output_dir, f"{base_name}.jpg")
            cv2.imwrite(original_path, original_image)
            
            # 2. Save augmented image (with all lesions, no bbox)
            augmented_path = os.path.join(output_dir, f"augmented_{base_name}.jpg")
            cv2.imwrite(augmented_path, augmented_images[-1])
            
            # 3. Save bbox visualization
            bbox_viz_image = augmented_images[-1].copy()
            
            # Draw original bbox in orange (if exists)
            if original_bbox:
                x, y, w, h = original_bbox
                cv2.rectangle(bbox_viz_image, (x, y), (x+w, y+h), (0, 165, 255), 3)  # Orange, thicker line
            
            # Draw augmented bboxes in green
            for bbox in bboxes:
                x, y, w, h = bbox
                cv2.rectangle(bbox_viz_image, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Green, thicker line
            
            bbox_viz_path = os.path.join(output_dir, f"bbox_{base_name}.jpg")
            cv2.imwrite(bbox_viz_path, bbox_viz_image)
            
            print(f"Saved results to {output_dir}:")
            print(f"- Original image: {original_path}")
            print(f"- Augmented image: {augmented_path}")
            print(f"- Bbox visualization: {bbox_viz_path}")
            print(f"- Bounding boxes: {bbox_path}")
                
        except Exception as e:
            raise ValueError(f"Error saving results: {e}")

    def __repr__(self):
        """String representation of the augmentor."""
        return f"MEDSAMAugmentor(device={self.device}, config={self.config})"

    def __str__(self):
        """User-friendly string representation."""
        return f"MEDSAMAugmentor using {self.device} device"