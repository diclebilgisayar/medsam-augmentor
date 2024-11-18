import cv2
import numpy as np

class GradientBlender:
    def __init__(self, config):
        """
        Initialize gradient blending module.
        
        Args:
            config (dict): Configuration for gradient blending
            - blend_radius: Radius for gradient blend effect
            - alpha_max: Maximum alpha value for blending
        """
        self.config = config
        self.blend_radius = config.get('blend_radius', 30)
        self.alpha_max = config.get('alpha_max', 0.9)
    
    def _apply_random_transforms(self, region, mask):
        """
        Apply random rotation and scaling to lesion.
        
        Args:
            region (numpy.ndarray): Lesion region image
            mask (numpy.ndarray): Lesion mask
            
        Returns:
            tuple: (transformed_region, transformed_mask)
        """
        # Random rotation angle
        angle = np.random.uniform(-30, 30)
        
        # Random scale factor (0.6 to 1.1 arası, daha çok küçük ölçek olasılığı)
        scale_choices = np.concatenate([
            np.random.uniform(0.6, 0.8, 3),  # Küçük ölçekler için daha fazla şans
            np.random.uniform(0.8, 1.1, 2)   # Büyük ölçekler için daha az şans
        ])
        scale = np.random.choice(scale_choices)
        
        # Get image center
        h, w = region.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply transformations
        borderMode = cv2.BORDER_REFLECT  # Yansıma ile kenar doldurma
        transformed_region = cv2.warpAffine(region, M, (w, h), borderMode=borderMode)
        transformed_mask = cv2.warpAffine(mask, M, (w, h), borderMode=borderMode)
        
        return transformed_region, transformed_mask
    
    def _create_alpha_map(self, mask):
        """Create alpha map for smooth blending."""
        # Convert mask to binary
        binary_mask = (mask > 0).astype(np.float32)
        
        # Create gradient from mask edges
        kernel_size = 15  # Blend radius
        kernel = np.ones((kernel_size, kernel_size), np.float32)
        dilated = cv2.dilate(binary_mask, kernel)
        
        # Create gradient
        alpha = cv2.GaussianBlur(dilated, (kernel_size, kernel_size), 0)
        
        # Normalize to [0, 1]
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-7)
        
        # Ensure mask area is fully opaque
        alpha[binary_mask > 0] = 1.0
        
        return alpha        
        
    def _apply_noise_and_transforms(self, region, alpha_map):
        """
        Apply noise and texture variations for realism.
        
        Args:
            region (numpy.ndarray): Image region
            alpha_map (numpy.ndarray): Alpha map
            
        Returns:
            numpy.ndarray: Enhanced region
        """
        region = region.astype(np.float32)
        
        # Gaussian noise
        noise_level = np.random.uniform(1.5, 2.5)
        noise = np.random.normal(0, noise_level, region.shape).astype(np.float32)
        
        # Expand alpha map if needed
        if len(alpha_map.shape) == 2 and len(region.shape) == 3:
            alpha_map = np.stack([alpha_map] * region.shape[2], axis=2)
        
        # Apply noise weighted by alpha map
        region_noised = region + noise * alpha_map
        
        # Texture variation
        texture_scale = np.random.uniform(0.95, 1.05, region.shape).astype(np.float32)
        region_noised = region_noised * texture_scale
        
        # Local contrast variation
        contrast = np.random.uniform(0.95, 1.05)
        region_noised = ((region_noised - 128) * contrast + 128)
        
        # Yerel parlaklık varyasyonu
        brightness_var = np.random.uniform(-5, 5)
        region_noised += brightness_var
        
        # Clip to valid range
        region_noised = np.clip(region_noised, 0, 255)
        
        return region_noised.astype(np.uint8)
        
    def _create_transparent_lesion(self, lesion_region, lesion_mask):
            """
            Create transparent lesion image with only the lesion region visible.
            
            Args:
                lesion_region: Cropped image containing lesion
                lesion_mask: Binary mask of lesion region
                
            Returns:
                numpy.ndarray: RGBA image with transparent background
            """
            # Ensure binary mask
            binary_mask = (lesion_mask > 0).astype(np.uint8)
            
            # Create RGBA image (4 channels)
            if len(lesion_region.shape) == 2:
                # Convert grayscale to RGB
                lesion_rgb = cv2.cvtColor(lesion_region, cv2.COLOR_GRAY2RGB)
            else:
                lesion_rgb = lesion_region.copy()
                
            # Create alpha channel from mask
            alpha = binary_mask * 255
            
            # Create RGBA image
            rgba = np.zeros((lesion_region.shape[0], lesion_region.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = lesion_rgb
            rgba[..., 3] = alpha
            
            return rgba  

    def _create_seamless_lesion(self, lesion_region, lesion_mask):
            """
            Create seamless lesion with feathered edges.
            """
            # Ensure binary mask
            binary_mask = (lesion_mask > 0).astype(np.uint8)
            
            # Create feathered edge
            kernel_size = 5
            feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            feathered_mask = cv2.GaussianBlur(binary_mask.astype(float), (kernel_size, kernel_size), 0)
            
            # Normalize mask
            feathered_mask = cv2.normalize(feathered_mask, None, 0, 1, cv2.NORM_MINMAX)
            
            return lesion_region, feathered_mask

    
    def blend(self, original_image, lesion_data, placement_coords):
        try:
            lesion_region = lesion_data['region']
            lesion_mask = lesion_data['mask']
            
            # Apply random transforms
            lesion_region, lesion_mask = self._apply_random_transforms(lesion_region, lesion_mask)
            
            # Create seamless lesion
            processed_lesion, feathered_mask = self._create_seamless_lesion(lesion_region, lesion_mask)
            
            x, y = placement_coords
            h, w = lesion_region.shape[:2]
            
            # Extract ROI
            augmented_image = original_image.copy()
            roi = augmented_image[y:y+h, x:x+w].astype(float)
            
            # Convert to float for calculations
            lesion_float = processed_lesion.astype(float)
            
            # Overlay blend mode
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            if len(lesion_float.shape) == 2:
                lesion_float = cv2.cvtColor(lesion_float, cv2.COLOR_GRAY2RGB)
                
            # Normalize intensity ranges
            roi_norm = roi / 255.0
            lesion_norm = lesion_float / 255.0
            
            # Overlay blend formula
            blended = np.where(roi_norm < 0.5,
                             2 * roi_norm * lesion_norm,
                             1 - 2 * (1 - roi_norm) * (1 - lesion_norm))
            
            # Apply feathered mask
            feathered_mask_3ch = np.stack([feathered_mask] * 3, axis=-1)
            final_blend = roi_norm * (1 - feathered_mask_3ch) + blended * feathered_mask_3ch
            
            # Convert back to uint8
            final_blend = (final_blend * 255).clip(0, 255).astype(np.uint8)
            
            # Place back
            augmented_image[y:y+h, x:x+w] = final_blend
            
            return augmented_image, (x, y, w, h)
            
        except Exception as e:
            raise ValueError(f"Blending failed: {str(e)}")    
                       
    
    def __repr__(self):
        """String representation of the blender."""
        return f"GradientBlender(blend_radius={self.blend_radius}, alpha_max={self.alpha_max})"