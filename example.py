import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
from medsam_augmentor import MEDSAMAugmentor

def main():
    # Initialize augmentor
    augmentor = MEDSAMAugmentor(config_path='config/config.yaml')
    
    # Define paths
    input_dir = "data/images"
    mask_dir = "data/masks"
    output_dir = "data/augmented"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process example image
    image_path = os.path.join(input_dir, "Mass-Training_P_00001_LEFT_CC_0.jpg")
    mask_path = os.path.join(mask_dir, "Mass-Training_P_00001_LEFT_CC_1_0.jpg")
    
    # Load images
    original_image = augmentor.load_mammogram(image_path)
    mask_image = augmentor.load_mammogram(mask_path)
    
    # Augment
    augmented_images, bboxes, original_bbox = augmentor.augment(
        image=original_image,
        mask=mask_image,
        num_augmentations=3,
        preserve_labels=True
    )
    
    # Save all results
    augmentor.save_results(
        augmented_images=augmented_images,
        bboxes=bboxes,
        output_dir=output_dir,
        original_image_path=image_path,
        original_image=original_image,
        original_bbox=original_bbox
    )

if __name__ == "__main__":
    main()