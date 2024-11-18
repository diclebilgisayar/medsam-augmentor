import os
import urllib.request
import sys

def download_sam_model():
    """Download SAM model file."""
    # Model URLs and paths
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    models_dir = os.path.join(os.path.expanduser('~'), '.medsam_augmentor', 'models')
    model_path = os.path.join(models_dir, 'sam_vit_h_4b8939.pth')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Download if not exists
    if not os.path.exists(model_path):
        print(f"Downloading SAM model (2.4 GB)...")
        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = int(downloaded * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
                
            urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
            print("\nDownload completed!")
        except Exception as e:
            print(f"\nError downloading model: {str(e)}")
            return None
    else:
        print(f"Model already exists at {model_path}")
        
    return model_path

if __name__ == "__main__":
    download_sam_model()