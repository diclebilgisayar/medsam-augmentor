import os
import urllib.request
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

class PostInstallCommand:
    """Post-installation tasks base class"""
    def download_sam_model(self):
        # Model URLs and paths
        models = {
            'sam_vit_h_4b8939.pth': {
                'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                'size': '2.38 GB'
            }
        }
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.expanduser('~'), '.medsam_augmentor', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model_info in models.items():
            model_path = os.path.join(models_dir, model_name)
            
            # Download only if model doesn't exist
            if not os.path.exists(model_path):
                print(f"\nDownloading {model_name} ({model_info['size']})...")
                try:
                    urllib.request.urlretrieve(
                        model_info['url'],
                        model_path,
                        reporthook=self._download_progress
                    )
                    print(f"\n{model_name} downloaded successfully!")
                except Exception as e:
                    print(f"\nError downloading {model_name}: {str(e)}")
                    print("Please download manually from:", model_info['url'])
            else:
                print(f"\n{model_name} already exists at {model_path}")

    def _download_progress(self, count, block_size, total_size):
        """Show download progress"""
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end='')

class PostInstall(install, PostInstallCommand):
    """Post-installation tasks for pip install"""
    def run(self):
        install.run(self)
        self.download_sam_model()

class PostDevelop(develop, PostInstallCommand):
    """Post-installation tasks for pip install -e"""
    def run(self):
        develop.run(self)
        self.download_sam_model()

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medsam-augmentor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent Mammographic Data Augmentation using Segment Anything Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medsam-augmentor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.2",
        "opencv-python>=4.5.0",
        "pydicom>=2.1.2",
        "torch>=1.9.0",
        "segment-anything>=1.0",
        "pyyaml>=5.4.1",
    ],
    cmdclass={
        'install': PostInstall,
        'develop': PostDevelop,
    },
)
