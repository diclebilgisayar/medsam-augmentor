<div align="center">

# MEDSAMAugmentor 🔬

<h3>Intelligent Mammographic Data Augmentation using Segment Anything Model</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![SAM](https://img.shields.io/badge/SAM-Compatible-green)](https://segment-anything.com/)

[Key Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Pipeline](#pipeline) • [Citation](#citation) • [License](#license)

<img src="assets/pipeline.png" alt="MEDSAMAugmentor Pipeline" width="800"/>

</div>

## 🌟 Key Features

- **SAM-based Lesion Extraction**: Utilizes Meta's Segment Anything Model for precise lesion segmentation
- **Anatomically-aware Placement**: Intelligently places lesions considering mammographic tissue patterns
- **Gradient Blending**: Seamless integration with progressive alpha blending
- **BBOX Generation**: Automatic bounding box generation for object detection models
- **CLAHE Enhancement**: Improved contrast for better segmentation
- **Flexible Pipeline**: Support for DICOM and standard image formats

## 🚀 Installation

```bash
pip install medsam-augmentor
```

## 📝 Dependencies

- Python 3.8+
- PyTorch 1.9+
- segment-anything
- OpenCV
- NumPy

## 💻 Quick Start

```python
import medsam_augmentor as msa

# Initialize augmentor
augmentor = msa.MEDSAMAugmentor(config_path='config.yaml')

# Load and augment mammogram
original_image = msa.load_mammogram('mammogram.dcm')
augmented_images = augmentor.augment(
    image=original_image,
    num_augmentations=5,
    preserve_labels=True
)
```

## 🔄 Pipeline

### 1. Lesion Extraction
- Utilizes expert masks for initial guidance
- Expands bounding box by 50 pixels
- Applies CLAHE enhancement
- SAM-based segmentation using rectangle ROI

### 2. Anatomical Placement
- Blob detection for mammographic contour
- Intelligent placement considering tissue patterns
- Random but anatomically valid positioning

### 3. Gradient Blending
- Progressive alpha blending from mask boundary
- Distance-based transparency
- Seamless integration with original image

## 📁 Input Format

```
data/
├── images/
│   └── Mass-Training_P_00001_LEFT_CC_0.jpg    # Original mammogram
└── masks/
    └── Mass-Training_P_00001_LEFT_CC_1_0.jpg  # Expert mask
```

## 📊 Output

- Augmented mammographic images
- Corresponding bounding box coordinates
- Preserved clinical validity
- Enhanced dataset diversity

## 🔬 Use Cases

- Medical image analysis research
- Object detection model training
- Clinical validation studies
- Data scarcity solutions

## 📚 Citation

If you use MEDSAMAugmentor in your research, please cite:

```bibtex
@article{medsam2024,
  title={MEDSAMAugmentor: A Python Framework for Intelligent Mammographic Data Augmentation using Segment Anything Model},
  author={[Author Names]},
  journal={Software Impacts},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and feedback:
- 📧 Email: [your-email]
- 🌐 Website: [your-website]

<div align="center">
Made with ❤️ for the Medical Imaging Community
</div>