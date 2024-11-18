import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import medsam_augmentor as msa

# Initialize augmentor
augmentor = msa.MEDSAMAugmentor(config_path='../config/config.yaml')

# Test için print
print("Device:", augmentor.device)
print("Initialization successful!")