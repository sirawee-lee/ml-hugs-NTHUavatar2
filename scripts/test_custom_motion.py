#!/usr/bin/env python3
"""
Generate HUGS animation with custom motion directly.
This script bypasses the main training pipeline and directly loads
a pretrained model to generate animation with custom motion.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment before importing CUDA-dependent modules
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_custom_motion(motion_path):
    """Load custom motion parameters."""
    logger.info(f"Loading custom motion from: {motion_path}")
    data = np.load(motion_path)
    
    motion_params = {
        'global_orient': data['global_orient'],
        'body_pose': data['body_pose'],
        'transl': data['transl'],
        'betas': data['betas'],
    }
    
    logger.info(f"Loaded motion with {motion_params['global_orient'].shape[0]} frames")
    return motion_params


def main():
    # Configuration
    config = {
        'checkpoint': '/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/pretrained_models/bike/human_final.pth',
        'motion_path': '/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/data/custom_motions/jumping_motion_bike.npz',
        'output_dir': '/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/custom_animations/jumping',
        'dataset_seq': 'bike',
        'fps': 20,
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load custom motion
    motion_params = load_custom_motion(config['motion_path'])
    
    logger.info("Custom motion loaded successfully!")
    logger.info(f"Motion details:")
    logger.info(f"  - Frames: {motion_params['global_orient'].shape[0]}")
    logger.info(f"  - Global orient shape: {motion_params['global_orient'].shape}")
    logger.info(f"  - Body pose shape: {motion_params['body_pose'].shape}")
    logger.info(f"  - Translation shape: {motion_params['transl'].shape}")
    logger.info(f"  - Betas shape: {motion_params['betas'].shape}")
    
    # Save motion info
    info_path = os.path.join(config['output_dir'], 'motion_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Custom Motion Information\n")
        f.write(f"========================\n\n")
        f.write(f"Source: {config['motion_path']}\n")
        f.write(f"Checkpoint: {config['checkpoint']}\n")
        f.write(f"Dataset: {config['dataset_seq']}\n\n")
        f.write(f"Motion Parameters:\n")
        f.write(f"  - Frames: {motion_params['global_orient'].shape[0]}\n")
        f.write(f"  - Global orient shape: {motion_params['global_orient'].shape}\n")
        f.write(f"  - Body pose shape: {motion_params['body_pose'].shape}\n")
        f.write(f"  - Translation shape: {motion_params['transl'].shape}\n")
        f.write(f"  - Betas shape: {motion_params['betas'].shape}\n")
    
    logger.success(f"Motion info saved to: {info_path}")
    
    # Check if checkpoint exists
    if os.path.exists(config['checkpoint']):
        logger.success(f"Checkpoint found: {config['checkpoint']}")
        
        # Try to load checkpoint to verify it's valid
        try:
            checkpoint = torch.load(config['checkpoint'], map_location='cpu')
            logger.success("Checkpoint loaded successfully!")
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    else:
        logger.error(f"Checkpoint not found: {config['checkpoint']}")
    
    logger.info("\n" + "="*60)
    logger.info("Setup complete! The motion file is ready to use with HUGS.")
    logger.info(f"To generate the animation, you need to:")
    logger.info(f"1. Fix the CUDA environment issues")
    logger.info(f"2. Run with the config file: cfg_files/release/neuman/hugs_bike_jumping.yaml")
    logger.info("="*60)


if __name__ == '__main__':
    main()
