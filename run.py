#!/usr/bin/env python3
"""
Simple Optimized Runner for 4GB GPUs
"""

import os
import sys

# Set memory optimizations BEFORE any imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYTORCH_CUDA_MEMORY_FRACTION'] = '0.7'

def main():
    print("🚀 Starting ULTRA-OPTIMIZED pipeline for 4GB GPU...")
    print("📋 Configuration:")
    print("   - Model: Stable Diffusion 1.5 (smallest)")
    print("   - Resolution: 512x512")
    print("   - Steps: 15")
    print("   - Memory: 70% VRAM limit")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        from main import ThreeViewImagePipeline
        from utils import setup_logging
        
        # Setup logging
        logger = setup_logging()
        
        # Initialize pipeline
        pipeline = ThreeViewImagePipeline()
        
        # Force ultra-optimized settings
        pipeline.config.GENERATION_MODEL = "runwayml/stable-diffusion-v1-5"
        pipeline.config.IMAGE_WIDTH = 512
        pipeline.config.IMAGE_HEIGHT = 512
        pipeline.config.NUM_INFERENCE_STEPS = 15
        pipeline.config.GUIDANCE_SCALE = 7.0
        pipeline.config.ENHANCEMENT_MODEL = "PIL"
        pipeline.config.VALIDATION_MODEL = "ViT-B/32"
        
        logger.info("🎯 ULTRA-OPTIMIZED settings applied for 4GB GPU")
        
        # Run the pipeline
        pipeline.run()
        
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)