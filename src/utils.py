"""
Enhanced Utility Functions with 4GB GPU Optimization
"""

import os
import json
import logging
import gc
import torch
from datetime import datetime
from pathlib import Path

class UnicodeFilter(logging.Filter):
    """Filter to replace Unicode characters that Windows can't display"""
    def filter(self, record):
        replacements = {
            '🚀': '[START]',
            '📸': '[CAMERA]',
            '📋': '[PLAN]',
            '📍': '[LOCATION]',
            '🎯': '[TARGET]',
            '📝': '[PROMPT]',
            '🎨': '[ART]',
            '✨': '[ENHANCE]',
            '🔍': '[CHECK]',
            '💾': '[SAVE]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '💻': '[COMPUTER]',
            '🖥️': '[GPU]',
            '💾': '[MEMORY]',
            '🔄': '[LOADING]',
            '📊': '[STATS]',
            '⏱️': '[TIME]',
            '🎊': '[SUCCESS]',
            '🧪': '[TEST]',
            '🔧': '[SETUP]',
            '📁': '[FOLDER]',
            '📄': '[FILE]',
            '🎉': '[CELEBRATE]',
            '💡': '[TIP]'
        }
        
        if record.msg:
            for emoji, text in replacements.items():
                record.msg = record.msg.replace(emoji, text)
        
        return True

def setup_logging():
    """Setup logging configuration with Unicode fix"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler (supports Unicode)
    file_handler = logging.FileHandler(f"{log_dir}/generation_log.txt", encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream handler with Unicode filter for Windows
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(UnicodeFilter())  # Add our Unicode filter
    logger.addHandler(stream_handler)
    
    return logging.getLogger(__name__)

def save_metadata(location, metadata):
    """Save metadata for generated images"""
    metadata_path = f"outputs/{location}/metadata.json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Add timestamp and device info
    metadata['generation_timestamp'] = datetime.now().isoformat()
    metadata['device_used'] = "GPU" if torch.cuda.is_available() else "CPU"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logging.info("[SAVE] Metadata saved: %s", metadata_path)

def clear_gpu_memory():
    """Clear GPU memory aggressively for 4GB GPUs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Additional aggressive cleanup
        try:
            import torch
            if hasattr(torch, 'cuda'):
                torch.cuda.synchronize()
                # Force garbage collection again
                gc.collect()
        except:
            pass

def get_gpu_memory_info():
    """Get detailed GPU memory usage information"""
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            return {
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'max_allocated_gb': round(max_allocated, 2),
                'utilization': f"{(allocated/reserved)*100:.1f}%" if reserved > 0 else "0%"
            }
        except Exception as e:
            logging.warning("[WARN] Failed to get GPU memory info: %s", e)
    
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0, 'utilization': '0%'}

def optimize_memory_usage():
    """Optimize memory usage for 4GB GPUs"""
    if torch.cuda.is_available():
        # Set smaller memory fraction for 4GB GPU
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of VRAM
        
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        gc.collect()
        
        logging.info("[MEMORY] Optimized for 4GB GPU (70% VRAM limit)")

def load_config_file(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("[WARN] Config file not found: %s", config_path)
        return {}

def ensure_directory(path):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)
    return path

def calculate_image_stats(image_path):
    """Calculate comprehensive image statistics"""
    from PIL import Image
    import numpy as np
    
    try:
        image = Image.open(image_path)
        img_array = np.array(image)
        
        stats = {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode,
            'mean_intensity': float(np.mean(img_array)),
            'std_intensity': float(np.std(img_array)),
            'min_intensity': float(np.min(img_array)),
            'max_intensity': float(np.max(img_array)),
            'file_size_kb': os.path.getsize(image_path) // 1024
        }
        
        return stats
        
    except Exception as e:
        logging.error("[ERROR] Failed to calculate image stats: %s", str(e))
        return {}

def format_prompt_for_display(prompt, max_length=100):
    """Format prompt for display purposes"""
    if len(prompt) > max_length:
        return prompt[:max_length] + "..."
    return prompt

def get_model_info():
    """Get information about loaded models"""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    
    return info