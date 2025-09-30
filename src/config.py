"""
Enhanced Configuration with YAML Support and Low VRAM Optimization
"""

import torch
import yaml
from pathlib import Path

class Config:
    """Configuration class optimized for 4GB GPUs"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.yaml_config = self.load_yaml_config()
        
        # Device setup
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.TORCH_DTYPE = torch.float16 if self.DEVICE == "cuda" else torch.float32
        
        # FORCE SD 1.5 for 4GB GPU (smaller than SD 2.1)
        self.GENERATION_MODEL = "runwayml/stable-diffusion-v1-5"
        self.ENHANCEMENT_MODEL = "PIL"  # Lightweight, no GPU memory
        self.VALIDATION_MODEL = "ViT-B/32"  # Smaller CLIP model
        
        # View definitions from YAML
        self.VIEWS = list(self.yaml_config.get('views', {}).keys())
        self.VIEW_DISPLAY_NAMES = {
            view: data.get('display_name', view.replace('_', ' ').title())
            for view, data in self.yaml_config.get('views', {}).items()
        }
        
        # Generation Parameters optimized for 4GB GPU
        self.IMAGE_WIDTH = 512  # Force 512x512 for 4GB
        self.IMAGE_HEIGHT = 512
        self.NUM_INFERENCE_STEPS = 15  # Fewer steps for 4GB
        self.GUIDANCE_SCALE = 7.0  # Lower guidance scale
        self.SEED = 42
        
        # Disable prompt enhancement for memory savings
        self.PROMPT_ENHANCEMENT = False
        
        # Enhancement Parameters
        self.UPSCALE_FACTOR = 2
        self.ENHANCEMENT_METHOD = 'pil'
        
        # Validation Parameters
        self.CONSISTENCY_THRESHOLD = 0.25
        
        # Performance settings optimized for 4GB GPU
        self.ENABLE_ATTENTION_SLICING = True
        self.ENABLE_XFORMERS = True
        self.ENABLE_MEMORY_EFFICIENT_ATTENTION = True
        self.ENABLE_VAE_SLICING = True  # Critical for 4GB
        self.USE_FP16 = True
        self.MAX_BATCH_SIZE = 1
        
        # Path Settings
        self.BASE_DIR = Path("image_generation_project")
        self.OUTPUT_DIR = Path("outputs")
        self.MODEL_CACHE_DIR = Path("models")
        self.LOG_DIR = Path("logs")
        
        self.validate_paths()
        self.print_system_info()
    
    def load_yaml_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file {self.config_path} not found, using optimized defaults for 4GB GPU")
            return {}
        except Exception as e:
            print(f"⚠️  Error loading YAML config: {e}, using optimized defaults")
            return {}
    
    def validate_paths(self):
        """Ensure all required paths exist"""
        paths_to_create = [
            self.OUTPUT_DIR,
            self.MODEL_CACHE_DIR,
            self.LOG_DIR
        ]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
    
    def print_system_info(self):
        """Print system information"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("[SYSTEM] Configuration initialized for 4GB GPU optimization")
        
        if self.DEVICE == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("[GPU] %s (%.1f GB VRAM) - OPTIMIZED", gpu_name, vram)
        else:
            logger.info("[CPU] Using CPU for generation")
        
        logger.info("[MODELS] Generation: %s (4GB optimized)", self.GENERATION_MODEL)
        logger.info("[MODELS] Enhancement: %s", self.ENHANCEMENT_MODEL)
        logger.info("[MODELS] Validation: %s", self.VALIDATION_MODEL)
        logger.info("[SETTINGS] Resolution: %dx%d (4GB optimized)", self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        logger.info("[SETTINGS] Inference steps: %d", self.NUM_INFERENCE_STEPS)
        logger.info("[OPTIMIZATIONS] VAE slicing: %s", self.ENABLE_VAE_SLICING)
        logger.info("[OPTIMIZATIONS] Attention slicing: %s", self.ENABLE_ATTENTION_SLICING)

# Global configuration instance
config = Config()