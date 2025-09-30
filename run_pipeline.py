#!/usr/bin/env python3
"""
Single-Click Image Generation Pipeline Runner
Enhanced with Best Open-Source Models and Flexible Location Support
"""

import os
import sys
import argparse
import time
from pathlib import Path

def main():
    """Main execution function"""
    # Parse command line arguments FIRST
    parser = argparse.ArgumentParser(
        description='Generate professional images for any locations with three views each',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --quick              # Fast generation with default locations
  python run_pipeline.py --manual             # Custom prompts for each view
  python run_pipeline.py --custom-locations   # Add your own locations
  python run_pipeline.py --quality            # Highest quality (slower)
  python run_pipeline.py --quick --manual     # Fast with custom prompts
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 20 inference steps, faster generation')
    parser.add_argument('--quality', action='store_true',
                       help='Quality mode: 40 inference steps, best results')
    parser.add_argument('--manual', action='store_true',
                       help='Manual mode: Custom prompt input for each view')
    parser.add_argument('--custom-locations', action='store_true',
                       help='Use custom locations instead of defaults')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency verification')
    parser.add_argument('--monitor-gpu', action='store_true',
                       help='Monitor GPU memory during generation')
    parser.add_argument('--estimate-time', action='store_true',
                       help='Estimate generation time and exit')
    
    args = parser.parse_args()
    
    # Now add src to path and import modules
    sys.path.append('src')
    
    from main import ThreeViewImagePipeline
    from utils import setup_logging, get_gpu_memory_info
    
    # Display welcome banner
    display_welcome_banner()
    
    # Check dependencies
    if not args.skip_deps:
        print("\n📦 Checking dependencies...")
        missing = check_environment()
        if missing:
            print("❌ Missing required packages:")
            for package in missing:
                print(f"   - {package}")
            print("\n💡 Install with: pip install -r requirements.txt")
            return 1
        print("✅ All dependencies available")
    
    # Check GPU
    gpu_available, gpu_info = check_gpu_availability()
    print(f"\n💻 Hardware: {gpu_info}")
    
    if gpu_available:
        mem_info = get_gpu_memory_info()
        print(f"💾 GPU Memory: {mem_info['allocated_gb']}GB used, {mem_info['reserved_gb']}GB reserved")
    
    # Display model information
    display_model_info()
    
    # Setup environment
    setup_environment()
    
    # Estimate time if requested
    if args.estimate_time:
        location_count = 3  # Default estimate
        mode = "quality" if args.quality else "quick" if args.quick else "standard"
        minutes, seconds = estimate_generation_time(location_count, mode)
        print(f"\n⏱️  Estimated time: {minutes}m {seconds}s for {location_count} locations")
        return 0
    
    # Display generation plan
    print("\n📋 GENERATION PLAN:")
    print("   📍 Location Mode:", "Custom Locations" if args.custom_locations else "Default Locations")
    print("   💬 Prompt Mode:", "Manual Input" if args.manual else "Auto-Generated")
    print("   🎯 Views per Location: Aerial, Side, Close-up")
    print("   ⚡ Speed:", "Quality" if args.quality else "Quick" if args.quick else "Standard")
    
    # Show location examples
    if args.custom_locations:
        print("\n💡 You can add locations like:")
        print("   - Eiffel Tower, Paris")
        print("   - Grand Canyon, USA") 
        print("   - Your local landmark")
        print("   - Any place you can describe!")
    else:
        print("\n📍 Default Locations:")
        print("   - Mumbai Bandra-Worli Sea Link")
        print("   - Rajasthan Hawa Mahal")
        print("   - Nagpur Rainforest")
    
    # Calculate estimated images
    location_count = "variable" if args.custom_locations else 3
    total_images = "variable" if args.custom_locations else "9"
    print(f"   📸 Total Images: {total_images} ({location_count} locations × 3 views)")
    
    # Estimate time
    location_count_num = 3 if not args.custom_locations else 3  # Base estimate
    mode = "quality" if args.quality else "quick" if args.quick else "standard"
    minutes, seconds = estimate_generation_time(location_count_num, mode)
    print(f"   ⏱️  Estimated Time: {minutes}m {seconds}s")
    
    # Warning for manual mode
    if args.manual:
        print("\n💡 MANUAL MODE: You will enter custom prompts for each view.")
        print("   Press Enter to use auto-generated prompts for any view.")
    
    # Warning for custom locations
    if args.custom_locations:
        print("\n💡 CUSTOM LOCATIONS: You can add as many locations as you want.")
        print("   For each location, provide a name and description.")
    
    # Confirm execution
    print("\n" + "="*70)
    try:
        if not args.quick:  # Don't prompt in quick mode
            input("🎯 Press Enter to start generation (Ctrl+C to cancel)...")
        else:
            print("🎯 Starting generation in quick mode...")
            time.sleep(2)  # Brief pause so user can read
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user")
        return 0
    
    # Start pipeline
    start_time = time.time()
    print("\n" + "="*70)
    print("🚀 STARTING IMAGE GENERATION PIPELINE")
    print("="*70)
    
    try:
        # Initialize and configure pipeline
        pipeline = ThreeViewImagePipeline()
        
        # Apply configuration based on arguments
        if args.manual:
            pipeline.manual_mode = True
            print("[CONFIG] Manual prompt mode enabled")
        
        if args.custom_locations:
            pipeline.custom_locations = True
            print("[CONFIG] Custom location mode enabled")
        
        if args.quick:
            from config import Config
            pipeline.config.NUM_INFERENCE_STEPS = 20
            print("[CONFIG] Quick mode: 20 inference steps")
        
        if args.quality:
            from config import Config
            pipeline.config.NUM_INFERENCE_STEPS = 40
            print("[CONFIG] Quality mode: 40 inference steps")
        
        if args.monitor_gpu and gpu_available:
            print("[CONFIG] GPU monitoring enabled")
        
        # Run the pipeline
        pipeline.run()
        
        # Calculate and display final statistics
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️  Total execution time: {minutes}m {seconds}s")
        
        if gpu_available:
            final_mem = get_gpu_memory_info()
            print(f"💾 Final GPU memory: {final_mem['allocated_gb']}GB allocated")
        
        # Display output information
        print("\n📁 OUTPUT FILES:")
        print("   📸 Generated images: outputs/[location]/")
        print("   📊 Validation scores: results/consistency_scores/")
        print("   📄 Summary report: results/final_evaluation_summary.md")
        print("   📋 Detailed logs: logs/generation_log.txt")
        
        print("\n✨ Check the 'outputs/' folder for your generated images!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Pipeline execution failed: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check GPU memory - try --quick mode")
        print("   2. Check internet connection for model downloads")
        print("   3. Verify all dependencies are installed")
        print("   4. Check logs/generation_log.txt for details")
        return 1
    
    return 0

def check_environment():
    """Check if all required dependencies are available"""
    print("🔍 Checking environment...")
    
    required_packages = [
        'torch',
        'transformers', 
        'diffusers',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL.Image')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_gpu_availability():
    """Check GPU availability and capabilities"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"{gpu_name} ({vram:.1f} GB)"
        return False, "No GPU available"
    except ImportError:
        return False, "PyTorch not installed"

def setup_environment():
    """Setup the environment for the pipeline"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        'outputs',
        'models',
        'logs', 
        'inputs/reference_images',
        'results/consistency_scores',
        'results/quality_assessment',
        'results/reference_comparison'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def display_welcome_banner():
    """Display welcome banner with feature information"""
    print("\n" + "="*70)
    print("🚀 ENHANCED IMAGE GENERATION PIPELINE")
    print("="*70)
    print("✨ Features:")
    print("   ✅ Best Open-Source Models (SD 2.1, SD Upscaler, CLIP Large)")
    print("   ✅ Any Locations Worldwide - Not Limited to Predefined")
    print("   ✅ Three Professional Views: Aerial, Side, Close-up")
    print("   ✅ Manual or Auto Prompt Modes")
    print("   ✅ Reference Image Integration")
    print("   ✅ Comprehensive Quality Validation")
    print("="*70)

def display_model_info():
    """Display information about the models being used"""
    from config import Config
    config = Config()
    
    print("\n🤖 MODEL INFORMATION:")
    print("   🎨 Generation:", config.GENERATION_MODEL)
    print("   ✨ Enhancement:", config.ENHANCEMENT_MODEL)
    print("   🔍 Validation:", config.VALIDATION_MODEL)
    print("   📐 Resolution:", f"{config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}")

def estimate_generation_time(location_count, mode):
    """Estimate generation time based on settings"""
    base_time_per_image = 45 if mode == "quality" else 25 if mode == "quick" else 35
    total_images = location_count * 3  # 3 views per location
    total_seconds = total_images * base_time_per_image
    
    # Add model loading time
    total_seconds += 120  # 2 minutes for model loading
    
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return minutes, seconds

def quick_start_guide():
    """Display quick start guide"""
    print("\n" + "="*70)
    print("🚀 QUICK START GUIDE")
    print("="*70)
    print("1. Basic usage (fastest):")
    print("   python run_pipeline.py --quick")
    print()
    print("2. Best quality (slower):")
    print("   python run_pipeline.py --quality")
    print()
    print("3. Custom locations:")
    print("   python run_pipeline.py --custom-locations")
    print()
    print("4. Manual prompts:")
    print("   python run_pipeline.py --manual")
    print()
    print("5. Estimate time only:")
    print("   python run_pipeline.py --estimate-time")
    print("="*70)

if __name__ == "__main__":
    # Show quick start guide if no arguments
    if len(sys.argv) == 1:
        quick_start_guide()
        print("\n💡 Run with --help for all options")
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)