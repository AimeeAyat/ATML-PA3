"""
Part 4: GradCAM Localization Analysis - Example Usage (UPDATED)

This script demonstrates how to:
1. Load trained models from Part 1 & 2
2. Generate GradCAM visualizations
3. Compare spatial attention patterns
4. Quantify localization similarity
5. Visualize results

SETUP:
1. Update CHECKPOINT_PATHS with your model locations
2. Verify model classes match your implementations
3. Run: python example_part4_main_updated.py [--quick] [--num-samples N]

KEY CHANGES FROM ORIGINAL:
==========================
1. ✅ Removed torch.no_grad() wrapper - GradCAM NEEDS gradients!
2. ✅ Added explicit gradient enabling for models
3. ✅ Better error handling with try-except blocks
4. ✅ Import from FIXED version of GradCAM
5. ✅ Memory optimization with .detach()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ IMPORTANT: Import from the FIXED version
from part4_gradcam_localization import LocalizationAnalyzer, GradCAM
from Part1.utils import vgg11, vgg16


# ============================================================================
# CHECKPOINT CONFIGURATION - Matches your actual checkpoint files
# ============================================================================

CHECKPOINT_PATHS = {
    'teacher': {
        'path': None,  # Teacher loaded directly from utils, not from checkpoint
        'model_fn': lambda: vgg16(num_classes=100),
        'description': 'VGG-16 (Teacher)'
    },
    'SI': {
        'path': './checkpoints/si_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Independent Student (No Distillation)'
    },
    'LM': {
        'path': './checkpoints/lm_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Logit Matching Student'
    },
    'LS': {
        'path': './checkpoints/ls_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Label Smoothing Student'
    },
    'DKD': {
        'path': './checkpoints/dkd_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Decoupled KD Student'
    },
    'Hints': {
        'path': './checkpoints/hints_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Hints-based Distillation Student'
    },
    'CRD': {
        'path': './checkpoints/crd_student_best.pth',
        'model_fn': lambda: vgg11(num_classes=100),
        'description': 'Contrastive Representation Distillation Student'
    }
}

# ============================================================================
# DATASET SETUP
# ============================================================================

def get_cifar100_loader(split='val', batch_size=32, num_workers=4):
    """Load CIFAR-100 with appropriate preprocessing."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    if split == 'val' or split == 'test':
        dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            transform=transform,
            download=True
        )
    else:
        dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            transform=transform,
            download=True
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, dataset


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_gradcam_analysis(checkpoint_paths, num_samples=None, quick_mode=False):
    """
    Main function to run Part 4 GradCAM analysis.
    
    Args:
        checkpoint_paths: Dictionary of checkpoint configurations
        num_samples: Number of samples to analyze (None = all)
        quick_mode: If True, use only 20 samples for quick testing
    
    CRITICAL CHANGES:
    =================
    1. NO torch.no_grad() wrapper in main loop
    2. GradCAM needs gradients to compute .backward()
    3. Each generate_localizations call handles gradients internally
    4. Better error handling and recovery
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Initialize analyzer
    analyzer = LocalizationAnalyzer(device=device)
    
    # Load validation set
    print("Loading CIFAR-100 validation set...")
    val_loader, val_dataset = get_cifar100_loader('val', batch_size=16)
    
    if quick_mode:
        num_samples = 20
        print(f"Quick mode: Using {num_samples} samples\n")
    
    if num_samples is not None:
        subset_loader = DataLoader(
            Subset(val_dataset, range(min(num_samples, len(val_dataset)))),
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        subset_loader = val_loader
    
    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    print("="*80)
    print("LOADING MODELS")
    print("="*80 + "\n")
    
    models_loaded = {}
    
    for model_key, config in checkpoint_paths.items():
        try:
            # Create model instance
            model = config['model_fn']().to(device)
            
            # Get checkpoint path
            checkpoint_path = config['path']
            
            # Skip if no checkpoint path
            if checkpoint_path is None:
                print(f"⚠ Skipping {model_key} (no checkpoint path)")
                continue
            
            # Check if checkpoint exists
            if not Path(checkpoint_path).exists():
                print(f"⚠ Warning: {checkpoint_path} not found, skipping {model_key}")
                continue
            
            # Load checkpoint
            print(f"  Loading {config['description']}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract model_state_dict if checkpoint contains metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"  ✓ Loaded {config['description']}")
            
            # ✅ CRITICAL: Set model to eval mode
            model.eval()
            
            # ✅ CRITICAL: Enable gradients for all parameters
            # This is REQUIRED for GradCAM to compute gradients via backward()
            for param in model.parameters():
                param.requires_grad = True
            
            # Get target layer (last conv layer) for GradCAM
            target_layer = model.features[-1]
            
            # Create GradCAM object
            gradcam = GradCAM(model, target_layer, device=device)
            
            # Register model and gradcam in analyzer
            analyzer.models[model_key] = model
            analyzer.gradcams[model_key] = gradcam
            
            models_loaded[model_key] = True
            
        except Exception as e:
            print(f"✗ Error loading {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(models_loaded) == 0:
        print("\n✗ ERROR: No models loaded. Cannot proceed.")
        return None, None
    
    print(f"\n✓ Successfully loaded {len(models_loaded)} models for analysis\n")
    
    # ========================================================================
    # GENERATE GRADCAM VISUALIZATIONS
    # ========================================================================
    print("="*80)
    print("GENERATING GRADCAM VISUALIZATIONS")
    print("="*80 + "\n")
    
    all_model_names = list(models_loaded.keys())
    reference_model = None
    reference_results = None
    
    # ✅ KEY FIX: NO torch.no_grad() wrapper here or inside generate_localizations!
    # The GradCAM.generate_cam() function needs gradients to compute backward()
    # Each generate_localizations() call handles gradients internally
    for model_name in all_model_names:
        try:
            print(f"Generating CAMs for {model_name}...")
            
            # This function internally handles gradient computation
            # (two forward passes: one in no_grad to find class, one outside to build graph)
            results = analyzer.generate_localizations(
                model_name,
                subset_loader,
                num_samples=num_samples
            )
            
            # Use first model as reference for sample selection
            if reference_model is None:
                reference_model = model_name
                reference_results = results
            
            print(f"  ✓ Completed\n")
            
        except RuntimeError as e:
            error_msg = str(e)
            
            if "does not require grad" in error_msg:
                print(f"  ✗ GradCAM Error for {model_name}: {e}")
                print(f"     This means gradients aren't enabled. Make sure:")
                print(f"     1. Model parameters have requires_grad=True")
                print(f"     2. Input tensor has requires_grad=True")
                print(f"     3. You're using part4_gradcam_localization_FIXED.py")
            else:
                print(f"  ✗ RuntimeError for {model_name}: {e}")
            
            print(f"     Skipping {model_name}...\n")
            all_model_names.remove(model_name)
            continue
            
        except Exception as e:
            print(f"  ✗ Error for {model_name}: {e}")
            print(f"     Skipping {model_name}...\n")
            import traceback
            traceback.print_exc()
            all_model_names.remove(model_name)
            continue
    
    if reference_model is None:
        print("✗ ERROR: Failed to generate CAMs for any model")
        return None, None
    
    print(f"✓ Successfully generated CAMs for {len(all_model_names)} models\n")
    
    # ========================================================================
    # ANALYZE CAM SIMILARITY
    # ========================================================================
    print("="*80)
    print("ANALYZING CAM SIMILARITY")
    print("="*80)
    
    # Compare all models using the first one as reference
    other_models = [m for m in all_model_names if m != reference_model]
    
    if other_models and reference_model:
        print(f"\nUsing '{reference_model}' as reference model for comparison\n")
        try:
            similarity_analysis = analyzer.analyze_similarity(
                reference_model,
                other_models,
                by_correctness=True
            )
        except Exception as e:
            print(f"✗ Error during similarity analysis: {e}")
            similarity_analysis = {}
    else:
        print("⚠ Not enough models for comparison (need at least 2)")
        similarity_analysis = {}
    
    # ========================================================================
    # CREATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Create results directory
    os.makedirs('./results/part4', exist_ok=True)
    
    # Plot similarity comparisons
    if other_models and similarity_analysis:
        print("Creating similarity comparison plots...")
        try:
            analyzer.plot_similarity_comparison(
                similarity_analysis,
                save_dir='./results/part4'
            )
            print("  ✓ Saved similarity plots\n")
        except Exception as e:
            print(f"  ✗ Error creating plots: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Generate sample CAM visualizations
    print("Generating sample CAM visualizations...")
    
    # Select diverse samples: correct and incorrect predictions
    if reference_results is not None:
        try:
            correct_indices = np.where(reference_results['correct'] == 1)[0]
            incorrect_indices = np.where(reference_results['correct'] == 0)[0]
            
            selected_indices = []
            
            # Pick some correct predictions
            if len(correct_indices) > 0:
                selected_indices.extend(correct_indices[:min(3, len(correct_indices))])
            
            # Pick some incorrect predictions
            if len(incorrect_indices) > 0:
                selected_indices.extend(incorrect_indices[:min(2, len(incorrect_indices))])
            
            if selected_indices:
                print(f"  Visualizing {len(selected_indices)} samples...")
                analyzer.visualize_cams(
                    all_model_names,
                    selected_indices,
                    save_dir='./results/part4'
                )
                print("  ✓ Saved CAM visualizations\n")
            else:
                print("  ⚠ No samples selected for visualization\n")
                
        except Exception as e:
            print(f"  ✗ Error creating visualizations: {e}\n")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # SUMMARY AND EXPORT
    # ========================================================================
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Export results
    try:
        analyzer.export_results('./results/part4/gradcam_analysis_results.json')
        print("\n✓ Exported results\n")
    except Exception as e:
        print(f"\n✗ Error exporting results: {e}\n")
    
    print("Output files:")
    print("  - ./results/part4/gradcam_analysis_results.json")
    print("  - ./results/part4/image_*.png (CAM comparisons)")
    print("  - ./results/part4/cam_*.png (similarity plots)")
    print()
    
    return analyzer, similarity_analysis


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part 4: GradCAM Localization Analysis')
    parser.add_argument('--quick', action='store_true', help='Quick test with 20 samples')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to analyze')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for DataLoader')
    
    args = parser.parse_args()
    
    try:
        analyzer, results = run_gradcam_analysis(
            CHECKPOINT_PATHS,
            num_samples=args.num_samples,
            quick_mode=args.quick
        )
        
        if analyzer is None:
            print("\n✗ Analysis failed. Check error messages above.")
            sys.exit(1)
        else:
            print("✓ Analysis completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)