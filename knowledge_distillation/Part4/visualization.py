"""
Complete Example: GradCAM Generation + Visualization
====================================================

Shows how to:
1. Load trained models
2. Generate GradCAM heatmaps
3. Visualize with FIXED functions (no pink!)
4. Compare models side-by-side
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import from complete GradCAM file
from gradcam_visualization import LocalizationAnalyzer, visualize_single_model, visualize_model_comparison

# Import your model definitions
from Part1.utils import vgg11


def main():
    """Main function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: Load validation data
    # ════════════════════════════════════════════════════════════════════════
    print("Loading CIFAR-100 validation set...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    # Use subset for faster testing
    num_samples = 100  # Change this to use more samples
    subset_indices = list(range(min(num_samples, len(val_dataset))))
    subset_dataset = Subset(val_dataset, subset_indices)
    
    val_loader = DataLoader(
        subset_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(subset_dataset)} validation samples\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # STEP 2: Load models
    # ════════════════════════════════════════════════════════════════════════
    print("Loading models...")
    
    model_configs = {
        'SI': './checkpoints/si_student_best.pth',
        'LM': './checkpoints/lm_student_best.pth',
        'LS': './checkpoints/ls_student_best.pth',
    }
    
    loaded_models = {}
    
    for model_name, checkpoint_path in model_configs.items():
        try:
            # Create model
            model = vgg11(num_classes=100).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # Enable gradients for GradCAM
            for param in model.parameters():
                param.requires_grad = True
            
            loaded_models[model_name] = model
            print(f"✓ Loaded {model_name}")
            
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")
    
    print()
    
    # ════════════════════════════════════════════════════════════════════════
    # STEP 3: Generate GradCAM heatmaps
    # ════════════════════════════════════════════════════════════════════════
    print("="*80)
    print("GENERATING GRADCAM HEATMAPS")
    print("="*80 + "\n")
    
    analyzer = LocalizationAnalyzer(device=device)
    
    for model_name, model in loaded_models.items():
        print(f"\nGenerating CAMs for {model_name}...")
        
        # Choose target layer (use -3 instead of -1 for better spatial resolution)
        target_layer = model.features[-3]
        
        # Generate CAMs
        analyzer.generate_localizations(
            model=model,
            model_name=model_name,
            data_loader=val_loader,
            target_layer=target_layer,
            num_samples=num_samples
        )
    
    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: Visualize GradCAMs
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("VISUALIZING GRADCAM HEATMAPS")
    print("="*80 + "\n")
    
    # Select images to visualize
    num_viz = 5
    viz_indices = list(range(min(num_viz, len(subset_dataset))))
    
    # Option A: Visualize each model individually
    print("Visualizing individual models...")
    for model_name in loaded_models.keys():
        visualize_single_model(
            analyzer,
            model_name,
            viz_indices,
            save_dir=f'./results/gradcam/{model_name}'
        )
    
    # Option B: Compare models side-by-side
    print("\nVisualizing model comparison...")
    visualize_model_comparison(
        analyzer,
        list(loaded_models.keys()),
        viz_indices,
        save_dir='./results/gradcam/comparison'
    )
    
    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: Summary
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    
    print("\nGenerated CAM shapes:")
    for model_name, results in analyzer.results.items():
        print(f"  {model_name}: {results['cams'].shape}")
    
    print("\nOutput saved to:")
    print("  ./results/gradcam/{model_name}/")
    print("  ./results/gradcam/comparison/")
    
    print("\n✅ You should see colorful Blue→Red heatmaps (NOT pink rectangles)!")
    print("   - Blue areas: model NOT focusing")
    print("   - Red areas: model IS focusing")


if __name__ == "__main__":
    main()