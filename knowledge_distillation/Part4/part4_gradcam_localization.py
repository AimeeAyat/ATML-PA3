"""
FIXED visualize_cams() Function for Part 4 GradCAM

This version properly handles:
1. CAM shape validation and reshaping
2. Resizing CAMs to image dimensions
3. Proper normalization before colormap
4. Better error handling and debugging
5. Visual distinction between correct/incorrect predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import cv2


def visualize_cams_FIXED(analyzer, model_names, image_indices, save_dir='part4_gradcam_viz',
                         denormalization_fn=None):
    """
    Create visualization comparing CAMs - FIXED VERSION
    
    This version fixes the "pink color" bug by:
    1. Properly reshaping CAM tensors to 2D spatial maps
    2. Resizing to match image dimensions
    3. Normalizing to [0, 1] before colormap application
    4. Better error handling and debugging
    
    Args:
        analyzer: LocalizationAnalyzer instance
        model_names: List of models to visualize
        image_indices: List of image indices to visualize
        save_dir: Directory to save visualizations
        denormalization_fn: Function to denormalize images
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Default denormalization for CIFAR-100
    if denormalization_fn is None:
        def denormalization_fn(img):
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            img = img * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
            return np.clip(img, 0, 1)
    
    for img_idx in image_indices:
        print(f"\n{'='*80}")
        print(f"Visualizing Image {img_idx}")
        print(f"{'='*80}")
        
        fig, axes = plt.subplots(2, len(model_names), figsize=(5*len(model_names), 10))
        if len(model_names) == 1:
            axes = axes.reshape(2, 1)
        
        # Get original image from first available model
        img_np = None
        for model_name in model_names:
            if model_name in analyzer.results:
                img_np = analyzer.results[model_name]['images'][img_idx]
                break
        
        if img_np is None:
            print(f"⚠ Image {img_idx} not found in any model results")
            continue
        
        # Denormalize image
        img_display = denormalization_fn(img_np)
        img_display = np.transpose(img_display, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        
        print(f"Original image shape: {img_display.shape}")
        
        # Plot for each model
        for j, model_name in enumerate(model_names):
            if model_name not in analyzer.results:
                print(f"⚠ Skipping {model_name} - no results")
                continue
            
            results = analyzer.results[model_name]
            
            # ✅ FIX: Safe indexing
            if img_idx >= len(results['cams']):
                print(f"⚠ Image {img_idx} not in {model_name} results (has {len(results['cams'])} samples)")
                continue
            
            cam = results['cams'][img_idx]
            pred = results['predictions'][img_idx]
            target = results['targets'][img_idx]
            is_correct = (pred == target)
            
            print(f"\n{model_name}:")
            print(f"  Prediction: {pred}, Target: {target}, Correct: {is_correct}")
            print(f"  CAM shape: {cam.shape}")
            
            # ─────────────────────────────────────────────────────────────
            # TOP ROW: Original image with prediction
            # ─────────────────────────────────────────────────────────────
            ax = axes[0, j]
            ax.imshow(img_display)
            
            # Color border based on correctness
            title_color = 'green' if is_correct else 'red'
            title_text = f"{model_name}\nPred: {pred}, GT: {target}"
            ax.set_title(title_text, fontweight='bold', color=title_color, fontsize=10)
            ax.axis('off')
            
            # ─────────────────────────────────────────────────────────────
            # BOTTOM ROW: CAM overlay - ✅ FIXED VERSION
            # ─────────────────────────────────────────────────────────────
            ax = axes[1, j]
            ax.imshow(img_display)
            
            try:
                # Step 1: Make a copy and validate shape
                cam_viz = cam.copy() if isinstance(cam, np.ndarray) else np.array(cam)
                print(f"  CAM dtype: {cam_viz.dtype}, min: {cam_viz.min():.4f}, max: {cam_viz.max():.4f}")
                
                # Step 2: ✅ RESHAPE to 2D (H, W)
                # Remove extra dimensions (batch, channel) if present
                while cam_viz.ndim > 2:
                    cam_viz = np.squeeze(cam_viz)
                    print(f"  After squeeze: shape {cam_viz.shape}, ndim {cam_viz.ndim}")
                
                # Handle case where squeeze removed everything
                if cam_viz.ndim < 2:
                    print(f"  ⚠ CAM collapsed to {cam_viz.ndim}D, expanding...")
                    cam_viz = np.atleast_2d(cam_viz)
                
                # Step 3: ✅ RESIZE to match image dimensions
                target_height, target_width = img_display.shape[:2]
                current_height, current_width = cam_viz.shape[:2]
                
                if (current_height, current_width) != (target_height, target_width):
                    print(f"  Resizing CAM from {cam_viz.shape} to {(target_height, target_width)}")
                    cam_viz = cv2.resize(cam_viz, (target_width, target_height), 
                                        interpolation=cv2.INTER_LINEAR)
                
                # Step 4: ✅ NORMALIZE to [0, 1]
                cam_min = cam_viz.min()
                cam_max = cam_viz.max()
                print(f"  Before normalization: min={cam_min:.4f}, max={cam_max:.4f}")
                
                if cam_max > cam_min:
                    cam_viz = (cam_viz - cam_min) / (cam_max - cam_min)
                else:
                    # Handle constant CAM (all same value)
                    print(f"  ⚠ CAM is constant (all values = {cam_min}), using uniform 0.5")
                    cam_viz = np.ones_like(cam_viz) * 0.5
                
                print(f"  After normalization: min={cam_viz.min():.4f}, max={cam_viz.max():.4f}")
                
                # Step 5: ✅ Verify data type and range for colormap
                assert cam_viz.dtype in [np.float32, np.float64], \
                    f"Expected float dtype, got {cam_viz.dtype}"
                assert cam_viz.min() >= 0 and cam_viz.max() <= 1, \
                    f"Expected range [0,1], got [{cam_viz.min()}, {cam_viz.max()}]"
                
                # Step 6: ✅ Apply colormap correctly
                cmap = cm.get_cmap('jet')
                cam_colored = cmap(cam_viz)  # Shape: (H, W, 4) RGBA
                
                print(f"  Colormap output shape: {cam_colored.shape}")
                
                # Step 7: ✅ Overlay on image
                ax.imshow(cam_colored, alpha=0.5)
                
                print(f"  ✓ Successfully visualized CAM")
                
            except Exception as e:
                print(f"  ✗ Error visualizing CAM: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback: show error message on plot
                ax.text(0.5, 0.5, f'CAM Error:\n{str(e)[:50]}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=8, color='red', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f"GradCAM", fontweight='bold', fontsize=10)
            ax.axis('off')
        
        # Save figure
        plt.tight_layout()
        save_path = f'{save_dir}/image_{img_idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {save_path}")
        plt.close()


def diagnose_cam_shapes(analyzer):
    """
    Diagnostic function to check CAM shapes across all models.
    Run this to identify if CAMs are stored correctly.
    """
    print("\n" + "="*80)
    print("CAM SHAPE DIAGNOSTIC")
    print("="*80)
    
    for model_name, results in analyzer.results.items():
        cams = results['cams']
        print(f"\n{model_name}:")
        print(f"  Total CAMs: {len(cams)}")
        print(f"  Full shape: {cams.shape}")
        
        if len(cams) > 0:
            first_cam = cams[0]
            print(f"  Single CAM shape: {first_cam.shape}")
            print(f"  Single CAM dtype: {first_cam.dtype}")
            print(f"  Single CAM min: {first_cam.min():.6f}, max: {first_cam.max():.6f}")
            
            # Check if CAM is too small (1x1)
            if first_cam.ndim >= 2:
                h, w = first_cam.shape[:2]
                if h == 1 and w == 1:
                    print(f"  ❌ ERROR: CAM is spatially collapsed to 1x1!")
                    print(f"  → Solution: Use an earlier layer as target_layer in GradCAM")
                    print(f"  → The current target_layer is too downsampled")
                elif h < 4 or w < 4:
                    print(f"  ⚠ WARNING: CAM is very small ({h}x{w})")
                    print(f"  → Consider using a less-downsampled layer")
                else:
                    print(f"  ✓ CAM spatial dimensions look good ({h}x{w})")


def check_target_layer(model, device='cuda'):
    """
    Diagnostic to find the best target layer for GradCAM.
    Shows feature map sizes at each layer.
    """
    print("\n" + "="*80)
    print("TARGET LAYER DIAGNOSTIC")
    print("="*80)
    
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, 3, 32, 32).to(device)
    
    print("\nFeature map sizes at each layer:")
    print(f"{'Layer':<30} {'Shape':<25} {'Spatial Dims':<20}")
    print("─" * 75)
    
    x = test_input
    for i, layer in enumerate(model.features):
        x = layer(x)
        spatial_dims = x.shape[2:]
        spatial_str = f"{spatial_dims[0]}x{spatial_dims[1]}" if len(spatial_dims) == 2 else str(spatial_dims)
        
        layer_name = str(layer).split('(')[0]
        print(f"Layer {i:<22} {str(x.shape):<25} {spatial_str:<20}")
        
        # Highlight good candidates for target layer
        if spatial_dims[0] >= 4 and spatial_dims[1] >= 4:
            print(f"  ✓ Good candidate for GradCAM")
    
    print("\n⚠ RECOMMENDATIONS:")
    print("  • Choose a layer with spatial dimensions >= 4x4")
    print("  • Avoid last layer if it's heavily downsampled")
    print("  • Example: features[-3] or features[-5]")


# ════════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ════════════════════════════════════════════════════════════════════════════════

"""
In your main script, use like this:

    # After generating CAMs
    from visualize_cams_fixed import visualize_cams_FIXED, diagnose_cam_shapes, check_target_layer
    
    # 1. Check CAM shapes
    diagnose_cam_shapes(analyzer)
    
    # 2. If CAMs are [1,1], check which layer to use
    check_target_layer(model, device)
    
    # 3. Visualize with fixed function
    visualize_cams_FIXED(
        analyzer,
        all_model_names,
        selected_indices,
        save_dir='./results/part4'
    )
"""