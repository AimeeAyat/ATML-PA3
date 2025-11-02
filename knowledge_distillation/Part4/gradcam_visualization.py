"""
Complete GradCAM Solution - Generation + Fixed Visualization
=============================================================

Contains:
1. GradCAM class - generates heatmaps
2. LocalizationAnalyzer - processes all models
3. Fixed visualization - correct colorful heatmaps (not pink!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════════
# 1. GRADCAM CLASS - Generate Heatmaps
# ════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM).
    Generates spatial heatmaps showing which regions activate the model.
    """
    
    def __init__(self, model, target_layer, device='cuda'):
        """
        Args:
            model: Neural network model
            target_layer: Layer to generate CAM for (e.g., model.features[-3])
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Forward hook to save activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate GradCAM heatmap for single image.
        
        Args:
            input_tensor: (1, 3, H, W) input image
            class_idx: Class to generate CAM for (default: predicted class)
            
        Returns:
            cam: (H, W) heatmap, normalized to [0, 1]
            pred_class: Predicted class index
        """
        self.model.eval()
        
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.requires_grad_(True)
        
        # First pass: get predicted class without gradients
        with torch.no_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
        
        # Second pass: generate gradients
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU to keep only positive activations
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy(), class_idx
    
    def generate_cam_batch(self, input_tensor):
        """
        Generate GradCAM for batch of images.
        
        Args:
            input_tensor: (B, 3, H, W) batch of images
            
        Returns:
            cams: (B, H_feat, W_feat) heatmaps
            classes: (B,) predicted class indices
        """
        cams = []
        classes = []
        
        for i in range(input_tensor.shape[0]):
            img = input_tensor[i:i+1].detach().clone()
            
            try:
                cam, cls = self.generate_cam(img)
                cams.append(cam)
                classes.append(cls)
            except RuntimeError as e:
                print(f"⚠ Error processing image {i}: {e}")
                cams.append(np.zeros((input_tensor.shape[2], input_tensor.shape[3])))
                classes.append(-1)
        
        return np.array(cams), np.array(classes)


# ════════════════════════════════════════════════════════════════════════════
# 2. LOCALIZATION ANALYZER - Process All Models
# ════════════════════════════════════════════════════════════════════════════

class LocalizationAnalyzer:
    """Analyze spatial attention patterns across models"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.gradcams = {}
        self.results = {}
    
    def generate_localizations(self, model, model_name, data_loader, 
                              target_layer, num_samples=None):
        """
        Generate GradCAM heatmaps for all images.
        
        Args:
            model: Trained model
            model_name: Name of model (for labeling)
            data_loader: DataLoader with images and labels
            target_layer: Layer to generate CAM from (e.g., model.features[-3])
            num_samples: Max samples to process (None = all)
            
        Returns:
            results: {
                'cams': (N, H_feat, W_feat) array,
                'images': (N, 3, H, W) images,
                'predictions': (N,) predicted classes,
                'targets': (N,) ground truth classes,
                'accuracy': float
            }
        """
        model.eval()
        gradcam = GradCAM(model, target_layer, device=self.device)
        
        all_cams = []
        all_images = []
        all_preds = []
        all_targets = []
        
        sample_count = 0
        
        for images, targets in tqdm(data_loader, desc=f"Generating CAMs for {model_name}"):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Generate CAMs
            cams, preds = gradcam.generate_cam_batch(images)
            
            all_cams.extend(cams)
            all_images.extend(images.cpu().detach().numpy())
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            sample_count += len(images)
            if num_samples is not None and sample_count >= num_samples:
                all_cams = all_cams[:num_samples]
                all_images = all_images[:num_samples]
                all_preds = all_preds[:num_samples]
                all_targets = all_targets[:num_samples]
                break
        
        all_cams = np.array(all_cams)
        all_images = np.array(all_images)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        correct = (all_preds == all_targets).astype(int)
        accuracy = np.mean(correct)
        
        results = {
            'cams': all_cams,
            'images': all_images,
            'predictions': all_preds,
            'targets': all_targets,
            'accuracy': accuracy
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name}:")
        print(f"  Generated {len(all_cams)} CAMs")
        print(f"  CAM shape: {all_cams.shape}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return results


# ════════════════════════════════════════════════════════════════════════════
# 3. FIXED VISUALIZATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def visualize_single_model(analyzer, model_name, image_indices, save_dir='./results/gradcam'):
    """
    Visualize GradCAM for single model.
    
    Args:
        analyzer: LocalizationAnalyzer with generated CAMs
        model_name: Name of model to visualize
        image_indices: List of image indices to show
        save_dir: Directory to save images
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if model_name not in analyzer.results:
        print(f"❌ Model {model_name} not found")
        return
    
    results = analyzer.results[model_name]
    cams = results['cams']
    images = results['images']
    predictions = results['predictions']
    targets = results['targets']
    
    # Denormalization for CIFAR-100
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    
    for idx in image_indices:
        if idx >= len(cams):
            print(f"⚠ Index {idx} out of range (max: {len(cams)-1})")
            continue
        
        # Get data
        cam = cams[idx]
        img = images[idx]
        pred = predictions[idx]
        target = targets[idx]
        is_correct = pred == target
        
        # Denormalize image
        img_display = img * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
        img_display = np.clip(img_display, 0, 1)
        img_display = np.transpose(img_display, (1, 2, 0))
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Original
        axes[0].imshow(img_display)
        title_color = 'green' if is_correct else 'red'
        axes[0].set_title(f'Original\nPred: {pred}, GT: {target}', 
                         fontweight='bold', color=title_color, fontsize=11)
        axes[0].axis('off')
        
        # Right: CAM overlay - ✅ THE FIX IS HERE
        axes[1].imshow(img_display)
        
        try:
            # ✅ STEP 1: Ensure CAM is 2D
            cam_viz = cam.copy()
            while cam_viz.ndim > 2:
                cam_viz = np.squeeze(cam_viz)
            if cam_viz.ndim < 2:
                cam_viz = np.atleast_2d(cam_viz)
            
            # ✅ STEP 2: Resize to image size
            img_h, img_w = img_display.shape[:2]
            cam_h, cam_w = cam_viz.shape[:2]
            if (cam_h, cam_w) != (img_h, img_w):
                cam_viz = cv2.resize(cam_viz, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            
            # ✅ STEP 3: Normalize to [0, 1]
            cam_min, cam_max = cam_viz.min(), cam_viz.max()
            if cam_max > cam_min:
                cam_viz = (cam_viz - cam_min) / (cam_max - cam_min)
            else:
                cam_viz = np.ones_like(cam_viz) * 0.5
            
            # ✅ STEP 4: Apply colormap
            cmap = cm.get_cmap('jet')
            cam_colored = cmap(cam_viz)  # (H, W, 4) RGBA
            
            # ✅ STEP 5: Overlay
            axes[1].imshow(cam_colored, alpha=0.5)
            
        except Exception as e:
            print(f"Error: {e}")
            axes[1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                        transform=axes[1].transAxes, color='red')
        
        axes[1].set_title(f'GradCAM - {model_name}', fontweight='bold', fontsize=11)
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = f'{save_dir}/{model_name}_sample_{idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()


def visualize_model_comparison(analyzer, model_names, image_indices, save_dir='./results/gradcam'):
    """
    Compare GradCAM across multiple models side-by-side.
    
    Args:
        analyzer: LocalizationAnalyzer with generated CAMs
        model_names: List of model names to compare
        image_indices: List of image indices to show
        save_dir: Directory to save images
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Check all models exist
    for model_name in model_names:
        if model_name not in analyzer.results:
            print(f"⚠ Model {model_name} not found")
            return
    
    # Denormalization
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    
    for idx in image_indices:
        # Check index in all models
        for model_name in model_names:
            if idx >= len(analyzer.results[model_name]['cams']):
                print(f"⚠ Index {idx} out of range for {model_name}")
                return
        
        # Create figure with columns: original + one per model
        fig, axes = plt.subplots(1, len(model_names) + 1, 
                                figsize=(4*(len(model_names)+1), 4))
        
        # Get image from first model
        img = analyzer.results[model_names[0]]['images'][idx]
        img_display = img * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
        img_display = np.clip(img_display, 0, 1)
        img_display = np.transpose(img_display, (1, 2, 0))
        
        # Column 0: Original image
        axes[0].imshow(img_display)
        axes[0].set_title('Original', fontweight='bold', fontsize=10)
        axes[0].axis('off')
        
        # Other columns: CAM from each model - ✅ FIX APPLIED
        for col, model_name in enumerate(model_names):
            ax = axes[col + 1]
            ax.imshow(img_display)
            
            cam = analyzer.results[model_name]['cams'][idx]
            pred = analyzer.results[model_name]['predictions'][idx]
            target = analyzer.results[model_name]['targets'][idx]
            is_correct = pred == target
            
            try:
                # ✅ FIX: Proper CAM visualization
                cam_viz = cam.copy()
                while cam_viz.ndim > 2:
                    cam_viz = np.squeeze(cam_viz)
                if cam_viz.ndim < 2:
                    cam_viz = np.atleast_2d(cam_viz)
                
                img_h, img_w = img_display.shape[:2]
                cam_h, cam_w = cam_viz.shape[:2]
                if (cam_h, cam_w) != (img_h, img_w):
                    cam_viz = cv2.resize(cam_viz, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                cam_min, cam_max = cam_viz.min(), cam_viz.max()
                if cam_max > cam_min:
                    cam_viz = (cam_viz - cam_min) / (cam_max - cam_min)
                else:
                    cam_viz = np.ones_like(cam_viz) * 0.5
                
                cmap = cm.get_cmap('jet')
                cam_colored = cmap(cam_viz)
                ax.imshow(cam_colored, alpha=0.5)
                
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                       transform=ax.transAxes, color='red')
            
            title_color = 'green' if is_correct else 'red'
            ax.set_title(f'{model_name}\nP:{pred} T:{target}', 
                        fontweight='bold', color=title_color, fontsize=9)
            ax.axis('off')
        
        plt.tight_layout()
        save_path = f'{save_dir}/comparison_sample_{idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()