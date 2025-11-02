"""
Part 3: Analyzing Probability Distribution Alignment
======================================================

This module analyzes how well student models mimic teacher's probability distributions
using KL divergence as a measure of alignment.

Key Analysis:
- Compare probability distributions between teacher and each student model
- Quantify knowledge transfer using KL divergence
- Analyze which models best capture teacher's decision-making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class DistributionAnalyzer:
    """Analyzes probability distribution alignment between teacher and students."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.metrics = {}
        
    def load_model(self, model_name, model_path, model_class):
        """Load a trained model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
        
        model = model_class()
        model.load_state_dict(model_state, strict=False)
        model = model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        print(f"✓ Loaded {model_name} from {model_path}")
        return model
    
    def get_output_distributions(self, model, data_loader, temperature=1.0):
        """
        Extract probability distributions from model outputs.
        
        Args:
            model: Neural network model
            data_loader: DataLoader for test/val data
            temperature: Softening temperature for distributions
            
        Returns:
            distributions: (N, num_classes) array of probability distributions
            logits: (N, num_classes) array of raw logits
            predictions: (N,) array of predicted classes
            targets: (N,) array of ground truth labels
        """
        all_logits = []
        all_probs = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Extracting distributions"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                
                # Store logits
                all_logits.append(outputs.cpu().numpy())
                
                # Compute probability distributions with temperature
                probs = F.softmax(outputs / temperature, dim=1)
                all_probs.append(probs.cpu().numpy())
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        return {
            'distributions': np.vstack(all_probs),
            'logits': np.vstack(all_logits),
            'predictions': np.hstack(all_preds),
            'targets': np.hstack(all_targets)
        }
    
    def compute_kl_divergence(self, p_dist, q_dist):
        """
        Compute KL divergence between two probability distributions.
        KL(P||Q) = Σ p(x) * log(p(x) / q(x))
        
        Args:
            p_dist: Reference distribution (teacher)
            q_dist: Comparison distribution (student)
            
        Returns:
            kl_divs: Array of KL divergences for each sample
            mean_kl: Mean KL divergence
            std_kl: Standard deviation
        """
        # Add small epsilon for numerical stability
        eps = 1e-10
        p_dist = np.clip(p_dist, eps, 1.0)
        q_dist = np.clip(q_dist, eps, 1.0)
        
        # Compute KL divergence per sample
        kl_divs = np.sum(p_dist * (np.log(p_dist) - np.log(q_dist)), axis=1)
        
        return {
            'per_sample': kl_divs,
            'mean': np.mean(kl_divs),
            'std': np.std(kl_divs),
            'median': np.median(kl_divs),
            'min': np.min(kl_divs),
            'max': np.max(kl_divs)
        }
    
    def compute_js_divergence(self, p_dist, q_dist):
        """
        Compute Jensen-Shannon divergence (symmetric version of KL).
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
        
        This is more symmetric and bounded [0, log(2)].
        """
        eps = 1e-10
        p_dist = np.clip(p_dist, eps, 1.0)
        q_dist = np.clip(q_dist, eps, 1.0)
        
        m_dist = 0.5 * (p_dist + q_dist)
        
        kl_pm = np.sum(p_dist * (np.log(p_dist) - np.log(m_dist)), axis=1)
        kl_qm = np.sum(q_dist * (np.log(q_dist) - np.log(m_dist)), axis=1)
        
        js_divs = 0.5 * (kl_pm + kl_qm)
        
        return {
            'per_sample': js_divs,
            'mean': np.mean(js_divs),
            'std': np.std(js_divs),
            'median': np.median(js_divs),
            'min': np.min(js_divs),
            'max': np.max(js_divs)
        }
    
    def compute_hellinger_distance(self, p_dist, q_dist):
        """
        Compute Hellinger distance between probability distributions.
        H(P,Q) = sqrt(0.5 * Σ (sqrt(p) - sqrt(q))^2)
        
        Range: [0, 1] where 0 means identical, 1 means completely different
        """
        eps = 1e-10
        p_dist = np.clip(p_dist, eps, 1.0)
        q_dist = np.clip(q_dist, eps, 1.0)
        
        distances = np.sqrt(0.5 * np.sum((np.sqrt(p_dist) - np.sqrt(q_dist)) ** 2, axis=1))
        
        return {
            'per_sample': distances,
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances),
            'min': np.min(distances),
            'max': np.max(distances)
        }
    
    def compute_wasserstein_distance(self, p_dist, q_dist):
        """
        Compute 1D Wasserstein distance (Earth Mover's Distance).
        
        For each class, compute cumulative distributions and measure distance.
        """
        eps = 1e-10
        p_dist = np.clip(p_dist, eps, 1.0)
        q_dist = np.clip(q_dist, eps, 1.0)
        
        # Compute cumulative distributions
        p_cdf = np.cumsum(p_dist, axis=1)
        q_cdf = np.cumsum(q_dist, axis=1)
        
        # L1 distance between CDFs
        wasserstein_dists = np.mean(np.abs(p_cdf - q_cdf), axis=1)
        
        return {
            'per_sample': wasserstein_dists,
            'mean': np.mean(wasserstein_dists),
            'std': np.std(wasserstein_dists),
            'median': np.median(wasserstein_dists),
            'min': np.min(wasserstein_dists),
            'max': np.max(wasserstein_dists)
        }
    
    def compute_cosine_similarity(self, p_dist, q_dist):
        """
        Compute cosine similarity between probability distributions.
        Similar distributions have similarity close to 1.
        """
        # Normalize distributions
        p_norm = p_dist / (np.linalg.norm(p_dist, axis=1, keepdims=True) + 1e-10)
        q_norm = q_dist / (np.linalg.norm(q_dist, axis=1, keepdims=True) + 1e-10)
        
        # Cosine similarity
        similarities = np.sum(p_norm * q_norm, axis=1)
        
        return {
            'per_sample': similarities,
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'median': np.median(similarities),
            'min': np.min(similarities),
            'max': np.max(similarities)
        }
    
    def analyze_alignment(self, teacher_outputs, student_outputs_dict, 
                         student_names, temperatures=[1.0, 2.0, 4.0, 8.0]):
        """
        Comprehensive analysis of distribution alignment.
        
        Args:
            teacher_outputs: Dict from get_output_distributions for teacher
            student_outputs_dict: Dict of student outputs {name: output_dict}
            student_names: List of student model names
            temperatures: List of temperatures to evaluate
            
        Returns:
            alignment_results: Comprehensive metrics dictionary
        """
        results = {
            'teacher_accuracy': np.mean(teacher_outputs['predictions'] == teacher_outputs['targets']),
            'students': {}
        }
        
        print("\n" + "="*80)
        print("PROBABILITY DISTRIBUTION ALIGNMENT ANALYSIS")
        print("="*80)
        
        # Get teacher distributions at different temperatures
        teacher_probs_dict = {}
        for temp in temperatures:
            if temp == 1.0:
                teacher_probs = teacher_outputs['distributions']
            else:
                logits = teacher_outputs['logits']
                teacher_probs = F.softmax(
                    torch.from_numpy(logits) / temp, dim=1
                ).numpy()
            teacher_probs_dict[temp] = teacher_probs
        
        # Analyze each student
        for student_name in student_names:
            if student_name not in student_outputs_dict:
                print(f"\n⚠ Warning: {student_name} not in outputs")
                continue
            
            student_outputs = student_outputs_dict[student_name]
            student_results = {}
            
            # Accuracy comparison
            student_acc = np.mean(student_outputs['predictions'] == student_outputs['targets'])
            student_results['accuracy'] = student_acc
            student_results['accuracy_diff'] = student_acc - results['teacher_accuracy']
            
            print(f"\n{'─'*80}")
            print(f"Student: {student_name}")
            print(f"{'─'*80}")
            print(f"Teacher Accuracy:  {results['teacher_accuracy']:.4f}")
            print(f"Student Accuracy:  {student_acc:.4f} (Δ {student_results['accuracy_diff']:+.4f})")
            
            # Compute divergences at each temperature
            divergence_metrics = {
                'kl_divergence': {},
                'js_divergence': {},
                'hellinger_distance': {},
                'wasserstein_distance': {},
                'cosine_similarity': {}
            }
            
            student_probs_dict = {}
            for temp in temperatures:
                if temp == 1.0:
                    student_probs = student_outputs['distributions']
                else:
                    logits = student_outputs['logits']
                    student_probs = F.softmax(
                        torch.from_numpy(logits) / temp, dim=1
                    ).numpy()
                student_probs_dict[temp] = student_probs
                
                teacher_probs = teacher_probs_dict[temp]
                
                # Compute all divergence metrics
                kl_div = self.compute_kl_divergence(teacher_probs, student_probs)
                js_div = self.compute_js_divergence(teacher_probs, student_probs)
                hellinger = self.compute_hellinger_distance(teacher_probs, student_probs)
                wasserstein = self.compute_wasserstein_distance(teacher_probs, student_probs)
                cosine_sim = self.compute_cosine_similarity(teacher_probs, student_probs)
                
                divergence_metrics['kl_divergence'][temp] = kl_div
                divergence_metrics['js_divergence'][temp] = js_div
                divergence_metrics['hellinger_distance'][temp] = hellinger
                divergence_metrics['wasserstein_distance'][temp] = wasserstein
                divergence_metrics['cosine_similarity'][temp] = cosine_sim
            
            student_results['divergences'] = divergence_metrics
            
            # Print summary at T=1.0
            print(f"\nDistribution Alignment Metrics (T=1.0):")
            print(f"  KL Divergence:           {divergence_metrics['kl_divergence'][1.0]['mean']:.6f} "
                  f"(±{divergence_metrics['kl_divergence'][1.0]['std']:.6f})")
            print(f"  JS Divergence:           {divergence_metrics['js_divergence'][1.0]['mean']:.6f} "
                  f"(±{divergence_metrics['js_divergence'][1.0]['std']:.6f})")
            print(f"  Hellinger Distance:      {divergence_metrics['hellinger_distance'][1.0]['mean']:.6f} "
                  f"(±{divergence_metrics['hellinger_distance'][1.0]['std']:.6f})")
            print(f"  Wasserstein Distance:    {divergence_metrics['wasserstein_distance'][1.0]['mean']:.6f} "
                  f"(±{divergence_metrics['wasserstein_distance'][1.0]['std']:.6f})")
            print(f"  Cosine Similarity:       {divergence_metrics['cosine_similarity'][1.0]['mean']:.6f} "
                  f"(±{divergence_metrics['cosine_similarity'][1.0]['std']:.6f})")
            
            # Analyze by correct/incorrect predictions
            correct_mask = student_outputs['predictions'] == student_outputs['targets']
            incorrect_mask = ~correct_mask
            
            kl_correct = divergence_metrics['kl_divergence'][1.0]['per_sample'][correct_mask]
            kl_incorrect = divergence_metrics['kl_divergence'][1.0]['per_sample'][incorrect_mask]
            
            student_results['kl_correct_mean'] = np.mean(kl_correct) if len(kl_correct) > 0 else 0
            student_results['kl_incorrect_mean'] = np.mean(kl_incorrect) if len(kl_incorrect) > 0 else 0
            
            print(f"\nKL Divergence by Prediction Correctness:")
            print(f"  Correct predictions:     {student_results['kl_correct_mean']:.6f}")
            print(f"  Incorrect predictions:   {student_results['kl_incorrect_mean']:.6f}")
            
            results['students'][student_name] = student_results
        
        return results
    
    def create_comparison_table(self, results, save_path='distribution_comparison.json'):
        """Create a structured comparison table of all metrics."""
        
        comparison_data = {
            'teacher_accuracy': float(results['teacher_accuracy']),
            'students_metrics': {}
        }
        
        for student_name, student_result in results['students'].items():
            comparison_data['students_metrics'][student_name] = {
                'accuracy': float(student_result['accuracy']),
                'accuracy_diff': float(student_result['accuracy_diff']),
                'kl_divergence_mean': float(student_result['divergences']['kl_divergence'][1.0]['mean']),
                'kl_divergence_std': float(student_result['divergences']['kl_divergence'][1.0]['std']),
                'js_divergence_mean': float(student_result['divergences']['js_divergence'][1.0]['mean']),
                'cosine_similarity_mean': float(student_result['divergences']['cosine_similarity'][1.0]['mean']),
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)
        
        return comparison_data
    
    def plot_divergence_comparison(self, results, student_names, save_dir='part3_plots'):
        """Create visualization plots for distribution analysis."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # 1. KL Divergence comparison
        plt.figure(figsize=(12, 6))
        kl_means = []
        kl_stds = []
        
        for student_name in student_names:
            if student_name in results['students']:
                kl_data = results['students'][student_name]['divergences']['kl_divergence'][1.0]
                kl_means.append(kl_data['mean'])
                kl_stds.append(kl_data['std'])
        
        x_pos = np.arange(len(student_names))
        plt.bar(x_pos, kl_means, yerr=kl_stds, capsize=5, alpha=0.7, color='steelblue')
        plt.xlabel('Student Model', fontsize=12)
        plt.ylabel('KL Divergence (lower is better)', fontsize=12)
        plt.title('KL Divergence from Teacher Distribution', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, student_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/kl_divergence_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/kl_divergence_comparison.png")
        plt.close()
        
        # 2. Accuracy vs KL Divergence scatter
        plt.figure(figsize=(10, 6))
        
        accuracies = []
        kl_divs = []
        labels = []
        colors = []
        
        for i, student_name in enumerate(student_names):
            if student_name in results['students']:
                accuracies.append(results['students'][student_name]['accuracy'])
                kl_divs.append(results['students'][student_name]['divergences']['kl_divergence'][1.0]['mean'])
                labels.append(student_name)
                colors.append(f'C{i}')
        
        plt.scatter(accuracies, kl_divs, s=200, alpha=0.6, c=colors)
        
        for i, label in enumerate(labels):
            plt.annotate(label, (accuracies[i], kl_divs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlabel('Classification Accuracy', fontsize=12)
        plt.ylabel('KL Divergence from Teacher', fontsize=12)
        plt.title('Accuracy vs Distribution Alignment', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/accuracy_vs_kl_divergence.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/accuracy_vs_kl_divergence.png")
        plt.close()
        
        # 3. Temperature effect on divergence
        plt.figure(figsize=(12, 6))
        temperatures = [1.0, 2.0, 4.0, 8.0]
        
        for student_name in student_names:
            if student_name in results['students']:
                kl_at_temps = [
                    results['students'][student_name]['divergences']['kl_divergence'][t]['mean']
                    for t in temperatures
                ]
                plt.plot(temperatures, kl_at_temps, marker='o', label=student_name, linewidth=2)
        
        plt.xlabel('Temperature', fontsize=12)
        plt.ylabel('Mean KL Divergence', fontsize=12)
        plt.title('Temperature Effect on Distribution Alignment', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temperature_effect.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/temperature_effect.png")
        plt.close()
        
        # 4. Multiple divergence metrics comparison
        plt.figure(figsize=(14, 6))
        
        metrics = ['kl_divergence', 'js_divergence', 'hellinger_distance', 'wasserstein_distance']
        metric_names = ['KL Div', 'JS Div', 'Hellinger', 'Wasserstein']
        x_pos = np.arange(len(student_names))
        width = 0.2
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = []
            for student_name in student_names:
                if student_name in results['students']:
                    values.append(results['students'][student_name]['divergences'][metric][1.0]['mean'])
            
            # Normalize different metrics to similar scale for comparison
            values = np.array(values)
            if metric == 'hellinger_distance' or metric == 'wasserstein_distance':
                # These are naturally smaller, scale them for visibility
                values = values * 10
            
            plt.bar(x_pos + i*width, values, width, label=metric_name, alpha=0.8)
        
        plt.xlabel('Student Model', fontsize=12)
        plt.ylabel('Divergence Value (Hellinger & Wasserstein scaled ×10)', fontsize=12)
        plt.title('Multiple Divergence Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + 1.5*width, student_names, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/multiple_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/multiple_metrics_comparison.png")
        plt.close()


if __name__ == "__main__":
    print("Part 3: Probability Distribution Alignment Analysis")
    print("This module should be imported and used with your trained models.")
    print("See example_part3_main.py for usage instructions.")