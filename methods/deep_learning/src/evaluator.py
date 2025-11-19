"""
Evaluation utilities for deep learning models.
Calculates metrics and generates visualizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple
import json


class ModelEvaluator:
    """
    Evaluator class for deep learning models.
    Calculates metrics and generates visualizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        results_dir: Path
    ):
        """
        Initialize evaluator.
        
        Parameters:
            model: Trained PyTorch model
            device: Device to run evaluation on
            output_dir: Directory for output files (predictions, visualizations)
            results_dir: Directory for results (metrics)
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.results_dir = results_dir
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def _calculate_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Parameters:
            pred: Predicted mask
            target: Ground truth mask
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of metrics
        """
        # Convert to binary
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        # Calculate metrics
        intersection = (pred_binary * target_binary).sum().item()
        union = pred_binary.sum().item() + target_binary.sum().item() - intersection
        
        # IoU (Intersection over Union)
        iou = intersection / union if union > 0 else 1.0
        
        # Dice coefficient
        dice = (2.0 * intersection) / (pred_binary.sum().item() + target_binary.sum().item()) if (pred_binary.sum().item() + target_binary.sum().item()) > 0 else 1.0
        
        # Precision, Recall, F1
        tp = intersection
        fp = pred_binary.sum().item() - intersection
        fn = target_binary.sum().item() - intersection
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True,
        num_visualizations: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Parameters:
            test_loader: DataLoader for test data
            save_predictions: Whether to save prediction images
            num_visualizations: Number of samples to visualize
            
        Returns:
            Dictionary of average metrics
        """
        all_metrics = {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        predictions_dir = self.output_dir / 'predictions'
        if save_predictions:
            predictions_dir.mkdir(parents=True, exist_ok=True)
        
        visualization_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Apply sigmoid if needed
                if outputs.min() < 0 or outputs.max() > 1:
                    outputs = torch.sigmoid(outputs)
                
                # Calculate metrics for each sample in batch
                for i in range(images.shape[0]):
                    pred = outputs[i:i+1]
                    target = masks[i:i+1]
                    
                    metrics = self._calculate_metrics(pred, target)
                    for key in all_metrics:
                        all_metrics[key].append(metrics[key])
                    
                    # Save predictions
                    if save_predictions:
                        pred_np = pred[0, 0].cpu().numpy()
                        pred_image = (pred_np * 255).astype(np.uint8)
                        pred_pil = Image.fromarray(pred_image)
                        pred_pil.save(predictions_dir / f'pred_{batch_idx}_{i}.png')
                    
                    # Create visualizations
                    if visualization_count < num_visualizations:
                        self._create_visualization(
                            images[i].cpu(),
                            masks[i, 0].cpu().numpy(),
                            outputs[i, 0].cpu().numpy(),
                            visualization_count
                        )
                        visualization_count += 1
        
        # Calculate average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        # Save metrics
        metrics_path = self.results_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        return avg_metrics
    
    def _create_visualization(
        self,
        image: torch.Tensor,
        mask: np.ndarray,
        prediction: np.ndarray,
        index: int
    ):
        """
        Create visualization comparing image, ground truth, and prediction.
        
        Parameters:
            image: Original image tensor
            mask: Ground truth mask
            prediction: Predicted mask
            index: Index for saving file
        """
        # Convert image to numpy
        if image.shape[0] == 3:  # RGB
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image[0].numpy()
            image_np = (image_np * 255).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        vis_path = self.output_dir / 'visualizations' / f'vis_{index}.png'
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in formatted way.
        
        Parameters:
            metrics: Dictionary of metrics to print
        """
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"IoU (Intersection over Union): {metrics['iou']:.4f}")
        print(f"Dice Coefficient: {metrics['dice']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("=" * 60)

