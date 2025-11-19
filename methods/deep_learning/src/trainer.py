"""
Training utilities for deep learning models.
Handles training loop, validation, and model saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


class ModelTrainer:
    """
    Trainer class for deep learning models.
    Handles training, validation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
        model_name: str = 'model'
    ):
        """
        Initialize trainer.
        
        Parameters:
            model: PyTorch model to train
            config: Configuration dictionary
            device: Device to run training on (cuda or cpu)
            model_name: Name identifier for this model
        """
        self.model = model
        self.config = config
        self.device = device
        self.model_name = model_name
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer and loss
        model_settings = config['model_settings']
        # Determine which model config to use based on model_name
        if 'unet' in model_name.lower():
            model_config = config.get('model1_unet', {})
        elif 'resnet' in model_name.lower():
            model_config = config.get('model2_resnet', {})
        else:
            model_config = {}
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_settings['learning_rate']
        )
        
        # Loss function based on config
        loss_type = model_config.get('loss', 'bce_with_logits')
        if loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'bce':
            self.criterion = nn.BCELoss()
        elif loss_type == 'dice':
            self.criterion = self._dice_loss
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.patience = model_config.get('patience', 10)
        self.patience_counter = 0
        
        # Output directories - resolve relative to method_root if path is relative
        output_path = config['output_paths']['models']
        if Path(output_path).is_absolute():
            self.output_dir = Path(output_path)
        else:
            # Try to resolve relative to method_root (if available in config)
            # Otherwise use current working directory
            if 'method_root' in config:
                self.output_dir = Path(config['method_root']) / output_path
            else:
                # Fallback: use absolute path from current working directory
                self.output_dir = Path(output_path).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss for binary segmentation.
        
        Parameters:
            pred: Predicted mask
            target: Ground truth mask
            
        Returns:
            Dice loss value
        """
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Intersection over Union (IoU) metric.
        
        Parameters:
            pred: Predicted mask
            target: Ground truth mask
            threshold: Threshold for binary classification
            
        Returns:
            IoU value
        """
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0
        
        iou = intersection / union
        return iou.item()
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Parameters:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of (average_loss, average_iou)
        """
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Apply sigmoid if not already applied
                if outputs.min() < 0 or outputs.max() > 1:
                    outputs_sigmoid = torch.sigmoid(outputs)
                else:
                    outputs_sigmoid = outputs
                
                batch_iou = self._calculate_iou(outputs_sigmoid, masks)
                total_iou += batch_iou
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        return avg_loss, avg_iou
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model on validation set.
        
        Parameters:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average_loss, average_iou)
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Apply sigmoid if not already applied
                if outputs.min() < 0 or outputs.max() > 1:
                    outputs_sigmoid = torch.sigmoid(outputs)
                else:
                    outputs_sigmoid = outputs
                
                # Calculate metrics
                batch_iou = self._calculate_iou(outputs_sigmoid, masks)
                total_iou += batch_iou
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        return avg_loss, avg_iou
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None, resume_from_latest: bool = True) -> Optional[int]:
        """
        Load model checkpoint to resume training.
        
        Parameters:
            checkpoint_path: Path to checkpoint file. If None, loads latest checkpoint.
            resume_from_latest: If True and checkpoint_path is None, loads latest checkpoint.
            
        Returns:
            Epoch number to resume from, or None if no checkpoint found
        """
        if checkpoint_path is None:
            if resume_from_latest:
                checkpoint_path = self.output_dir / f'{self.model_name}_latest.pth'
            else:
                checkpoint_path = self.output_dir / f'{self.model_name}_best.pth'
        
        if not checkpoint_path.exists():
            return None
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': []
        })
        
        # Load best metrics
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation loss so far: {self.best_val_loss:.4f}")
        print(f"Best validation IoU so far: {self.best_val_iou:.4f}")
        
        return start_epoch
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Parameters:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f'{self.model_name}_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch} with val_loss={self.best_val_loss:.4f}, val_iou={self.best_val_iou:.4f}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        resume_from_checkpoint: bool = True
    ):
        """
        Main training loop.
        
        Parameters:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            resume_from_checkpoint: If True, automatically resume from latest checkpoint
        """
        # Try to resume from checkpoint
        start_epoch = 1
        if resume_from_checkpoint:
            loaded_epoch = self.load_checkpoint(resume_from_latest=True)
            if loaded_epoch is not None:
                start_epoch = loaded_epoch + 1  # Start from next epoch
        
        print(f"Starting training for {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Starting from epoch: {start_epoch}")
        print("-" * 60)
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Train
            train_loss, train_iou = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_iou = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            
            # Check for improvement
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_iou = val_iou
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save training history
        history_path = self.output_dir / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed. Best val_loss: {self.best_val_loss:.4f}, Best val_iou: {self.best_val_iou:.4f}")

