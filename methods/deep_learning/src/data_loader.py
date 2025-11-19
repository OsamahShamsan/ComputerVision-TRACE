"""
Data loading utilities for deep learning models.
Handles loading images and masks from processed dataset.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings


class CopyMoveDataset(Dataset):
    """
    Dataset class for copy-move forgery detection.
    Loads image-mask pairs from processed data directory.
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize dataset.
        
        Parameters:
            images_dir: Directory containing input images
            masks_dir: Directory containing ground truth masks
            transform: Transformations to apply to images
            mask_transform: Transformations to apply to masks
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all image files and match with masks
        self.image_files = sorted(list(self.images_dir.glob('*.png')))
        self.pairs = []
        self.corrupted_files = []
        
        # Match images with corresponding masks and validate files
        for img_path in self.image_files:
            # Extract image number from filename (e.g., image_000001.png -> 000001)
            img_name = img_path.stem
            if img_name.startswith('image_'):
                mask_number = img_name.replace('image_', '')
                mask_path = self.masks_dir / f'mask_{mask_number}.png'
                
                if not mask_path.exists():
                    warnings.warn(f"Mask not found for {img_path.name}, skipping pair")
                    continue
                
                # Validate that both files can be opened (not corrupted)
                try:
                    # Try to open and verify image
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's a valid image
                    # Try to open and verify mask
                    with Image.open(mask_path) as mask:
                        mask.verify()  # Verify it's a valid image
                    
                    # If both files are valid, add to pairs
                    self.pairs.append((str(img_path), str(mask_path)))
                except (UnidentifiedImageError, IOError, OSError) as e:
                    # File is corrupted or cannot be opened
                    self.corrupted_files.append((str(img_path), str(mask_path)))
                    warnings.warn(f"Skipping corrupted file pair: {img_path.name} / {mask_path.name} - {str(e)}")
        
        if self.corrupted_files:
            print(f"\nWarning: Found {len(self.corrupted_files)} corrupted file pairs that will be skipped")
            print(f"Total valid pairs: {len(self.pairs)}")
    
    def __len__(self) -> int:
        """Return number of image-mask pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image-mask pair at given index.
        
        Parameters:
            idx: Index of the pair
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        img_path, mask_path = self.pairs[idx]
        
        # Load image and mask with error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load image and mask
                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')  # Grayscale mask
                break  # Success, exit retry loop
            except (UnidentifiedImageError, IOError, OSError) as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, raise error
                    raise RuntimeError(f"Failed to load image pair after {max_retries} attempts: {img_path} / {mask_path} - {str(e)}")
                # Wait a bit before retrying (in case file is being downloaded from iCloud)
                import time
                time.sleep(0.1)
        
        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        mask = np.array(mask, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Apply transforms if provided
        if self.transform:
            # Convert to PIL for transforms
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image_pil)
            # Convert back to tensor if transform returns tensor
            if isinstance(image, torch.Tensor):
                pass  # Already a tensor
            else:
                image = transforms.ToTensor()(image)
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if self.mask_transform:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask = self.mask_transform(mask_pil)
            if isinstance(mask, torch.Tensor):
                pass
            else:
                mask = transforms.ToTensor()(mask)
        else:
            # Convert to tensor and add channel dimension
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask


def get_data_loaders(
    config: Dict,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Parameters:
        config: Configuration dictionary with data paths and settings
        train_transform: Transformations for training data
        val_transform: Transformations for validation/test data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_paths = config['data_paths']
    batch_size = config['model_settings']['batch_size']
    
    # Default transforms if not provided
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Create datasets
    train_dataset = CopyMoveDataset(
        images_dir=data_paths['train_images'],
        masks_dir=data_paths['train_masks'],
        transform=train_transform,
        mask_transform=val_transform  # No augmentation for masks
    )
    
    val_dataset = CopyMoveDataset(
        images_dir=data_paths['val_images'],
        masks_dir=data_paths['val_masks'],
        transform=val_transform,
        mask_transform=val_transform
    )
    
    test_dataset = CopyMoveDataset(
        images_dir=data_paths['test_images'],
        masks_dir=data_paths['test_masks'],
        transform=val_transform,
        mask_transform=val_transform
    )
    
    # Create data loaders
    # Use num_workers=0 on macOS to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Darwin' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False if num_workers == 0 else True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if num_workers == 0 else True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if num_workers == 0 else True
    )
    
    return train_loader, val_loader, test_loader

