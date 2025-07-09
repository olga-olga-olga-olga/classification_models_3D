import torch
import torchio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Tuple, Optional


class MedicalImageDataset(Dataset):
    """TorchIO-based dataset for MRI classification"""
    def __init__(self, subjects: List[torchio.Subject], transform=None, num_channels=3):
        self.subjects = subjects
        self.transform = transform
        self.num_channels = num_channels
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        if self.transform:
            subject = self.transform(subject)
        # Stack all available channels
        channel_keys = ['t1', 't2', 'flair'][:self.num_channels]
        image_tensor = torch.cat([subject[k].data for k in channel_keys], dim=0)
        image_tensor = image_tensor.permute(1, 2, 3, 0)
        label_tensor = torch.tensor(subject['label'], dtype=torch.long)
        return image_tensor, label_tensor


class MedicalDataLoader:
    """High-level data loader for multi-dataset training"""
    def __init__(self, 
                 datasets_instance,
                 target_size: Tuple[int, int, int] = (240, 240, 189),
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 num_channels: int = 3,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.num_channels = num_channels
        
        # Get all subjects and create splits
        print("Loading subjects...")
        all_subjects = datasets_instance.return_total_samples()
        self._create_splits(all_subjects, test_size, val_size, random_state)
        self._calculate_class_weights()
        
        print(f"Train: {len(self.train_subjects)}, Val: {len(self.val_subjects)}, Test: {len(self.test_subjects)}")
    
    def _create_splits(self, all_subjects, test_size, val_size, random_state):
        """Create stratified train/val/test splits"""
        labels = [subject['label'] for subject in all_subjects]
        
        # Split off test set
        train_val_subjects, test_subjects, train_val_labels, _ = train_test_split(
            all_subjects, labels, test_size=test_size, stratify=labels, random_state=random_state
        )
        
        # Split train/val
        if val_size > 0:
            train_subjects, val_subjects, _, _ = train_test_split(
                train_val_subjects, train_val_labels,
                test_size=val_size/(1-test_size),
                stratify=train_val_labels,
                random_state=random_state
            )
        else:
            train_subjects, val_subjects = train_val_subjects, []
        
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects  
        self.test_subjects = test_subjects
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        train_labels = [subject['label'] for subject in self.train_subjects]
        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        self.class_weights = torch.FloatTensor(class_weights)
        print(f"Class weights: {dict(zip(classes, class_weights))}")
    
    def get_transforms(self, augment: bool = False):
        """Get preprocessing transforms"""
        transforms = [
            torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
            torchio.Resample(target=self.target_spacing),
            torchio.CropOrPad(target_shape=self.target_size),
        ]
        
        if augment:
            transforms.extend([
                torchio.RandomFlip(axes=(0, 1, 2), p=0.5),
                torchio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10), p=0.5),
                torchio.RandomNoise(std=(0, 0.05), p=0.3),
            ])
        
        return torchio.Compose(transforms)
    
    def get_dataloader(self, split='train', batch_size=4, num_workers=0):
        if split == 'train':
            subjects = self.train_subjects
            transform = self.get_transforms(augment=True)
            shuffle = True
        elif split == 'val':
            subjects = self.val_subjects
            transform = self.get_transforms(augment=False)
            shuffle = False
        elif split == 'test':
            subjects = self.test_subjects
            transform = self.get_transforms(augment=False)
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}")
        dataset = MedicalImageDataset(subjects, transform, num_channels=self.num_channels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, pin_memory=False)


def create_batch_generators(
    datasets_instance, 
    batch_size_train=12, 
    batch_size_valid=12, 
    target_size=(249, 240, 189), 
    target_spacing=(1.0, 1.0, 1.0), 
    num_channels=3,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    **loader_kwargs
):
    """
    Create batch generators that mimic the original gen_random_volume interface
    All shape variables are now configurable.
    """
    loader_config = {
        'target_size': target_size,
        'target_spacing': target_spacing,
        'num_channels': num_channels,
        'test_size': test_size,
        'val_size': val_size,
        'random_state': random_state
    }
    loader_config.update(loader_kwargs)
    data_loader = MedicalDataLoader(datasets_instance, **loader_config)
    train_loader = data_loader.get_dataloader('train', batch_size=batch_size_train, num_workers=0)
    val_loader = data_loader.get_dataloader('val', batch_size=batch_size_valid, num_workers=0)
    def pytorch_to_keras_generator(dataloader):
        while True:
            for images, labels in dataloader:
                images_np = images.numpy().astype(np.float32)
                labels_np = labels.numpy().astype(np.int32)
                num_classes = len(data_loader.class_weights)
                labels_onehot = np.zeros((len(labels_np), num_classes), dtype=np.float32)
                for i, label in enumerate(labels_np):
                    if 0 <= label < num_classes:
                        labels_onehot[i, label] = 1.0
                yield images_np, labels_onehot
    gen_train = pytorch_to_keras_generator(train_loader)
    gen_valid = pytorch_to_keras_generator(val_loader)
    return gen_train, gen_valid, data_loader.class_weights


def get_preprocess_input_dummy():
    """
    Dummy preprocessing function to maintain compatibility with original code
    Since we handle preprocessing in TorchIO transforms, this just returns the input
    """
    def preprocess_input(x):
        return x
    return preprocess_input