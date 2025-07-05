import os
import pandas as pd
from pathlib import Path
from glob import glob
import torchio
from abc import ABC, abstractmethod

class BaseDatasetHandler(ABC):
    """Base class for dataset handlers"""
    
    @abstractmethod
    def extract_labels(self, excel_path):
        """Extract labels from Excel file - to be implemented by each dataset"""
        pass
    
    @abstractmethod
    def find_patient_files(self, dataset_path, patient_id):
        """Find MRI files for a patient - to be implemented by each dataset"""
        pass

class Dataset1Handler(BaseDatasetHandler):
    """Handler for your original dataset"""
    
    def extract_labels(self, excel_path):
        """Extract tumor type labels from pathology diagnosis text"""
        df = pd.read_excel(excel_path)
        labels_dict = {}
        
        for _, row in df.iterrows():
            patient_id = str(row['Ids'])
            diagnosis = str(row['Final pathologic diagnosis (WHO 2021)']).lower()
            
            # Extract tumor type and assign label
            if 'astrocytoma' in diagnosis:
                label = 0  # Astrocytoma
            elif 'oligodendroglioma' in diagnosis:
                label = 1  # Oligodendroglioma  
            elif 'glioblastoma' in diagnosis:
                label = 2  # Glioblastoma
            else:
                print(f"Warning: Unknown diagnosis for {patient_id}: {diagnosis}")
                continue
            
            labels_dict[patient_id] = label
            
        return labels_dict
    
    def find_patient_files(self, dataset_path, patient_id):
        """Find T1, T2, FLAIR files with original naming convention"""
        t1_files = glob(f"{dataset_path}/{patient_id}/*T1_bias.nii.gz", recursive=True)
        t2_files = glob(f"{dataset_path}/{patient_id}/*T2_bias.nii.gz", recursive=True) 
        flair_files = glob(f"{dataset_path}/{patient_id}/*FLAIR_bias.nii.gz", recursive=True)
        
        return {
            't1': t1_files[0] if t1_files else None,
            't2': t2_files[0] if t2_files else None,
            'flair': flair_files[0] if flair_files else None
        }

class Dataset2Handler(BaseDatasetHandler):
    """Handler for Imago dataset - customize as needed"""
    
    def extract_labels(self, excel_path):
        """Extract labels from second dataset Excel format"""
        df = pd.read_excel(excel_path)
        labels_dict = {}
        
        # Modify these column names based on your second dataset's Excel structure
        for _, row in df.iterrows():
            patient_id = str(row['Pseudo'])  # Adjust column name
            # Option 2: If you need to extract from diagnosis text
            if 'Tumor_type' in df.columns:
                diagnosis = str(row['Tumor_type']).lower()
                if 'astrocytoma' in diagnosis:
                    label = 0
                elif 'oligodendroglioma' in diagnosis:
                    label = 1
                elif 'gbm' in diagnosis:
                    label = 2
                else:
                    continue
            else:
                continue
                
            labels_dict[patient_id] = label
            
        return labels_dict
    
    def find_patient_files(self, dataset_path, patient_id):
        """Find files with different naming convention for dataset 2"""
        # Adjust these patterns based on your second dataset's file structure
        t1_files = glob(f"{dataset_path}/{patient_id}/ses-01/anat/*T1.nii.gz", recursive=True)
        t2_files = glob(f"{dataset_path}/{patient_id}/ses-01/anat/*T2.nii.gz", recursive=True) 
        flair_files = glob(f"{dataset_path}/{patient_id}/ses-01/anat/*FLAIR.nii.gz", recursive=True)
        
        return {
            't1': t1_files[0] if t1_files else None,
            't2': t2_files[0] if t2_files else None,
            'flair': flair_files[0] if flair_files else None
        }

class Dataset3Handler(BaseDatasetHandler):
    """Handler for Rotterdam dataset - customize as needed"""
    
    def extract_labels(self, excel_path):
        """Extract labels from third dataset Excel format"""
        df = pd.read_excel(excel_path)
        labels_dict = {}
        
        # Modify based on your third dataset's structure
        for _, row in df.iterrows():
            patient_id = str(row['Subject'])  # Adjust column name
            # Customize label extraction logic here
            label = int(row['type'])  # Adjust as needed
            labels_dict[patient_id] = label
            
        return labels_dict
    
    def find_patient_files(self, dataset_path, patient_id):
        """Find files with different naming convention for dataset 3"""
        # Adjust these patterns based on your third dataset's file structure
        t1_files = glob(f"{dataset_path}/{patient_id}/**/1_T1/NIFTI/T1_biasfield.nii.gz", recursive=True)
        t2_files = glob(f"{dataset_path}/{patient_id}/**/3_T2/NIFTI/T2_biasfield.nii.gz", recursive=True) 
        flair_files = glob(f"{dataset_path}/{patient_id}/**/4_FLAIR/NIFTI/FLAIR_biasfield.nii.gz", recursive=True)
        
        return {
            't1': t1_files[0] if t1_files else None,
            't2': t2_files[0] if t2_files else None,
            'flair': flair_files[0] if flair_files else None
        }

class Datasets:
    def __init__(self, target_size=(96, 96, 96), target_spacing=(1.0, 1.0, 1.0)):
        """
        Initialize dataset handler
        
        Args:
            target_size: Target spatial dimensions (depth, height, width)
            target_spacing: Target voxel spacing in mm
        
        Note: Preprocessing transforms are defined here but applied in training
        """
        self.datasets = []
        self.target_size = target_size
        self.target_spacing = target_spacing

        # Define preprocessing transforms (these will be applied in training, not here)
        # This is just for reference/documentation
        self.preprocessing_transforms = torchio.Compose([
            torchio.RescaleIntensity((0.05, 99.5)),
            torchio.Resample(target=target_spacing),
            torchio.CropOrPad(target_size),
        ])

        print(f"Dataset configured for target size: {target_size}")
        print(f"Target spacing: {target_spacing} mm")
        print("Note: Images will be preprocessed during training, not at loading time")

    def add_dataset(self, dataset_path, excel_path, handler_class):
        """Add a dataset with its specific handler"""
        handler = handler_class()
        labels = handler.extract_labels(excel_path)
        
        dataset_info = {
            'path': dataset_path,
            'labels': labels,
            'handler': handler
        }
        self.datasets.append(dataset_info)
        print(f"Added dataset with {len(labels)} patients using {handler_class.__name__}")

    def return_total_samples(self):
        """
        Return all subjects from all datasets as TorchIO subjects with file paths
        
        Images are not loaded or preprocessed here - that happens during training.
        This makes dataset creation much faster and more memory efficient.
        """
        all_subjects = []

        for dataset_info in self.datasets:
            dataset_path = dataset_info['path']
            labels = dataset_info['labels']
            handler = dataset_info['handler']

            print(f"Processing dataset: {dataset_path}")

            for patient_id, label in labels.items():
                # Find files using the dataset-specific handler
                files = handler.find_patient_files(dataset_path, patient_id)
                
                # Create subject with metadata and file paths
                subject_data = {
                    'label': label,
                    'dataset': dataset_path,
                    'patient_id': patient_id
                }
                
                sequences_found = 0
                # Store file paths as TorchIO ScalarImages (lazy loading)
                for seq_name, file_path in files.items():
                    if file_path and os.path.exists(file_path):
                        # Create TorchIO ScalarImage with file path (not loaded yet)
                        subject_data[seq_name] = torchio.ScalarImage(file_path)
                        sequences_found += 1

                # Only add if we found at least one sequence
                if sequences_found > 0:
                    all_subjects.append(torchio.Subject(**subject_data))
                    if len(all_subjects) % 50 == 0:
                        print(f"Processed {len(all_subjects)} subjects...")

        print(f"Total subjects across all datasets: {len(all_subjects)}")
        print(f"Images will be loaded and standardized to size: {self.target_size} during training")
        return all_subjects

    def get_dataset_summary(self):
        """Print summary of all datasets"""
        total_patients = 0
        class_counts = {0: 0, 1: 0, 2: 0}  # Count by class
        
        for i, dataset_info in enumerate(self.datasets):
            dataset_labels = list(dataset_info['labels'].values())
            dataset_class_counts = {0: 0, 1: 0, 2: 0}
            
            for label in dataset_labels:
                if label in dataset_class_counts:
                    dataset_class_counts[label] += 1
                    class_counts[label] += 1
            
            print(f"Dataset {i+1}: {len(dataset_info['labels'])} patients")
            print(f"  Astrocytoma (0): {dataset_class_counts[0]}")
            print(f"  Oligodendroglioma (1): {dataset_class_counts[1]}")
            print(f"  Glioblastoma (2): {dataset_class_counts[2]}")
            
            total_patients += len(dataset_info['labels'])
        
        print(f"\nOVERALL SUMMARY:")
        print(f"Total patients: {total_patients}")
        print(f"Class distribution:")
        print(f"  Astrocytoma (0): {class_counts[0]} ({class_counts[0]/total_patients*100:.1f}%)")
        print(f"  Oligodendroglioma (1): {class_counts[1]} ({class_counts[1]/total_patients*100:.1f}%)")
        print(f"  Glioblastoma (2): {class_counts[2]} ({class_counts[2]/total_patients*100:.1f}%)")
        print(f"Target standardization: {self.target_size} @ {self.target_spacing} mm spacing")
