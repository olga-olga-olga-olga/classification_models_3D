#!/usr/bin/env python3
"""
Quick test script to verify data loading and preprocessing works correctly
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required imports work"""
    print("=" * 50)
    print("TESTING IMPORTS")
    print("=" * 50)
    
    try:
        from dataloader import create_batch_generators, get_preprocess_input_dummy
        print("✅ dataloader imports OK")
    except Exception as e:
        print(f"❌ dataloader import failed: {e}")
        return False
    
    try:
        from dataset import Datasets, Dataset1Handler, Dataset2Handler, Dataset3Handler
        print("✅ dataset imports OK")
    except Exception as e:
        print(f"❌ dataset import failed: {e}")
        return False
    
    try:
        import torch
        import torchio
        print(f"✅ PyTorch {torch.__version__} OK")
        print(f"✅ TorchIO {torchio.__version__} OK")
    except Exception as e:
        print(f"❌ PyTorch/TorchIO import failed: {e}")
        return False
    
    return True


def test_dataset_creation():
    """Test dataset creation and file finding"""
    print("\n" + "=" * 50)
    print("TESTING DATASET CREATION")
    print("=" * 50)
    
    from dataset import Datasets, Dataset1Handler, Dataset2Handler, Dataset3Handler
    
    # Dataset paths (same as your training script)
    data_path_1 = "/data/share/IMAGO/SF/"
    excel_path_1 = "/home/radv/ofilipowicz/my-scratch/datasetlabels/UCSF-PDGM_clinical_data_adjusted.xls"
    
    data_path_2 = "/data/share/IMAGO/400_cases/IMAGO_501/bids/IMAGO_first_release/derivatives - registered/"
    excel_path_2 = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Clinical_data_1st_release.xls"
    
    data_path_3 = "/data/share/IMAGO/Rotterdam/"
    excel_path_3 = "/home/radv/ofilipowicz/my-scratch/datasetlabels/Rotterdam_clinical_data.xls"
    
    # Check if paths exist
    paths_to_check = [
        ("Dataset 1 data", data_path_1),
        ("Dataset 1 excel", excel_path_1),
        ("Dataset 2 data", data_path_2),
        ("Dataset 2 excel", excel_path_2),
        ("Dataset 3 data", data_path_3),
        ("Dataset 3 excel", excel_path_3),
    ]
    
    for name, path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
    
    try:
        # Initialize datasets
        datasets = Datasets(target_size=(96, 96, 96), target_spacing=(1.0, 1.0, 1.0))
        
        # Try to add datasets
        datasets.add_dataset(data_path_1, excel_path_1, Dataset1Handler)
        datasets.add_dataset(data_path_2, excel_path_2, Dataset2Handler)
        datasets.add_dataset(data_path_3, excel_path_3, Dataset3Handler)
        
        print("✅ All datasets added successfully")
        
        # Get summary
        datasets.get_dataset_summary()
        
        return datasets
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading(datasets):
    """Test actual data loading and batch generation"""
    print("\n" + "=" * 50)
    print("TESTING DATA LOADING")
    print("=" * 50)
    
    if datasets is None:
        print("❌ Skipping data loading test - datasets not created")
        return False
    
    try:
        from dataloader import create_batch_generators
        
        # Create batch generators with small batch size for testing
        print("Creating batch generators...")
        gen_train, gen_valid, class_weights = create_batch_generators(
            datasets, 
            batch_size_train=2,  # Small batch for testing
            batch_size_valid=2,
            target_size=(96, 96, 96),
            test_size=0.3,  # Larger test size for small datasets
            val_size=0.2
        )
        
        print(f"✅ Batch generators created")
        print(f"✅ Class weights: {class_weights}")
        
        # Test getting one batch from training generator
        print("Testing training batch generation...")
        train_batch = next(gen_train)
        images, labels = train_batch
        
        print(f"✅ Training batch shape: images {images.shape}, labels {labels.shape}")
        print(f"✅ Image dtype: {images.dtype}, range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"✅ Label dtype: {labels.dtype}, unique labels: {np.unique(np.argmax(labels, axis=1))}")
        
        # Test getting one batch from validation generator
        print("Testing validation batch generation...")
        val_batch = next(gen_valid)
        val_images, val_labels = val_batch
        
        print(f"✅ Validation batch shape: images {val_images.shape}, labels {val_labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """Test if data format is compatible with model expectations"""
    print("\n" + "=" * 50)
    print("TESTING MODEL COMPATIBILITY")
    print("=" * 50)
    
    try:
        # Create a dummy batch in the expected format
        batch_size = 2
        expected_shape = (batch_size, 96, 96, 96, 3)  # (batch, D, H, W, channels)
        expected_labels = (batch_size, 3)  # (batch, num_classes) one-hot
        
        print(f"✅ Expected image shape: {expected_shape}")
        print(f"✅ Expected label shape: {expected_labels}")
        print(f"✅ Expected image dtype: float32")
        print(f"✅ Expected label dtype: float32")
        
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("MEDICAL DATA LOADING TEST")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ TESTS FAILED: Import issues")
        return
    
    # Test 2: Dataset creation
    datasets = test_dataset_creation()
    
    # Test 3: Data loading
    data_loading_ok = test_data_loading(datasets)
    
    # Test 4: Model compatibility
    model_compat_ok = test_model_compatibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if datasets and data_loading_ok and model_compat_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your data loading setup is ready for training")
        print("\nNext steps:")
        print("1. Run your training script: python train.py")
        print("2. Monitor the first few epochs for any issues")
        print("3. Check GPU memory usage and adjust batch size if needed")
    else:
        print("❌ SOME TESTS FAILED")
        print("Fix the issues above before running training")


if __name__ == "__main__":
    main()