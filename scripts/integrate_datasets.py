#!/usr/bin/env python3
"""
Dataset Integration Script
Easy integration of real medical datasets with training pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path


current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from data.real_dataset_loader import RealMedicalDataLoader
except ImportError:
    try:
        # Fallback if running from different location
        sys.path.insert(0, str(current_dir.parent))
        from src.data.real_dataset_loader import RealMedicalDataLoader
    except ImportError:
        # Last resort - direct path
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "real_dataset_loader", 
                current_dir.parent / "src" / "data" / "real_dataset_loader.py"
            )
            real_dataset_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(real_dataset_module)
            RealMedicalDataLoader = real_dataset_module.RealMedicalDataLoader
        except Exception as e:
            print("Error: Cannot find real_dataset_loader.py")
            print("Make sure you've run setup_project.py first!")
            print("Or check that real_dataset_loader.py is in src/data/")
            print(f"Debug info: {e}")
            sys.exit(1)

def create_training_ready_datasets():
    """Create training-ready datasets for different use cases"""
    
    loader = RealMedicalDataLoader()
    
    # Define dataset combinations for different purposes
    dataset_configs = {
        "small_mixed": {
            "datasets": ["pubmedqa", "medmcqa"], 
            "sample_size": 100,
            "description": "Small mixed dataset for quick testing",
            "total_size": 200
        },
        "medium_comprehensive": {
            "datasets": ["pubmedqa", "medmcqa", "healthfact"],
            "sample_size": 500,
            "description": "Medium comprehensive dataset for development",
            "total_size": 1500
        },
        "large_professional": {
            "datasets": ["pubmedqa", "medqa", "medmcqa", "medical_meadow"],
            "sample_size": 1000,
            "description": "Large professional dataset for production",
            "total_size": 4000
        },
        "exam_focused": {
            "datasets": ["medqa", "medmcqa"],
            "sample_size": 800,
            "description": "Medical exam focused dataset",
            "total_size": 1600
        },
        "qa_focused": {
            "datasets": ["pubmedqa", "medical_meadow"],
            "sample_size": 1000,
            "description": "Q&A focused dataset",
            "total_size": 2000
        }
    }
    
    # Create output directory
    output_dir = Path("data/real_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏥 Creating Real Medical Training Datasets")
    print("=" * 50)
    
    for config_name, config in dataset_configs.items():
        print(f"\n📚 Creating {config_name} dataset...")
        print(f"   Description: {config['description']}")
        print(f"   Expected size: ~{config['total_size']} examples")
        
        try:
            # Load data
            data = loader.load_custom_dataset(
                dataset_names=config["datasets"],
                sample_size_per_dataset=config["sample_size"]
            )
            
            if data:
                # Save in training format
                output_file = output_dir / f"{config_name}.json"
                loader.save_dataset(data, str(output_file), format_type="qa")
                
                print(f"   ✅ Created: {output_file} ({len(data)} examples)")
                
                # Create metadata file
                metadata = {
                    "name": config_name,
                    "description": config["description"],
                    "datasets_used": config["datasets"],
                    "sample_size_per_dataset": config["sample_size"],
                    "actual_size": len(data),
                    "format": "qa"
                }
                
                metadata_file = output_dir / f"{config_name}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                print(f"Failed to create {config_name}")
                
        except Exception as e:
            print(f"Error creating {config_name}: {e}")
    
    print(f"\nAll datasets saved to: {output_dir}")
    print("\nReady to train! Use these commands:")
    
    for config_name in dataset_configs.keys():
        print(f"\npython advanced_train.py \\")
        print(f"  --base-model 'gpt2-medium' \\")
        print(f"  --dataset 'data/real_datasets/{config_name}.json' \\")
        print(f"  --output 'models/medical-{config_name}' \\")
        print(f"  --epochs 3")

def train_with_real_data():
    """Complete training pipeline with real data"""
    
    # Step 1: Create datasets
    print("Step 1: Creating datasets...")
    create_training_ready_datasets()
    
    # Step 2: Show training commands
    print("\n" + "=" * 60)
    print("Step 2: Train models with real data")
    print("=" * 60)
    
    training_commands = [
        {
            "name": "Quick Test Model",
            "dataset": "small_mixed",
            "model": "gpt2",
            "epochs": 1,
            "description": "Fast training for testing"
        },
        {
            "name": "Development Model", 
            "dataset": "medium_comprehensive",
            "model": "gpt2-medium",
            "epochs": 3,
            "description": "Good for development and experimentation"
        },
        {
            "name": "Production Model",
            "dataset": "large_professional", 
            "model": "gpt2-medium",
            "epochs": 5,
            "description": "High-quality model for deployment"
        }
    ]
    
    for cmd in training_commands:
        print(f"\n {cmd['name']} ({cmd['description']}):")
        print(f"python advanced_train.py \\")
        print(f"  --base-model '{cmd['model']}' \\")
        print(f"  --dataset 'data/real_datasets/{cmd['dataset']}.json' \\")
        print(f"  --output 'models/medical-{cmd['dataset']}' \\")
        print(f"  --epochs {cmd['epochs']}")
    
    print(f"\n After training, evaluate with:")
    print(f"python advanced_evaluate.py --model-path models/medical-medium_comprehensive")
    
    print(f"\n Then demo with:")
    print(f"streamlit run universal_demo.py")



def main():
    parser = argparse.ArgumentParser(description="Dataset Integration for Medical AI")
    parser.add_argument("--create-datasets", action="store_true", 
                       help="Create training-ready real datasets")
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Show complete training pipeline with real data") 


    parser.add_argument("--list-real", action="store_true",
                       help="List available real datasets")
    
    args = parser.parse_args()
    
    if args.list_real:
        loader = RealMedicalDataLoader()
        loader.list_available_datasets()
    
    elif args.create_datasets:
        create_training_ready_datasets()
    
    elif args.full_pipeline:
        train_with_real_data()
    
    else:
        parser.print_help()
        print(f"\n🚀 Quick start with real data:")
        print(f"python integrate_datasets.py --create-datasets")
        print(f"python advanced_train.py --dataset data/real_datasets/medium_comprehensive.json --base-model gpt2-medium --output models/real-medical-ai")

if __name__ == "__main__":
    main()