#!/usr/bin/env python3
"""
Medical LLM Model Manager
Handles multiple models, configurations, and deployments
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

class ModelManager:
    """Manages multiple medical LLM models and configurations"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "created": datetime.now().isoformat()}
    
    def _save_registry(self):
        """Save model registry"""
        self.registry["updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, model_path: str, base_model: str, 
                      performance_metrics: Optional[Dict] = None, 
                      notes: Optional[str] = None):
        """Register a trained model"""
        
        model_info = {
            "model_path": str(model_path),
            "base_model": base_model,
            "created": datetime.now().isoformat(),
            "performance_metrics": performance_metrics or {},
            "notes": notes or "",
            "status": "active"
        }
        
        self.registry["models"][model_name] = model_info
        self._save_registry()
        print(f"✅ Registered model: {model_name}")
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        return [
            {"name": name, **info} 
            for name, info in self.registry["models"].items()
            if info.get("status") != "deleted"
        ]
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        return self.registry["models"].get(model_name)
    
    def get_best_model(self, metric: str = "eval_loss") -> Optional[Tuple[str, Dict]]:
        """Get the best performing model based on a metric"""
        best_model = None
        best_score = float('inf') if 'loss' in metric else 0
        
        for name, info in self.registry["models"].items():
            if info.get("status") != "active":
                continue
                
            metrics = info.get("performance_metrics", {})
            if metric in metrics:
                score = metrics[metric]
                if ('loss' in metric and score < best_score) or \
                   ('loss' not in metric and score > best_score):
                    best_score = score
                    best_model = (name, info)
        
        return best_model
    
    def compare_models(self, model_names: List[str]) -> Dict:
        """Compare multiple models"""
        comparison = {}
        
        for name in model_names:
            if name in self.registry["models"]:
                info = self.registry["models"][name]
                comparison[name] = {
                    "base_model": info.get("base_model"),
                    "performance": info.get("performance_metrics", {}),
                    "created": info.get("created"),
                    "path": info.get("model_path")
                }
        
        return comparison
    
    def create_deployment_config(self, model_name: str, config_type: str = "demo") -> str:
        """Create deployment configuration for a model"""
        
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        configs = {
            "demo": {
                "model_path": model_info["model_path"],
                "base_model": model_info["base_model"],
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "interface": "streamlit"
            },
            "api": {
                "model_path": model_info["model_path"],
                "base_model": model_info["base_model"],
                "max_length": 1024,
                "temperature": 0.5,
                "top_p": 0.95,
                "batch_size": 1,
                "timeout": 30,
                "interface": "fastapi"
            },
            "production": {
                "model_path": model_info["model_path"],
                "base_model": model_info["base_model"],
                "max_length": 2048,
                "temperature": 0.3,
                "top_p": 0.9,
                "load_in_8bit": True,
                "safety_filters": True,
                "logging": True,
                "interface": "grpc"
            }
        }
        
        config = configs.get(config_type, configs["demo"])
        
        # Save config file
        config_path = self.models_dir / f"{model_name}_{config_type}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)

def create_training_configs():
    """Create optimized training configurations for different scenarios"""
    
    configs = {
        "quick_test": {
            "description": "Quick test configuration for rapid iteration",
            "dataset_size": "small",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-4,
            "max_length": 256
        },
        "development": {
            "description": "Development configuration for model experimentation",
            "dataset_size": "medium",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 3e-4,
            "max_length": 512
        },
        "production": {
            "description": "Production configuration for final model training",
            "dataset_size": "large",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "max_length": 1024,
            "early_stopping": True,
            "validation_split": 0.15
        }
    }
    
    # Save configs
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    for name, config in configs.items():
        config_path = configs_dir / f"{name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created config: {config_path}")

def main():
    """CLI interface for model management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical LLM Model Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models
    subparsers.add_parser("list", help="List all registered models")
    
    # Register model
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("--name", required=True, help="Model name")
    register_parser.add_argument("--path", required=True, help="Model path")
    register_parser.add_argument("--base-model", required=True, help="Base model used")
    register_parser.add_argument("--notes", help="Additional notes")
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("--models", nargs="+", required=True, help="Model names to compare")
    
    # Get best model
    best_parser = subparsers.add_parser("best", help="Get best performing model")
    best_parser.add_argument("--metric", default="eval_loss", help="Metric to use for comparison")
    
    # Create deployment config
    deploy_parser = subparsers.add_parser("deploy-config", help="Create deployment configuration")
    deploy_parser.add_argument("--model", required=True, help="Model name")
    deploy_parser.add_argument("--type", choices=["demo", "api", "production"], default="demo", help="Deployment type")
    
    # Create training configs
    subparsers.add_parser("create-configs", help="Create training configurations")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ModelManager()
    
    if args.command == "list":
        models = manager.list_models()
        if not models:
            print("No models registered yet.")
        else:
            print("📋 Registered Models:")
            print("-" * 50)
            for model in models:
                print(f"Name: {model['name']}")
                print(f"Base Model: {model['base_model']}")
                print(f"Path: {model['model_path']}")
                print(f"Created: {model['created']}")
                if model.get('performance_metrics'):
                    print(f"Performance: {model['performance_metrics']}")
                print("-" * 50)
    
    elif args.command == "register":
        manager.register_model(
            model_name=args.name,
            model_path=args.path,
            base_model=args.base_model,
            notes=args.notes
        )
    
    elif args.command == "compare":
        comparison = manager.compare_models(args.models)
        print("📊 Model Comparison:")
        print("-" * 50)
        for name, info in comparison.items():
            print(f"\n{name}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    elif args.command == "best":
        best = manager.get_best_model(args.metric)
        if best:
            name, info = best
            print(f"🏆 Best model by {args.metric}: {name}")
            print(f"Score: {info['performance_metrics'].get(args.metric, 'N/A')}")
            print(f"Path: {info['model_path']}")
        else:
            print("No models found with the specified metric.")
    
    elif args.command == "deploy-config":
        try:
            config_path = manager.create_deployment_config(args.model, args.type)
            print(f"✅ Created deployment config: {config_path}")
        except ValueError as e:
            print(f"❌ Error: {e}")
    
    elif args.command == "create-configs":
        create_training_configs()

if __name__ == "__main__":
    main()