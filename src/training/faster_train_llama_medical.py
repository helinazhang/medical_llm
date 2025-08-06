#!/usr/bin/env python3
"""
Ultra-Safe LLaMA Training - Designed to prevent system crashes
Uses the most conservative settings possible
"""

import os
import json
import torch
import logging
import argparse
import time
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import Dataset

# Ultra-conservative environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:False"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA operations

# Limit memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ultra_safe_memory_cleanup():
    """Ultra-aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)  # Give system time to clean up

class UltraSafeTrainer:
    """Ultra-conservative trainer designed to never crash the system"""
    
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Ultra-conservative settings
        torch.backends.cuda.matmul.allow_tf32 = False  # More stable
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False  # More stable
        
        # Force single-threaded
        torch.set_num_threads(1)
        
        logger.info("🛡️ Ultra-Safe mode initialized")
    
    def load_model_and_tokenizer(self):
        """Load model with ultra-conservative settings"""
        logger.info("📥 Loading model with ULTRA-SAFE settings...")
        
        # Clear memory first
        ultra_safe_memory_cleanup()
        
        # Load tokenizer first
        logger.info("📖 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,  # Slower but more stable
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("✅ Tokenizer loaded")
        ultra_safe_memory_cleanup()
        
        # Load model with maximum safety
        logger.info("🤖 Loading model with 4-bit quantization...")
        
        # Ultra-conservative quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # More stable than bfloat16
            bnb_4bit_use_double_quant=False,  # Simpler
        )
        
        # Ultra-conservative model loading
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "torch_dtype": torch.float16,  # More stable
            "low_cpu_mem_usage": True,
            "max_memory": {0: "18GB", "cpu": "4GB"},  # Very conservative limits
            "offload_folder": "./offload_temp",  # Use disk for safety
        }
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            logger.info("✅ Model loaded with quantization")
            
            # Prepare for training
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=True  # Save memory
            )
            
            # Setup ultra-conservative LoRA
            self._setup_ultra_safe_lora()
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"💾 GPU memory: {memory_gb:.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.info("💡 System might have hardware issues")
            raise e
    
    def _setup_ultra_safe_lora(self):
        """Setup LoRA with ultra-conservative settings"""
        logger.info("🎯 Setting up ultra-safe LoRA...")
        
        # Very conservative LoRA settings
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Lower rank for stability
            lora_alpha=32,
            lora_dropout=0.05,  # Lower dropout
            target_modules=["q_proj", "v_proj"],  # Only essential modules
            bias="none",
            use_rslora=False,  # Simpler
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("✅ Ultra-safe LoRA configured")
    
    def prepare_dataset(self, dataset_path: str, max_length: int = 256):
        """Prepare dataset with ultra-conservative settings"""
        logger.info("📁 Loading dataset (ultra-safe)...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process with safety checks
        texts = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'text' in item:
                text = item['text'].strip()
                if text and 50 < len(text) < 1000:  # Conservative length limits
                    texts.append(text)
                
                # Limit dataset size for safety
                # if len(texts) >= 500:  # Only use first 500 examples
                #     logger.info(f"📊 Limited to {len(texts)} examples for safety")
                #     break
        
        logger.info(f"📊 Using {len(texts)} examples")
        
        if len(texts) == 0:
            raise ValueError("No valid examples found")
        
        # Ultra-conservative tokenization
        def safe_tokenize(examples):
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
                add_special_tokens=True,
            )
            
            # Simple labels (no masking for safety)
            tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
            return tokens
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            safe_tokenize,
            batched=True,
            batch_size=10,  # Very small batches
            remove_columns=dataset.column_names,
            num_proc=1,
        )
        
        # Conservative filtering
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: 20 <= len(x["input_ids"]) <= max_length
        )
        
        logger.info(f"✅ Final dataset: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, dataset, epochs: int = 1, learning_rate: float = 5e-5):
        """Train with ultra-safe settings"""
        logger.info("🛡️ Starting ULTRA-SAFE training...")
        
        # Ultra-conservative training settings
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=16,  # Still get effective batch of 16
            learning_rate=learning_rate,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            fp16=True,  # Use fp16 instead of bf16
            tf32=False,
            dataloader_num_workers=0,  # No multiprocessing
            dataloader_pin_memory=False,
            gradient_checkpointing=True,  # Save memory
            optim="adamw_torch",  # Updated optimizer name
            max_grad_norm=0.5,  # Lower gradient clipping
            seed=42,
            eval_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_drop_last=True,
            prediction_loss_only=True,
            save_safetensors=False,  # Use legacy format for stability
        )
        
        # Ultra-simple data collator
        def ultra_safe_collate(features):
            # Very simple padding
            max_len = max(len(f["input_ids"]) for f in features)
            
            batch = {
                "input_ids": [],
                "labels": [],
                "attention_mask": []
            }
            
            for f in features:
                ids = f["input_ids"]
                labels = f["labels"]
                
                # Pad to max length
                padding = [self.tokenizer.pad_token_id] * (max_len - len(ids))
                label_padding = [-100] * (max_len - len(ids))
                
                batch["input_ids"].append(ids + padding)
                batch["labels"].append(labels + label_padding)
                batch["attention_mask"].append([1] * len(ids) + [0] * len(padding))
            
            return {
                "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
                "labels": torch.tensor(batch["labels"], dtype=torch.long),
                "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            }
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=ultra_safe_collate,
        )
        
        logger.info("🎯 Effective batch size: 16")
        logger.info("⚠️ Using ultra-conservative settings to prevent crashes")
        
        # Memory check before training
        ultra_safe_memory_cleanup()
        
        try:
            # Train with safety measures
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info("✅ ULTRA-SAFE training completed!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.info("💡 This might indicate hardware issues")
            raise e

def main():
    parser = argparse.ArgumentParser(description="Ultra-Safe LLaMA Training")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    logger.info("🛡️ ULTRA-SAFE LLaMA Training Starting...")
    logger.info("⚠️ Using maximum safety settings to prevent system crashes")
    
    try:
        # Initialize ultra-safe trainer
        trainer = UltraSafeTrainer(
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        # Load model (most crash-prone step)
        trainer.load_model_and_tokenizer()
        
        # Prepare limited dataset
        dataset = trainer.prepare_dataset(args.dataset_path, max_length=args.max_length)
        
        # Train with ultra-safe settings
        trainer.train(dataset, epochs=args.epochs, learning_rate=args.learning_rate)
        
        print("\n✅ ULTRA-SAFE TRAINING COMPLETED!")
        print(f"📁 Model saved to: {args.output_dir}")
        print("🛡️ No system crashes!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 If this still crashes, there might be:")
        print("   - Hardware issues (GPU, RAM, PSU)")
        print("   - Driver problems")
        print("   - Thermal issues")
        print("   - Power supply problems")

if __name__ == "__main__":
    main()