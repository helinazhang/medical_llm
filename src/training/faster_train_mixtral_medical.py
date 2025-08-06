#!/usr/bin/env python3
"""
Complete fix for tensor shape issues
Custom data collator to handle inconsistent tensor shapes
"""

import os
import torch
import multiprocessing as mp

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomDataCollator:
    """Custom data collator that properly handles tensor shapes"""
    tokenizer: Any
    pad_to_multiple_of: int = 8
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract sequences
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Find max length in batch
        max_len = max(len(seq) for seq in input_ids)
        
        # Pad to multiple
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad sequences
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for input_seq, label_seq in zip(input_ids, labels):
            # Ensure sequences are lists of integers
            if not isinstance(input_seq, list):
                input_seq = input_seq.tolist() if hasattr(input_seq, 'tolist') else list(input_seq)
            if not isinstance(label_seq, list):
                label_seq = label_seq.tolist() if hasattr(label_seq, 'tolist') else list(label_seq)
            
            # Truncate if necessary
            if len(input_seq) > max_len:
                input_seq = input_seq[:max_len]
                label_seq = label_seq[:max_len]
            
            # Calculate padding
            padding_length = max_len - len(input_seq)
            
            # Pad input_ids
            padded_input = input_seq + [self.tokenizer.pad_token_id] * padding_length
            batch_input_ids.append(padded_input)
            
            # Create attention mask
            attention_mask = [1] * len(input_seq) + [0] * padding_length
            batch_attention_mask.append(attention_mask)
            
            # Pad labels (use -100 for padding tokens)
            padded_labels = label_seq + [-100] * padding_length
            batch_labels.append(padded_labels)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

class FixedMistralTrainer:
    """Mistral trainer with tensor shape fixes"""
    
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self, use_quantization: bool = True):
        """Load model and tokenizer"""
        logger.info("Loading model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer loaded. Pad token ID: {self.tokenizer.pad_token_id}")
        
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Add Flash Attention if available
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention enabled")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Using SDPA attention")
        
        # Add quantization if requested
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            logger.info("4-bit quantization enabled")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        # Setup for training
        if use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model loaded and configured")
    
    def prepare_dataset(self, dataset_path: str, max_length: int = 2048):
        """Prepare dataset with proper tensor handling"""
        logger.info(f"📁 Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract and format texts
        texts = []
        for item in data:
            if isinstance(item, dict):
                if 'text' in item:
                    text = item['text']
                elif 'instruction' in item and 'output' in item:
                    input_text = item.get('input', '')
                    if input_text:
                        text = f"<s>[INST] {item['instruction']}\n\nInput: {input_text} [/INST] {item['output']}</s>"
                    else:
                        text = f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"
                elif 'conversations' in item:
                    text = self._format_conversations(item['conversations'])
                else:
                    continue
                texts.append(text)
        
        logger.info(f"Extracted {len(texts)} text examples")
        
        # Tokenize carefully to avoid tensor issues
        logger.info("Tokenizing dataset...")
        tokenized_data = {
            "input_ids": [],
            "labels": []
        }
        
        # Process in small batches
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            batch_encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=False,  # We handle special tokens in text
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
            )
            
            # Process each example
            for j in range(len(batch_texts)):
                input_ids = batch_encoded["input_ids"][j]
                
                # Ensure proper format
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.tolist()
                elif not isinstance(input_ids, list):
                    input_ids = list(input_ids)
                
                # Validate sequence
                if 32 <= len(input_ids) <= max_length:  # Reasonable length
                    tokenized_data["input_ids"].append(input_ids)
                    tokenized_data["labels"].append(input_ids.copy())  # Copy for labels
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} examples")
        
        logger.info(f"Tokenized {len(tokenized_data['input_ids'])} valid examples")
        
        # Create dataset
        dataset = Dataset.from_dict(tokenized_data)
        
        # Print some statistics
        if len(dataset) > 0:
            lengths = [len(ex["input_ids"]) for ex in dataset.select(range(min(100, len(dataset))))]
            logger.info(f"Sequence length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f}")
        
        return dataset
    
    def _format_conversations(self, conversations):
        """Format conversation data for Mistral"""
        formatted = "<s>"
        for turn in conversations:
            role = turn.get('role', turn.get('from', 'user'))
            content = turn.get('content', turn.get('value', ''))
            
            if role in ['user', 'human']:
                formatted += f"[INST] {content} [/INST]"
            elif role in ['assistant', 'gpt', 'bot']:
                formatted += f" {content}</s><s>"
        
        return formatted.rstrip("<s>")
    
    def train(self, dataset, **training_kwargs):
        """Train with fixed tensor handling"""
        logger.info("Starting training with tensor fixes...")
        
        # Training arguments
        training_args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "logging_steps": 25,
            "save_steps": 500,
            "save_total_limit": 2,
            
            # Safe dataloader settings
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "dataloader_drop_last": True,
            
            # Precision settings
            "bf16": torch.cuda.get_device_capability(0)[0] >= 8,
            "fp16": torch.cuda.get_device_capability(0)[0] < 8,
            "tf32": True,
            
            # Other settings
            "remove_unused_columns": False,
            "report_to": [],
            "local_rank": -1,
            "seed": 42,
        }
        
        training_args_dict.update(training_kwargs)
        training_args = TrainingArguments(**training_args_dict)
        
        # Use custom data collator
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("🎯 Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully!")
        return trainer

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-quantization", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    
    args = parser.parse_args()
    
    logger.info("🔧 Starting Mistral training with tensor fixes...")
    
    # Initialize trainer
    trainer = FixedMistralTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Load model
    trainer.load_model_and_tokenizer(use_quantization=args.use_quantization)
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(args.dataset_path, max_length=args.max_length)
    
    # Train
    trainer.train(
        dataset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size
    )
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()