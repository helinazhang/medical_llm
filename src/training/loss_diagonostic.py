#!/usr/bin/env python3
"""
Quick diagnostic for high training loss
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def diagnose_training_loss(dataset_path, model_name="meta-llama/Llama-2-7b-hf"):
    """Diagnose why training loss is so high"""
    
    print("🔍 Diagnosing high training loss...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load sample data
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Process one example
    sample_item = data[0]
    if 'Instructions' in sample_item and 'Responses' in sample_item:
        inst = sample_item['Instructions'][0]
        resp = sample_item['Responses'][0]
        text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}\n"
    else:
        text = str(sample_item)
    
    print(f"Sample text:\n{text[:300]}...")
    
    # Tokenize
    tokens = tokenizer.encode(text, return_tensors="pt")
    print(f"Token count: {tokens.shape[1]}")
    
    # Test model loss
    model.eval()
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
        
    print(f"Model loss on sample: {loss.item():.4f}")
    
    # Check label masking
    response_start = "### Response:"
    prefix = text.split(response_start)[0] + response_start
    prefix_tokens = tokenizer.encode(prefix)
    
    print(f"Instruction length: {len(prefix_tokens)} tokens")
    print(f"Total length: {tokens.shape[1]} tokens")
    print(f"Response length: {tokens.shape[1] - len(prefix_tokens)} tokens")
    
    if tokens.shape[1] - len(prefix_tokens) < 5:
        print("PROBLEM: Response too short after instruction!")
    
    # Expected loss range
    vocab_size = tokenizer.vocab_size
    random_loss = torch.log(torch.tensor(float(vocab_size)))
    print(f"Random baseline loss: {random_loss.item():.4f}")
    print(f"Good fine-tuned loss: 2.0-4.0")
    print(f"Your loss: {loss.item():.4f}")
    
    if loss.item() > 8:
        print("CRITICAL: Loss way too high!")
        print("Likely causes:")
        print("   - Wrong tokenization")
        print("   - Bad label masking")
        print("   - Data format issues")
        print("   - Learning rate too low")
    elif loss.item() > 6:
        print("WARNING: Loss higher than expected")
    else:
        print("Loss in reasonable range")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loss_diagnostic.py <dataset_path>")
        sys.exit(1)
    
    diagnose_training_loss(sys.argv[1])