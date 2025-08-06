#!/usr/bin/env python3
"""
Flash Attention Verification Script
Run this to confirm Flash Attention is properly installed and working
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_flash_attention():
    """Test Flash Attention performance"""
    print("Testing Flash Attention...")
    
    # Check installation
    try:
        import flash_attn
        print(f"Flash Attention v{flash_attn.__version__} installed")
    except ImportError:
        print("Flash Attention not installed")
        print("Install with: pip install flash-attn --no-build-isolation")
        return False
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # Test with Flash Attention
    print("\n🚀 Testing with Flash Attention...")
    try:
        model_flash = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Test input
        text = "Explain quantum computing in simple terms. " * 50  # Long text
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            _ = model_flash(**inputs)
        
        # Time Flash Attention
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                outputs = model_flash(**inputs)
        
        torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / 10
        
        print(f"Flash Attention working! Average time: {flash_time:.3f}s")
        
        # Memory usage
        flash_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Memory used: {flash_memory:.2f} GB")
        
        del model_flash
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Flash Attention test failed: {e}")
        return False

def benchmark_attention_types():
    """Benchmark different attention implementations"""
    print("\n🔬 Benchmarking attention implementations...")
    
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test input
    text = "Write a detailed explanation of machine learning. " * 100
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    print(f"Input sequence length: {inputs['input_ids'].shape[1]}")
    
    implementations = []
    
    # Test Flash Attention
    try:
        import flash_attn
        implementations.append(("flash_attention_2", "🚀 Flash Attention 2"))
    except ImportError:
        pass
    
    # Test SDPA (PyTorch optimized)
    implementations.append(("sdpa", "📊 SDPA (PyTorch)"))
    
    # Test eager (default)
    implementations.append(("eager", "🐌 Eager (Default)"))
    
    results = {}
    
    for impl_name, display_name in implementations:
        print(f"\nTesting {display_name}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=impl_name
            )
            
            # Warmup
            with torch.no_grad():
                _ = model(**inputs)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(5):
                    outputs = model(**inputs)
            
            torch.cuda.synchronize()
            avg_time = (time.time() - start_time) / 5
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            
            results[impl_name] = {
                'time': avg_time,
                'memory': memory_used,
                'display_name': display_name
            }
            
            print(f"   Time: {avg_time:.3f}s")
            print(f"   Memory: {memory_used:.2f} GB")
            
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        except Exception as e:
            print(f"Failed: {e}")
    
    # Show comparison
    if len(results) > 1:
        print("\n📈 Performance Comparison:")
        baseline = None
        for impl_name, result in results.items():
            if baseline is None:
                baseline = result['time']
            
            speedup = baseline / result['time']
            print(f"   {result['display_name']}: {speedup:.2f}x speed")
    
    return results

if __name__ == "__main__":
    print("Flash Attention Verification and Benchmark")
    print("=" * 50)
    
    # Basic test
    flash_working = test_flash_attention()
    
    # Detailed benchmark
    if flash_working:
        benchmark_attention_types()
    
    print("\nRecommendations:")
    if flash_working:
        print("Use Flash Attention for maximum speed")
    else:
        print("Use SDPA attention (still optimized)")
    print("Use bfloat16 precision on modern GPUs")
    print("Enable TF32 for additional speedup")