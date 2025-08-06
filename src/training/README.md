# faster_train_llama_medical
## Key Speed Improvements
1. Flash Attention 2 (50-80% speedup)

Reduces memory usage and increases speed dramatically
Essential for L4 GPU optimization

2. Optimized Batch Sizes

Increased batch size to 4 (from 1)
Reduced gradient accumulation to 4 (from 16)
Better GPU utilization

3. BFloat16 + TF32 (20-30% speedup)

More efficient than FP16 on modern GPUs
Better numerical stability

4. Torch Compile (10-20% speedup)

Latest PyTorch optimization
Automatically optimizes model execution

5. Data Loading Optimizations

Multi-threaded data loading (4 workers)
Pin memory for faster GPU transfer
Group by length for efficiency

6. Reduced I/O Operations

Less frequent logging/saving
Skip evaluation during training
Minimal checkpointing

## Set up 
Run `bash installation_setup.sh` before faster_train_llama_medical.py 


## Finetuning
``` bash 
python src/training/quick_fix.py \
  --dataset "/home/aicadium/documents/github_bf/medical-llm-suite/data/real_datasets/medium_comprehensive.json" \
  --output "models/ultra-fast-medical-llama" \
  --epochs 1 

python src/training/faster_train_llama_medical.py \
  --dataset "/home/aicadium/documents/github_bf/medical-llm-suite/data/real_datasets/medium_comprehensive.json" \
  --output "models/ultra-fast-medical-llama" \
  --epochs 3 \
  --max-length 512  # Shorter sequences = faster training
```