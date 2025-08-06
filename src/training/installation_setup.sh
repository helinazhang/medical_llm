#!/bin/bash
# Ultra-Fast Training Setup for L4 GPU

echo "🚀 Setting up ultra-fast training environment..."

# 1. Install Flash Attention 2 (CRITICAL for speed)
echo "📦 Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# 2. Install latest optimized transformers
echo "📦 Installing optimized transformers..."
pip install transformers[torch] --upgrade
pip install accelerate --upgrade
pip install peft --upgrade

# 3. Install speed optimization packages
echo "📦 Installing speed optimizations..."
pip install torch-compile-backend
pip install triton  # For fused operations

# 4. Check GPU optimization
echo "🎮 Checking GPU setup..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else 'No GPU')
print(f'Torch version: {torch.__version__}')

# Check Flash Attention
try:
    import flash_attn
    print('Flash Attention available')
except ImportError:
    print('Flash Attention NOT available')

# Check bfloat16 support
if torch.cuda.is_available():
    print(f'BFloat16 support: {torch.cuda.is_bf16_supported()}')
    print(f'TF32 support: {torch.backends.cuda.matmul.allow_tf32}')
"

# 5. Set environment variables for speed
echo "⚙️ Setting optimization environment variables..."
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TOKENIZERS_PARALLELISM=true