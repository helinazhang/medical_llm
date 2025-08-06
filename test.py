1. data loader 
1) mixtral 
python src/data/real_dataset_loader.py --datasets medqa medmcqa pubmedqa --sample-size 1500 --format mixtral --output data/model_data/mixtral_medical_data.json

2) llama 
python src/data/real_dataset_loader.py --datasets medqa medmcqa pubmedqa --sample-size 1500 --format instruction --output data/model_data/llama_medical_data.json



1. llama 

python src/training/faster_train_llama_medical.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --dataset-path /home/aicadium/documents/github_bf/medical-llm-suite/data/model_data/llama_medical_data_train.json \
  --output-dir models/llama-flash \
  --max-length 512 \
  --mode flash \
  --epochs 3 

(with 4 bit quantization)

python src/training/faster_train_llama_medical.py \
    --model-name "meta-llama/Llama-2-7b-hf" \
    --dataset-path "/home/aicadium/documents/github_bf/medical-llm-suite/data/model_data/llama_medical_data_train.json" \
    --output-dir "models/llama-safe" \
    --max-length 512



2. mistral
python src/training/faster_train_mixtral_medical.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset-path /home/aicadium/documents/github_bf/medical-llm-suite/data/model_data/mixtral_medical_data_train.json \
    --output-dir models/mixtral-medical-model \
    --max-length 1024 \
    --epochs 2 \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --learning-rate 5e-5 \
    --validation-split 0.1


python src/training/faster_train_mixtral_medical.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset-path /home/aicadium/documents/github_bf/medical-llm-suite/data/model_data/mixtral_medical_data_train.json \
    --output-dir models/mixtral-output \
    --use-quantization \
    --epochs 3 \
    --max-length 2048




python src/evaluation/evaluate_medical_llm.py \
  --model-path "models/llama-safe" \
  --compare-models "models/mixtral-output" \
  --model-names "llama-2-FineTuned" "mixtral-finetuned" \
  --output-dir "comprehensive_evaluation"





python src/training/faster_train_llama_medical.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --dataset-path /home/aicadium/documents/github_bf/medical-llm-suite/data/clean/clean_medical_dataset_train.json \
  --output-dir ./trained_model_llama_fast \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --max-length 512 \
  --use-quantization \
  --use-lora \
  --model-type llama

CUDA_VISIBLE_DEVICES=0 \
WORLD_SIZE=1 \
RANK=0 \
LOCAL_RANK=0 \
python src/training/faster_train_llama_medical.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --dataset-path /home/aicadium/documents/github_bf/medical-llm-suite/data/clean/clean_medical_dataset_train.json \
  --output-dir models/trained_model_llama_fast \
  --model-type llama \
  --max-length 1024 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --use-quantization \
  --use-lora




python -c "
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import subprocess
import sys

cmd = [
    sys.executable, 'src/training/faster_train_llama_medical.py',
    '--model-name', 'meta-llama/Llama-2-7b-hf',
    '--dataset-path', '/home/aicadium/documents/github_bf/medical-llm-suite/data/clean/clean_medical_dataset_train.json',
    '--output-dir', './trained_model_llama_fast',
    '--use-flash-attention',
    '--use-lora',
    '--use-quantization',
    '--batch-size', '4',
    '--gradient-accumulation', '8'
]

subprocess.run(cmd)
"