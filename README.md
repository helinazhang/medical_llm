# Medical AI Assistant - Domain-Specific LLM Fine-tuning

> **Fine-tuned Llama 2 7B model for medical question answering with 4-bit quantization and LoRA adaptation**

## Project Overview

This project demonstrates advanced LLM fine-tuning techniques for domain-specific applications in healthcare. By leveraging **Llama2 7B** and **Mixtral** model with **QLoRA** (4-bit quantization + LoRA), we achieve 75%/67% accuracy on medical benchmarks (Multiple Choice Questions)

## Performance Results

### 1. **llama-2-FineTuned**

| Attribute | Value |
|-----------|-------|
| **Model Type** | `auto` |
| **Architecture** | Decoder-Only |
| **Base Model** | `None` |
| **Multiple Choice Accuracy** | 75.0% |
| **Open-Ended Keyword Score** | 35.3% |
| **Clinical Reasoning Accuracy** | 50.0% |


### 2. **mixtral-finetuned**

| Attribute | Value |
|-----------|-------|
| **Model Type** | `auto` |
| **Architecture** | Decoder-Only |
| **Base Model** | `None` |
| **Multiple Choice Accuracy** | 66.7% |
| **Open-Ended Keyword Score** | 34.0% |
| **Clinical Reasoning Accuracy** | 75.0% |

## Quick Start

### Installation
```bash
git clone https://github.com/helinazhang/medical-llm-suite.git
cd medical-llm-suite
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Download data 
```bash
# (Option 1): create sample data (fastest) 
python scripts/download_data.py --create-sample

# (Option 2): download real datasets from hugging face and process data to our format 
python scripts/download_data.py --dataset all
```


### Usage
#### Data Prepration
```bash
python scripts/integrate_datasets.py --create-datasets
```

### Data Loader
```bash
python src/data/real_dataset_loader.py --datasets medqa medmcqa pubmedqa --sample-size 1500 --format instruction --output data/model_data/llama_medical_data.json


python src/data/real_dataset_loader.py --datasets medqa medmcqa pubmedqa --sample-size 1500 --format mixtral --output data/model_data/mixtral_medical_data.json
```



#### Finetune
```bash
python src/training/faster_train_llama_medical.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --dataset-path /data/model_data/llama_medical_data_train.json \
  --output-dir models/llama-flash \
  --max-length 512 \
  --mode flash \
  --epochs 3 

python src/training/faster_train_mixtral_medical.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset-path data/model_data/mixtral_medical_data_train.json \
    --output-dir models/mixtral-medical-model \
    --max-length 1024 \
    --epochs 2 \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --learning-rate 5e-5 \
    --validation-split 0.1

```
#### Demo 
```python
from medical_llm import MedicalLLMDemo

```
#### Register model 
```bash 
python src/utils/model_manager.py register \
  --name "medical-gpt2-base" \
  --path "models/medical-gpt2" \
  --description "Initial GPT2 medical training (baseline)"
```

### Evaluation
```bash
python src/evaluation/evaluate_medical_llm.py   --model-path "models/medical-gpt2"   --compare-models "models/medical-gpt2-5-epochs" "models/medical-llama2-tutor" "models/ultra-fast-medical-llama"  --model-names "GPT2-FineTuned-2-Epochs" "GPT2-FineTuned-5-Epochs" "Llama2-4bit-QLoRA-2-Epochs" "Llama2-4bit-QLoRA-3-Epochs"   --output-dir "comprehensive_evaluation"
```

# Load fine-tuned model
demo = MedicalLLMDemo("./models/medical-llm-final")

# Ask medical questions
response = demo.generate_medical_response(
    question="What are the symptoms of Type 2 diabetes?",
    context="Patient is 45 years old, overweight"
)
print(response)
```



### Interactive Demo
```bash
python demo.py  # Launches Gradio interface
```

## Future work
1. Finetuning on newer models like Qwen3-8B, etc.
2. Multi-round question answering to build an "online doctor"