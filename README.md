# Medical AI Assistant - Domain-Specific LLM Fine-tuning

> **Fine-tuned Open-Sourced LLMs for medical question answering with 4-bit quantization and LoRA adaptation**

## Project Overview

This project demonstrates advanced LLM fine-tuning techniques for domain-specific applications in healthcare.

### Evaluation 

The models are evaulated on Multiple Choices, Open-ended Question Answering and Clinical Reasoning. 


## Performance Results
![Performance Analysis](comprehensive_evaluation/detailed_medical_evaluation_report.md)


## Quick Start

### Installation
```bash
cd medical_llm
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Download data 
```bash
# (Option 1): create sample data (fastest) 
python scripts/download_data.py --create-sample

# (Option 2): download real datasets from hugging face and process data to our format 
python sripts/download_data.py --dataset all
```


### Usage
#### Data Prepration
```bash
python scripts/integrate_datasets.py --create-datasets
```
#### Finetune
```bash
python src/training/advanced_train.py   --base-model 'gpt2-medium'   --dataset 'data/real_datasets/medium_comprehensive.json'   --output 'models/medical-medium_comprehensive'   --epochs 3
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
python enhanced_evaluation.py \
  --model-path "models/medical-gpt2-tutor" \
  --compare-models "models/medical-llama2-tutor" "models/baseline-model" \
  --model-names "GPT2-FineTuned" "Llama2-4bit-QLoRA" "Baseline" \
  --output-dir "comprehensive_evaluation"



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

---

