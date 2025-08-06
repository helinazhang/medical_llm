# Medical LLM Model Evaluation Report

**Generated:** 2025-06-22 07:33:24  
**Evaluation Framework:** Enhanced Medical Benchmark Suite  
**Models Evaluated:** 2

---

## Executive Summary

This comprehensive evaluation report analyzes the performance of fine-tuned medical language models across multiple dimensions including factual accuracy, clinical reasoning, and response quality. The evaluation framework uses model-specific prompt templates optimized for each architecture.

### Key Findings


- ** Best Overall Accuracy:** llama-2-FineTuned (75.0%)
- ** Fastest Response:** llama-2-FineTuned (2.98s avg)
- ** Model Types:** auto
- ** Performance Range:** 66.7% - 75.0%


---

## Models Evaluated


### 1. **llama-2-FineTuned**

| Attribute | Value |
|-----------|-------|
| **Model Type** | `auto` |
| **Architecture** | Decoder-Only |
| **Base Model** | `None` |
| **Multiple Choice Accuracy** | 75.0% |
| **Open-Ended Keyword Score** | 35.3% |
| **Clinical Reasoning Accuracy** | 50.0% |

**Model Path:** `models/llama-safe`


### 2. **mixtral-finetuned**

| Attribute | Value |
|-----------|-------|
| **Model Type** | `auto` |
| **Architecture** | Decoder-Only |
| **Base Model** | `None` |
| **Multiple Choice Accuracy** | 66.7% |
| **Open-Ended Keyword Score** | 34.0% |
| **Clinical Reasoning Accuracy** | 75.0% |

**Model Path:** `models/mixtral-output`



---

## Performance Analysis

### Overall Performance Metrics

| Model | Architecture | MC Accuracy | OE Keywords | CR Accuracy | Avg Time | Performance Score* |
|-------|-------------|-------------|-------------|-------------|----------|-------------------|
| llama-2-FineTuned | Dec-Only | 75.0% | 35.3% | 50.0% | 2.98s | 53.4% |
| mixtral-finetuned | Dec-Only | 66.7% | 34.0% | 75.0% | 3.16s | 58.6% |


*Performance Score = Average of MC Accuracy, OE Keywords, and CR Accuracy

### Category-wise Performance Analysis


#### llama-2-FineTuned - Category Performance

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Emergency Medicine | 100.0% | 1/1 |
| Endocrinology | 50.0% | 1/2 |
| Pulmonology | 100.0% | 1/1 |
| Pharmacology | 100.0% | 1/1 |
| Infectious Disease | 33.3% | 1/3 |
| Neurology | 100.0% | 2/2 |
| Biochemistry | 100.0% | 1/1 |
| Public Health | 100.0% | 1/1 |

#### mixtral-finetuned - Category Performance

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Emergency Medicine | 100.0% | 1/1 |
| Endocrinology | 100.0% | 2/2 |
| Pulmonology | 100.0% | 1/1 |
| Pharmacology | 0.0% | 0/1 |
| Infectious Disease | 33.3% | 1/3 |
| Neurology | 50.0% | 1/2 |
| Biochemistry | 100.0% | 1/1 |
| Public Health | 100.0% | 1/1 |


### Clinical Reasoning Analysis


#### llama-2-FineTuned - Clinical Reasoning Performance

| Scenario | Correct Diagnosis | Response Quality |
|----------|------------------|------------------|
| A 45-year-old man presents with severe abdominal pain radiat... | Y | 47 words |
| A 70-year-old woman with a history of atrial fibrillation su... | N | 50 words |
| A 30-year-old pregnant woman at 20 weeks gestation presents ... | N | 47 words |
| A 55-year-old man with diabetes presents with chest pain, di... | Y | 50 words |

#### mixtral-finetuned - Clinical Reasoning Performance

| Scenario | Correct Diagnosis | Response Quality |
|----------|------------------|------------------|
| A 45-year-old man presents with severe abdominal pain radiat... | Y 26 words |
| A 70-year-old woman with a history of atrial fibrillation su... | N 41 words |
| A 30-year-old pregnant woman at 20 weeks gestation presents ... | Y 41 words |
| A 55-year-old man with diabetes presents with chest pain, di... | N 24 words |


---

## Detailed Analysis

### Model Architecture Impact

**Decoder-Only Models:** 2 models
- Advantages: Better instruction following, natural conversation flow
- Challenges: Context length limitations, potential for repetition

**Encoder-Decoder Models:** 0 models  
- Advantages: Better structured output, task-specific optimization
- Challenges: Less natural conversation, prompt format sensitivity

### Prompt Template Optimization

Different model types used optimized prompt templates:

#### Mistral/Mixtral Models
```
<s>[INST] You are a medical expert. Answer this medical question accurately.

{question}
{options}

Provide the correct answer. [/INST]
```

#### GPT-2 Models  
```
Q: {question}
{options}
A:
```

#### Llama Models
```
### Question:
{question}
{options}

### Answer:
```

### Performance Insights


1. **Average Performance:** 70.8% ± 5.9%
2. **Performance Consistency:** High
3. **Speed vs Accuracy Trade-off:** Models with higher accuracy tend to have similar response times


### Recommendations

#### For Clinical Practice:
1. **High-Stakes Decisions:** Use models with >80% accuracy on clinical reasoning
2. **Real-time Applications:** Consider models with <2s response time
3. **Patient Education:** Models with good open-ended performance are suitable

#### For Further Development:
1. **Data Augmentation:** Focus on categories with lower performance
2. **Prompt Engineering:** Optimize templates for specific model architectures  
3. **Fine-tuning:** Additional training on clinical reasoning scenarios

---

## Methodology

### Evaluation Framework
- **Multiple Choice Questions:** 12 questions across medical specialties
- **Open-ended Questions:** 5 pharmacology and mechanism questions
- **Clinical Reasoning:** 4 diagnostic scenarios

### Metrics
- **Accuracy:** Exact match for multiple choice, keyword presence for open-ended
- **Response Time:** Average generation time per question
- **Semantic Similarity:** Cosine similarity with reference answers (when available)
- **Clinical Relevance:** Key feature identification in diagnostic scenarios

### Deterministic Evaluation
- **Fixed Random Seeds:** All models evaluated with seed=42
- **Consistent Prompts:** Same prompts used across all models of same type
- **Temperature Control:** Low temperature (0.01-0.1) for multiple choice, moderate (0.1-0.3) for reasoning
- **Reproducible Results:** All evaluations can be exactly reproduced

---

## Files Generated

- **Detailed JSON Reports:** `evaluation_report_{model_name}_{timestamp}.json`
- **Comparison CSV:** `model_comparison_{timestamp}.csv` 
- **Visualization Dashboard:** `model_comparison_plots_{timestamp}.png`
- **This Report:** `detailed_medical_evaluation_report_{timestamp}.md`

---
