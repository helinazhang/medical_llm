#!/usr/bin/env python3
"""
Enhanced Medical LLM Evaluation System with Mistral Support
Model-agnostic evaluation with proper prompting for different architectures including Mistral
Deterministic evaluation for reproducible results
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTemplates:
    """Prompt templates for different model types and tasks including Mistral"""
    
    @staticmethod
    def get_templates_for_model(model_type: str) -> Dict[str, str]:
        """Get appropriate prompt templates for each model type"""
        
        if model_type == "gpt2":
            return {
                "multiple_choice": "Q: {question}\n{options}\nA:",
                "open_ended": "Q: {question}\nA:",
                "clinical_reasoning": "Clinical case: {scenario}\nDiagnosis:",
                "instruction": "{question}\nAnswer:",
                "conversation": "Human: {question}\n\nAssistant:"
            }
        
        elif model_type == "llama":
            return {
                "multiple_choice": "### Question:\n{question}\n{options}\n\n### Answer:",
                "open_ended": "### Question:\n{question}\n\n### Answer:",
                "clinical_reasoning": "### Clinical Scenario:\n{scenario}\n\n### Most Likely Diagnosis:",
                "instruction": "### Instruction:\nAnswer the following medical question.\n\n### Question:\n{question}\n\n### Response:",
                "conversation": "[INST] {question} [/INST]"
            }
        
        elif model_type == "mistral" or model_type == "mixtral":
            return {
                "multiple_choice": "<s>[INST] You are a medical expert. Answer this medical question accurately.\n\n{question}\n{options}\n\nProvide the correct answer. [/INST]",
                "open_ended": "<s>[INST] You are a medical expert. Answer this medical question accurately.\n\n{question} [/INST]",
                "clinical_reasoning": "<s>[INST] You are a medical expert. Analyze this clinical scenario and provide the most likely diagnosis.\n\n{scenario} [/INST]",
                "instruction": "<s>[INST] You are a medical expert. {question} [/INST]",
                "conversation": "<s>[INST] {question} [/INST]"
            }
        
        elif model_type == "dialogue" or model_type == "blenderbot":
            return {
                "multiple_choice": "{question} {options}",
                "open_ended": "{question}",
                "clinical_reasoning": "What is the most likely diagnosis for this clinical scenario: {scenario}",
                "instruction": "{question}",
                "conversation": "{question}"
            }
        
        elif model_type == "t5":
            return {
                "multiple_choice": "question: {question} options: {options}",
                "open_ended": "question: {question}",
                "clinical_reasoning": "diagnose: {scenario}",
                "instruction": "medical question: {question}",
                "conversation": "answer: {question}"
            }
        
        elif model_type == "alpaca" or model_type == "vicuna":
            return {
                "multiple_choice": "### Instruction:\nAnswer the following medical multiple choice question.\n\n### Input:\n{question}\n{options}\n\n### Response:",
                "open_ended": "### Instruction:\nAnswer the following medical question.\n\n### Input:\n{question}\n\n### Response:",
                "clinical_reasoning": "### Instruction:\nProvide the most likely diagnosis for the clinical scenario.\n\n### Input:\n{scenario}\n\n### Response:",
                "instruction": "### Instruction:\n{question}\n\n### Response:",
                "conversation": "### Instruction:\n{question}\n\n### Response:"
            }
        
        else:  # default/auto
            return {
                "multiple_choice": "Medical Question: {question}\n{options}\nAnswer:",
                "open_ended": "Medical Question: {question}\nAnswer:",
                "clinical_reasoning": "Clinical Scenario: {scenario}\nMost likely diagnosis:",
                "instruction": "Question: {question}\nAnswer:",
                "conversation": "Question: {question}\nAnswer:"
            }

class MedicalBenchmarks:
    """Enhanced Medical evaluation benchmarks with more comprehensive questions"""
    
    @staticmethod
    def get_clinical_questions() -> List[Dict]:
        """Clinical reasoning questions"""
        return [
            {
                "question": "A 65-year-old diabetic patient presents with chest pain, sweating, and nausea. What is the most likely diagnosis?",
                "options": ["Gastroesophageal reflux", "Myocardial infarction", "Anxiety attack", "Muscle strain"],
                "correct_answer": "Myocardial infarction",
                "category": "Emergency Medicine",
                "difficulty": "intermediate",
                "reasoning": "Classic presentation of MI in diabetic patient with typical symptoms"
            },
            {
                "question": "What is the first-line treatment for type 2 diabetes mellitus?",
                "options": ["Insulin", "Metformin", "Sulfonylureas", "Lifestyle modification only"],
                "correct_answer": "Metformin",
                "category": "Endocrinology",
                "difficulty": "basic",
                "reasoning": "Metformin is the first-line pharmacological treatment for T2DM"
            },
            {
                "question": "A patient with COPD presents with increased dyspnea and purulent sputum. What is the most appropriate initial treatment?",
                "options": ["Bronchodilators only", "Antibiotics and corticosteroids", "Oxygen therapy only", "Chest physiotherapy"],
                "correct_answer": "Antibiotics and corticosteroids",
                "category": "Pulmonology",
                "difficulty": "intermediate",
                "reasoning": "COPD exacerbation with signs of infection requires antibiotics and steroids"
            },
            {
                "question": "What is the mechanism of action of ACE inhibitors?",
                "options": ["Block calcium channels", "Inhibit angiotensin-converting enzyme", "Block beta receptors", "Diuretic effect"],
                "correct_answer": "Inhibit angiotensin-converting enzyme",
                "category": "Pharmacology",
                "difficulty": "basic",
                "reasoning": "ACE inhibitors block the conversion of angiotensin I to angiotensin II"
            },
            {
                "question": "A 25-year-old woman presents with palpitations, weight loss, and heat intolerance. TSH is suppressed and free T4 is elevated. What is the most likely diagnosis?",
                "options": ["Hypothyroidism", "Hyperthyroidism", "Thyroid cancer", "Normal variant"],
                "correct_answer": "Hyperthyroidism",
                "category": "Endocrinology",
                "difficulty": "intermediate",
                "reasoning": "Classic symptoms with lab findings consistent with hyperthyroidism"
            },
            {
                "question": "Which antibiotic is first-line for treating uncomplicated urinary tract infection in women?",
                "options": ["Amoxicillin", "Trimethoprim-sulfamethoxazole", "Ciprofloxacin", "Azithromycin"],
                "correct_answer": "Trimethoprim-sulfamethoxazole",
                "category": "Infectious Disease",
                "difficulty": "basic",
                "reasoning": "TMP-SMX is first-line for uncomplicated UTI when resistance rates are low"
            },
            {
                "question": "What is the most common cause of community-acquired pneumonia?",
                "options": ["Haemophilus influenzae", "Streptococcus pneumoniae", "Mycoplasma pneumoniae", "Legionella pneumophila"],
                "correct_answer": "Streptococcus pneumoniae",
                "category": "Infectious Disease",
                "difficulty": "basic",
                "reasoning": "S. pneumoniae is the most common bacterial cause of CAP"
            },
            {
                "question": "A patient presents with sudden severe headache described as 'worst headache of my life'. What is the most concerning diagnosis?",
                "options": ["Migraine", "Tension headache", "Subarachnoid hemorrhage", "Cluster headache"],
                "correct_answer": "Subarachnoid hemorrhage",
                "category": "Neurology",
                "difficulty": "intermediate",
                "reasoning": "Thunderclap headache is classic for SAH and requires immediate investigation"
            },
            {
                "question": "What is the drug of choice for Falciparum Malaria?",
                "options": ["Chloroquine", "Mefloquine", "ACT", "Proguanil"],
                "correct_answer": "ACT",
                "category": "Infectious Disease",
                "difficulty": "intermediate",
                "reasoning": "Artemisinin Combination Therapy is first-line for falciparum malaria"
            },
            {
                "question": "True about neuropraxia:",
                "options": ["Prolongation of conduction velocity", "Good prognosis", "Both", "None"],
                "correct_answer": "Both",
                "category": "Neurology",
                "difficulty": "intermediate",
                "reasoning": "Neuropraxia involves conduction delay and has good prognosis"
            },
            {
                "question": "Acyl carnitine functions in:",
                "options": ["Transport of long chain fatty acid", "Transport of short chain fatty acid", "Transport of NADH", "Transport of FADH"],
                "correct_answer": "Transport of long chain fatty acid",
                "category": "Biochemistry",
                "difficulty": "basic",
                "reasoning": "Acyl carnitine is essential for fatty acid oxidation in mitochondria"
            },
            {
                "question": "Millennium development goals formulated in 2000 were to be achieved by:",
                "options": ["2005", "2015", "2010", "2020"],
                "correct_answer": "2015",
                "category": "Public Health",
                "difficulty": "basic",
                "reasoning": "MDGs had a 15-year timeline from 2000 to 2015"
            }
        ]
    
    @staticmethod
    def get_diagnostic_scenarios() -> List[Dict]:
        """Diagnostic reasoning scenarios"""
        return [
            {
                "scenario": "A 45-year-old man presents with severe abdominal pain radiating to the back, nausea, and vomiting. He has a history of alcohol use disorder.",
                "differential_diagnosis": ["Acute pancreatitis", "Peptic ulcer disease", "Cholecystitis", "Myocardial infarction"],
                "most_likely": "Acute pancreatitis",
                "key_features": ["abdominal pain radiating to back", "alcohol history", "nausea and vomiting"],
                "category": "Gastroenterology"
            },
            {
                "scenario": "A 70-year-old woman with a history of atrial fibrillation suddenly develops left-sided weakness and speech difficulties.",
                "differential_diagnosis": ["Ischemic stroke", "Hemorrhagic stroke", "Transient ischemic attack", "Migraine with aura"],
                "most_likely": "Ischemic stroke",
                "key_features": ["sudden onset", "focal neurological deficits", "atrial fibrillation history"],
                "category": "Neurology"
            },
            {
                "scenario": "A 30-year-old pregnant woman at 20 weeks gestation presents with burning sensation during urination and increased frequency.",
                "differential_diagnosis": ["Urinary tract infection", "Normal pregnancy changes", "Kidney stones", "Sexually transmitted infection"],
                "most_likely": "Urinary tract infection",
                "key_features": ["dysuria", "frequency", "pregnancy"],
                "category": "Obstetrics/Gynecology"
            },
            {
                "scenario": "A 55-year-old man with diabetes presents with chest pain, diaphoresis, and shortness of breath for the past 2 hours.",
                "differential_diagnosis": ["Myocardial infarction", "Angina", "Panic attack", "GERD"],
                "most_likely": "Myocardial infarction",
                "key_features": ["chest pain", "diabetes", "diaphoresis", "acute onset"],
                "category": "Cardiology"
            }
        ]
    
    @staticmethod
    def get_pharmacology_questions() -> List[Dict]:
        """Pharmacology and drug interaction questions"""
        return [
            {
                "question": "What is a major side effect of long-term corticosteroid use?",
                "expected_keywords": ["osteoporosis", "bone", "fracture", "calcium", "density", "bone loss"],
                "category": "Pharmacology",
                "difficulty": "basic",
                "reference_answer": "Long-term corticosteroid use can cause osteoporosis and increased fracture risk due to decreased bone density."
            },
            {
                "question": "Why should ACE inhibitors be used cautiously in patients with renal disease?",
                "expected_keywords": ["hyperkalemia", "potassium", "kidney", "creatinine", "GFR", "renal function"],
                "category": "Pharmacology", 
                "difficulty": "intermediate",
                "reference_answer": "ACE inhibitors can cause hyperkalemia and worsen renal function in patients with kidney disease."
            },
            {
                "question": "What are the common side effects of statins?",
                "expected_keywords": ["muscle", "myalgia", "rhabdomyolysis", "liver", "CK", "pain"],
                "category": "Pharmacology",
                "difficulty": "basic",
                "reference_answer": "Common statin side effects include muscle pain, myalgia, and rarely rhabdomyolysis with elevated CK levels."
            },
            {
                "question": "What is the mechanism of action of warfarin?",
                "expected_keywords": ["vitamin K", "coagulation", "clotting", "anticoagulant", "INR"],
                "category": "Pharmacology",
                "difficulty": "intermediate",
                "reference_answer": "Warfarin inhibits vitamin K-dependent clotting factors, requiring INR monitoring for anticoagulation."
            },
            {
                "question": "What are the contraindications for metformin?",
                "expected_keywords": ["kidney", "renal", "contrast", "lactic acidosis", "GFR", "creatinine"],
                "category": "Pharmacology",
                "difficulty": "intermediate",
                "reference_answer": "Metformin is contraindicated in severe renal impairment due to risk of lactic acidosis."
            }
        ]

class ModelEvaluator:
    """Enhanced model evaluation system with Mistral-specific handling"""
    
    def __init__(self, model_path: str, model_type: str = "auto", base_model: str = None, prompt_format: str = "auto"):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.base_model = base_model
        self.prompt_format = prompt_format
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        self.model_name = os.path.basename(model_path)
        self.is_encoder_decoder = False
        
        # Detect model type and load
        self._detect_and_load_model()
        
        # Get appropriate prompt templates
        self.prompt_templates = PromptTemplates.get_templates_for_model(self.model_type)
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            logger.warning("Could not load sentence transformer for semantic evaluation")
    
    def _detect_model_architecture(self):
        """Enhanced model architecture detection including Mistral"""
        model_path_lower = self.model_path.lower()
        
        # Check for specific model types in path
        if any(x in model_path_lower for x in ["mistral", "mixtral", "zephyr", "dolphin-mistral", "openorca-mistral"]):
            return "mistral"
        elif any(x in model_path_lower for x in ["gpt2", "gpt-2"]):
            return "gpt2"
        elif any(x in model_path_lower for x in ["llama", "alpaca", "vicuna"]):
            return "llama"
        elif any(x in model_path_lower for x in ["blender", "dialogue"]):
            return "dialogue"
        elif any(x in model_path_lower for x in ["t5", "flan"]):
            return "t5"
        elif any(x in model_path_lower for x in ["alpaca"]):
            return "alpaca"
        elif any(x in model_path_lower for x in ["vicuna"]):
            return "vicuna"
        
        # Try to detect from config
        try:
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                model_type = config.get("model_type", "").lower()
                architectures = config.get("architectures", [])
                
                if "mistral" in model_type or any("Mistral" in arch for arch in architectures):
                    return "mistral"
                elif "gpt2" in model_type or any("GPT2" in arch for arch in architectures):
                    return "gpt2"
                elif "llama" in model_type or any("Llama" in arch for arch in architectures):
                    return "llama"
                elif "blender" in model_type or any("Blender" in arch for arch in architectures):
                    return "dialogue"
                elif "t5" in model_type or any("T5" in arch for arch in architectures):
                    return "t5"
        except:
            pass
        
        return "auto"
    
    def _detect_and_load_model(self):
        """Detect model type and load appropriately"""
        logger.info(f"Loading model from: {self.model_path}")
        
        if self.model_type == "auto":
            self.model_type = self._detect_model_architecture()
            logger.info(f"Detected model type: {self.model_type}")
        
        try:
            # Load based on detected/specified model type
            if self.model_type == "gpt2":
                self._load_gpt2_model()
            elif self.model_type == "llama":
                self._load_llama_model()
            elif self.model_type == "mistral" or self.model_type == "mixtral":
                self._load_mistral_model()
            elif self.model_type == "dialogue" or self.model_type == "blenderbot":
                self._load_dialogue_model()
            elif self.model_type == "t5":
                self._load_t5_model()
            elif self.model_type in ["alpaca", "vicuna"]:
                self._load_instruction_model()
            else:
                self._load_generic_model()
            
            self.model.eval()
            logger.info(f"Model loaded successfully - Type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try generic loading as fallback
            try:
                logger.info("Attempting generic model loading...")
                self._load_generic_model()
                self.model.eval()
                logger.info("Generic model loading successful")
            except Exception as e2:
                logger.error(f"Generic loading also failed: {e2}")
                raise e
    
    def _load_mistral_model(self):
        """Load Mistral/Mixtral models"""
        logger.info("Loading Mistral model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Check if it's a PEFT model
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            if not self.base_model:
                # Try to get base model from adapter config
                config_path = os.path.join(self.model_path, "adapter_config.json")
                try:
                    with open(config_path, 'r') as f:
                        adapter_config = json.load(f)
                    self.base_model = adapter_config.get("base_model_name_or_path", "mistralai/Mistral-7B-v0.1")
                except:
                    self.base_model = "mistralai/Mistral-7B-v0.1"  # Default fallback
            
            logger.info(f"Loading PEFT Mistral model with base: {self.base_model}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            logger.info("Loading full Mistral model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Set padding token for Mistral
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Mistral model loaded - Vocab size: {len(self.tokenizer)}")
    
    def _load_gpt2_model(self):
        """Load GPT2 model"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        
        # Check if it's a PEFT model
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            if not self.base_model:
                self.base_model = "gpt2"
            base_model = GPT2LMHeadModel.from_pretrained(self.base_model)
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_llama_model(self):
        """Load Llama model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Check if it's a PEFT model
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            if not self.base_model:
                config_path = os.path.join(self.model_path, "adapter_config.json")
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                self.base_model = adapter_config.get("base_model_name_or_path", "meta-llama/Llama-2-7b-hf")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_dialogue_model(self):
        """Load dialogue model (BlenderBot, etc.)"""
        try:
            self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_path)
            self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_path)
            self.is_encoder_decoder = True
        except:
            # Fallback to generic
            self._load_generic_model()
    
    def _load_t5_model(self):
        """Load T5 model"""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.is_encoder_decoder = True
        except:
            # Fallback to generic
            self._load_generic_model()
    
    def _load_instruction_model(self):
        """Load instruction-tuned models (Alpaca, Vicuna, etc.)"""
        # These are typically Llama-based, so use similar loading
        self._load_llama_model()
    
    def _load_generic_model(self):
        """Generic model loading"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_prompt(self, template_key: str, **kwargs) -> str:
        """Format prompt using appropriate template"""
        template = self.prompt_templates.get(template_key, self.prompt_templates["instruction"])
        return template.format(**kwargs)
    
    def _format_multiple_choice_options(self, options: List[str]) -> str:
        """Format multiple choice options consistently"""
        if self.model_type in ["dialogue", "t5"]:
            # Simple format for dialogue models
            return " | ".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        else:
            # Standard format
            return "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.1, deterministic: bool = True) -> Tuple[str, float]:
        """Generate response with model-specific handling including Mistral - DETERMINISTIC VERSION"""
        try:
            start_time = time.time()
            
            # Set deterministic generation parameters
            if deterministic:
                # Force deterministic behavior
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)
                temperature = max(0.01, temperature)  # Minimum temperature for stability
            
            # Adjust max prompt length based on model type
            if self.model_type == "mistral":
                max_prompt_length = 2048  # Mistral can handle longer contexts
            elif self.model_type == "dialogue":
                max_prompt_length = 512
            else:
                max_prompt_length = 1024
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length)
            
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.is_encoder_decoder:
                    # For encoder-decoder models (T5, BlenderBot) - DETERMINISTIC
                    generation_kwargs = {
                        **inputs,
                        "max_length": max_length,
                        "do_sample": False if deterministic else True,  # Use greedy decoding for deterministic
                        "num_beams": 1 if deterministic else 4,
                        "early_stopping": True,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    }
                    
                    if not deterministic:
                        generation_kwargs.update({
                            "temperature": temperature,
                            "top_p": 0.9,
                            "repetition_penalty": 1.1,
                        })
                    
                    outputs = self.model.generate(**generation_kwargs)
                else:
                    # For decoder-only models (GPT2, Llama, Mistral) - DETERMINISTIC
                    generation_kwargs = {
                        **inputs,
                        "max_new_tokens": max_length,
                        "do_sample": False if deterministic else True,  # Use greedy decoding for deterministic
                        "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    }
                    
                    if deterministic:
                        # Deterministic generation - greedy decoding
                        generation_kwargs.update({
                            "num_beams": 1,
                            "repetition_penalty": 1.0,  # No repetition penalty for deterministic
                        })
                    else:
                        # Non-deterministic generation
                        generation_kwargs.update({
                            "temperature": temperature,
                            "top_p": 0.9,
                            "repetition_penalty": 1.1,
                        })
                    
                    # Mistral-specific optimizations
                    if self.model_type == "mistral":
                        if not deterministic:
                            generation_kwargs.update({
                                "top_k": 50,
                                "temperature": max(0.01, temperature),
                            })
                    
                    outputs = self.model.generate(**generation_kwargs)
            
            # Decode response
            if self.is_encoder_decoder:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the generated part for decoder-only models
                if prompt in response:
                    response = response[len(prompt):].strip()
                elif self.model_type == "mistral" and "[/INST]" in response:
                    # For Mistral, extract response after [/INST]
                    parts = response.split("[/INST]")
                    if len(parts) > 1:
                        response = parts[-1].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            generation_time = time.time() - start_time
            return response, generation_time
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "", 0
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize model response with Mistral-specific handling"""
        if not response:
            return ""
        
        # Remove common artifacts
        response = response.strip()
        
        # Mistral-specific cleaning
        if self.model_type == "mistral":
            # Remove Mistral instruction artifacts
            response = re.sub(r'<s>\s*\[INST\].*?\[/INST\]\s*', '', response)
            response = re.sub(r'</s>.*', '', response)  # Remove everything after </s>
            response = response.replace('<s>', '').replace('</s>', '')
        
        # For instruction models, remove template artifacts
        elif self.model_type in ["alpaca", "vicuna"]:
            # Remove instruction template remnants
            response = re.sub(r'### Response:.*?\n', '', response)
            response = re.sub(r'### Instruction:.*?\n', '', response)
        
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response)
        
        # Take first meaningful sentence/response
        sentences = response.split('\n')
        if sentences:
            response = sentences[0].strip()
        
        # Limit length to prevent very long responses
        words = response.split()
        if len(words) > 50:  # Reasonable limit for evaluation
            response = ' '.join(words[:50])
        
        return response.strip()
    
    def evaluate_multiple_choice(self, questions: List[Dict], deterministic: bool = True) -> Dict:
        """Evaluate multiple choice questions with model-specific prompting - DETERMINISTIC"""
        logger.info("Evaluating multiple choice questions...")
        
        results = []
        correct = 0
        total = len(questions)
        total_time = 0
        
        for i, q in enumerate(questions, 1):
            logger.info(f"Question {i}/{total}")
            
            # Format options appropriately for the model
            options_text = self._format_multiple_choice_options(q['options'])
            
            # Create prompt using appropriate template
            prompt = self._format_prompt(
                "multiple_choice",
                question=q['question'],
                options=options_text
            )
            
            # Generate response with deterministic settings for multiple choice
            response, gen_time = self.generate_response(
                prompt, 
                max_length=50, 
                temperature=0.01,  # Very low temperature for deterministic MC
                deterministic=deterministic
            )
            total_time += gen_time
            
            # Extract answer choice
            predicted_choice = self._extract_choice(response, q['options'])
            is_correct = predicted_choice == q['correct_answer']
            
            if is_correct:
                correct += 1
            
            result = {
                "question": q['question'],
                "correct_answer": q['correct_answer'],
                "predicted_answer": predicted_choice,
                "raw_response": response,
                "is_correct": is_correct,
                "category": q['category'],
                "difficulty": q['difficulty'],
                "generation_time": gen_time,
                "prompt_used": prompt
            }
            results.append(result)
            
            logger.info(f" Correct: {is_correct} - {predicted_choice}")
        
        accuracy = correct / total
        avg_time = total_time / total
        
        # Category-wise and difficulty-wise performance
        category_performance = {}
        difficulty_performance = {}
        
        for result in results:
            # Category performance
            cat = result['category']
            if cat not in category_performance:
                category_performance[cat] = {'correct': 0, 'total': 0}
            category_performance[cat]['total'] += 1
            if result['is_correct']:
                category_performance[cat]['correct'] += 1
            
            # Difficulty performance
            diff = result['difficulty']
            if diff not in difficulty_performance:
                difficulty_performance[diff] = {'correct': 0, 'total': 0}
            difficulty_performance[diff]['total'] += 1
            if result['is_correct']:
                difficulty_performance[diff]['correct'] += 1
        
        # Calculate accuracy for each category/difficulty
        for cat in category_performance:
            category_performance[cat]['accuracy'] = (
                category_performance[cat]['correct'] / category_performance[cat]['total']
            )
        
        for diff in difficulty_performance:
            difficulty_performance[diff]['accuracy'] = (
                difficulty_performance[diff]['correct'] / difficulty_performance[diff]['total']
            )
        
        return {
            "overall_accuracy": accuracy,
            "correct_answers": correct,
            "total_questions": total,
            "average_time_per_question": avg_time,
            "category_performance": category_performance,
            "difficulty_performance": difficulty_performance,
            "detailed_results": results
        }
    
    def evaluate_open_ended(self, questions: List[Dict], deterministic: bool = True) -> Dict:
        """Enhanced open-ended evaluation with model-specific prompting - DETERMINISTIC"""
        logger.info("Evaluating open-ended questions...")
        
        results = []
        total_keyword_score = 0
        total_semantic_score = 0
        total_time = 0
        
        for i, q in enumerate(questions, 1):
            logger.info(f"Question {i}/{len(questions)}")
            
            # Create prompt using appropriate template
            prompt = self._format_prompt("open_ended", question=q['question'])
            
            # Generate response with controlled randomness
            response, gen_time = self.generate_response(
                prompt, 
                max_length=150, 
                temperature=0.3 if deterministic else 0.7,  # Lower temperature for more consistent results
                deterministic=deterministic
            )
            total_time += gen_time
            
            # Keyword-based evaluation
            keyword_score, found_keywords = self._evaluate_keywords(response, q['expected_keywords'])
            
            # Semantic similarity evaluation (if available)
            semantic_score = 0
            if self.sentence_model and 'reference_answer' in q:
                semantic_score = self._evaluate_semantic_similarity(response, q['reference_answer'])
            
            # Response quality metrics
            response_length = len(response.split())
            relevance_score = self._evaluate_relevance(response, q['question'])
            
            total_keyword_score += keyword_score
            if semantic_score > 0:
                total_semantic_score += semantic_score
            
            result = {
                "question": q['question'],
                "response": response,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "relevance_score": relevance_score,
                "response_length": response_length,
                "found_keywords": found_keywords,
                "expected_keywords": q['expected_keywords'],
                "category": q['category'],
                "generation_time": gen_time,
                "prompt_used": prompt
            }
            results.append(result)
        
        avg_keyword_score = total_keyword_score / len(questions)
        avg_semantic_score = total_semantic_score / len(questions) if total_semantic_score > 0 else 0
        avg_time = total_time / len(questions)
        avg_response_length = np.mean([r['response_length'] for r in results])
        
        return {
            "average_keyword_score": avg_keyword_score,
            "average_semantic_score": avg_semantic_score,
            "average_time_per_question": avg_time,
            "average_response_length": avg_response_length,
            "detailed_results": results
        }
    
    def evaluate_clinical_reasoning(self, scenarios: List[Dict], deterministic: bool = True) -> Dict:
        """Enhanced clinical reasoning evaluation with model-specific prompting - DETERMINISTIC"""
        logger.info("Evaluating clinical reasoning...")
        
        results = []
        correct_diagnoses = 0
        total_time = 0
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"Scenario {i}/{len(scenarios)}")
            
            # Create prompt using appropriate template
            prompt = self._format_prompt("clinical_reasoning", scenario=scenario['scenario'])
            
            # Generate response with low temperature for consistent reasoning
            response, gen_time = self.generate_response(
                prompt, 
                max_length=100, 
                temperature=0.1 if deterministic else 0.3,  # Very low temperature for consistent diagnosis
                deterministic=deterministic
            )
            total_time += gen_time
            
            # Check if the most likely diagnosis is mentioned
            diagnosis_mentioned = any(
                diag.lower() in response.lower() 
                for diag in [scenario['most_likely']]
            )
            
            if diagnosis_mentioned:
                correct_diagnoses += 1
            
            # Check for key features recognition
            key_features_mentioned = [
                feature for feature in scenario['key_features']
                if feature.lower() in response.lower()
            ]
            
            # Check for differential diagnosis consideration
            diff_dx_mentioned = [
                dx for dx in scenario['differential_diagnosis']
                if dx.lower() in response.lower()
            ]
            
            result = {
                "scenario": scenario['scenario'],
                "most_likely_diagnosis": scenario['most_likely'],
                "response": response,
                "diagnosis_correct": diagnosis_mentioned,
                "key_features_mentioned": key_features_mentioned,
                "differential_diagnoses_mentioned": diff_dx_mentioned,
                "category": scenario['category'],
                "generation_time": gen_time,
                "prompt_used": prompt
            }
            results.append(result)
        
        diagnostic_accuracy = correct_diagnoses / len(scenarios)
        avg_time = total_time / len(scenarios)
        
        return {
            "diagnostic_accuracy": diagnostic_accuracy,
            "correct_diagnoses": correct_diagnoses,
            "total_scenarios": len(scenarios),
            "average_time_per_scenario": avg_time,
            "detailed_results": results
        }
    
    def _evaluate_relevance(self, response: str, question: str) -> float:
        """Simple relevance score based on question keywords in response"""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        question_words -= stop_words
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(response_words))
        return overlap / len(question_words)
    
    def _extract_choice(self, response: str, options: List[str]) -> str:
        """Enhanced answer extraction with Mistral-specific handling"""
        response_lower = response.lower().strip()
        
        # Mistral-specific extraction
        if self.model_type == "mistral":
            # Look for letter choices at the beginning
            for i, option in enumerate(options):
                letter = chr(65 + i)
                # Common patterns for Mistral responses
                patterns = [
                    f"({letter.lower()})",
                    f"{letter.lower()})",
                    f"{letter.lower()}.",
                    f"{letter.lower()}:",
                    f" {letter.lower()} ",
                    f"answer is ({letter.lower()})",
                    f"answer is {letter.lower()}",
                    f"correct answer is ({letter.lower()})",
                    f"correct answer is {letter.lower()}"
                ]
                
                for pattern in patterns:
                    if pattern in response_lower:
                        return option
            
            # Look for exact option matches
            for option in options:
                if option.lower() in response_lower:
                    return option
            
            # Look for key medical terms from options
            best_match = ""
            max_score = 0
            for option in options:
                option_words = option.lower().split()
                medical_words = [word for word in option_words if len(word) > 4]  # Focus on longer medical terms
                score = sum(1 for word in medical_words if word in response_lower)
                if score > max_score:
                    max_score = score
                    best_match = option
            
            if max_score > 0:
                return best_match
        
        # For dialogue models, look for simple patterns
        elif self.model_type in ["dialogue", "blenderbot"]:
            # Look for option text directly
            for option in options:
                if option.lower() in response_lower:
                    return option
            
            # Look for letters/numbers
            for i, option in enumerate(options):
                if f"{chr(65+i).lower()}" in response_lower[:10]:
                    return option
        
        else:
            # Standard extraction for other models
            # Look for letter choices (A, B, C, D) at the beginning
            for i, option in enumerate(options):
                letter = chr(65 + i)
                patterns = [
                    f"({letter.lower()})",
                    f"{letter.lower()})",
                    f"{letter.lower()}.",
                    f"{letter.lower()}:",
                    f" {letter.lower()} "
                ]
                
                for pattern in patterns:
                    if pattern in response_lower[:30]:
                        return option
            
            # Look for exact matches with options
            for option in options:
                if option.lower() in response_lower:
                    return option
            
            # Look for key words from options with scoring
            best_match = ""
            max_score = 0
            for option in options:
                option_words = option.lower().split()
                score = sum(1 for word in option_words if word in response_lower and len(word) > 3)
                if score > max_score:
                    max_score = score
                    best_match = option
            
            if max_score > 0:
                return best_match
        
        return "No clear answer"
    
    def _evaluate_keywords(self, response: str, expected_keywords: List[str]) -> Tuple[float, List[str]]:
        """Enhanced keyword evaluation with partial matching"""
        response_lower = response.lower()
        found_keywords = []
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in response_lower:
                found_keywords.append(keyword)
            else:
                # Check for partial matches or related words
                keyword_words = keyword_lower.split()
                if any(word in response_lower for word in keyword_words if len(word) > 3):
                    found_keywords.append(keyword)
        
        score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        return score, found_keywords
    
    def _evaluate_semantic_similarity(self, response: str, reference: str) -> float:
        """Evaluate semantic similarity using sentence transformers"""
        if not self.sentence_model:
            return 0
        
        try:
            embeddings = self.sentence_model.encode([response, reference])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0
    
    def generate_report(self, evaluation_results: Dict, output_dir: str):
        """Generate comprehensive evaluation report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"evaluation_report_{self.model_name}_{timestamp}.json")
        
        # Add metadata
        report_data = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "prompt_format": self.prompt_format,
            "is_encoder_decoder": self.is_encoder_decoder,
            "evaluation_timestamp": timestamp,
            "evaluation_results": evaluation_results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Report generated: {report_file}")
        return report_file

def compare_models(model_results: List[Dict], output_dir: str):
    """Create comprehensive model comparison"""
    
    # Create comparison dataframe
    comparison_data = []
    
    for result in model_results:
        model_name = result['model_name']
        eval_results = result['evaluation_results']
        
        row = {
            'Model': model_name,
            'Model_Type': result.get('model_type', 'Unknown'),
            'Is_Encoder_Decoder': result.get('is_encoder_decoder', False),
        }
        
        # Multiple choice metrics
        if 'multiple_choice' in eval_results:
            mc = eval_results['multiple_choice']
            row.update({
                'MC_Accuracy': mc['overall_accuracy'],
                'MC_Avg_Time': mc['average_time_per_question'],
                'MC_Correct': mc['correct_answers'],
                'MC_Total': mc['total_questions']
            })
            
            # Category performance
            for cat, perf in mc['category_performance'].items():
                row[f'MC_{cat.replace(" ", "_")}_Acc'] = perf['accuracy']
        
        # Open ended metrics
        if 'open_ended' in eval_results:
            oe = eval_results['open_ended']
            row.update({
                'OE_Keyword_Score': oe['average_keyword_score'],
                'OE_Avg_Time': oe['average_time_per_question'],
                'OE_Avg_Length': oe['average_response_length']
            })
        
        # Clinical reasoning metrics
        if 'clinical_reasoning' in eval_results:
            cr = eval_results['clinical_reasoning']
            row.update({
                'CR_Accuracy': cr['diagnostic_accuracy'],
                'CR_Avg_Time': cr['average_time_per_scenario'],
                'CR_Correct': cr['correct_diagnoses'],
                'CR_Total': cr['total_scenarios']
            })
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    # Create visualizations
    _create_comparison_plots(df, output_dir, timestamp)
    
    # Generate enhanced markdown report
    markdown_file = _generate_enhanced_markdown_report(df, model_results, output_dir, timestamp)
    
    return df, markdown_file

def _create_comparison_plots(df: pd.DataFrame, output_dir: str, timestamp: str):
    """Create comprehensive comparison plots"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Medical LLM Model Comparison Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Overall Accuracy Comparison
    accuracy_cols = ['MC_Accuracy', 'OE_Keyword_Score', 'CR_Accuracy']
    available_cols = [col for col in accuracy_cols if col in df.columns]
    
    if available_cols:
        df_plot = df[['Model'] + available_cols].set_index('Model')
        df_plot.plot(kind='bar', ax=axes[0,0], width=0.8, colormap='viridis')
        axes[0,0].set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Accuracy/Score', fontsize=12)
        axes[0,0].legend(['Multiple Choice', 'Open Ended (Keywords)', 'Clinical Reasoning'], 
                        bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Response Time Comparison
    time_cols = ['MC_Avg_Time', 'OE_Avg_Time', 'CR_Avg_Time']
    available_time_cols = [col for col in time_cols if col in df.columns]
    
    if available_time_cols:
        df_time = df[['Model'] + available_time_cols].set_index('Model')
        df_time.plot(kind='bar', ax=axes[0,1], width=0.8, color=['skyblue', 'lightgreen', 'coral'])
        axes[0,1].set_title('Average Response Time Comparison', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Time (seconds)', fontsize=12)
        axes[0,1].legend(['Multiple Choice', 'Open Ended', 'Clinical Reasoning'], 
                        bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Model Type Distribution
    if 'Model_Type' in df.columns:
        type_counts = df['Model_Type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        wedges, texts, autotexts = axes[0,2].pie(type_counts.values, labels=type_counts.index, 
                                                autopct='%1.1f%%', colors=colors)
        axes[0,2].set_title('Model Type Distribution', fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # 4. Category-wise Performance Heatmap
    if any(col.startswith('MC_') and col.endswith('_Acc') for col in df.columns):
        category_cols = [col for col in df.columns if col.startswith('MC_') and col.endswith('_Acc')]
        if category_cols:
            category_data = df[['Model'] + category_cols].set_index('Model')
            category_data.columns = [col.replace('MC_', '').replace('_Acc', '').replace('_', ' ') 
                                   for col in category_data.columns]
            
            sns.heatmap(category_data.T, annot=True, cmap='YlOrRd', ax=axes[1,0], 
                       cbar_kws={'label': 'Accuracy'}, fmt='.2f')
            axes[1,0].set_title('Category-wise Performance Heatmap', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Models', fontsize=12)
            axes[1,0].set_ylabel('Medical Categories', fontsize=12)
    
    # 5. Response Length vs Quality
    if 'OE_Avg_Length' in df.columns and 'OE_Keyword_Score' in df.columns:
        scatter = axes[1,1].scatter(df['OE_Avg_Length'], df['OE_Keyword_Score'], 
                                  s=150, alpha=0.7, c=range(len(df)), cmap='plasma')
        axes[1,1].set_xlabel('Average Response Length (words)', fontsize=12)
        axes[1,1].set_ylabel('Keyword Score', fontsize=12)
        axes[1,1].set_title('Response Length vs Quality', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(df['Model']):
            axes[1,1].annotate(model, (df['OE_Avg_Length'].iloc[i], df['OE_Keyword_Score'].iloc[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=10, 
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 6. Performance vs Speed Trade-off
    if 'MC_Accuracy' in df.columns and 'MC_Avg_Time' in df.columns:
        # Create bubble chart where bubble size represents total questions answered correctly
        bubble_sizes = df['MC_Correct'] * 20 if 'MC_Correct' in df.columns else [100] * len(df)
        
        scatter = axes[1,2].scatter(df['MC_Avg_Time'], df['MC_Accuracy'], 
                                  s=bubble_sizes, alpha=0.6, c=range(len(df)), cmap='viridis')
        axes[1,2].set_xlabel('Average Response Time (seconds)', fontsize=12)
        axes[1,2].set_ylabel('Multiple Choice Accuracy', fontsize=12)
        axes[1,2].set_title('Performance vs Speed Trade-off', fontsize=14, fontweight='bold')
        axes[1,2].grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(df['Model']):
            axes[1,2].annotate(model, (df['MC_Avg_Time'].iloc[i], df['MC_Accuracy'].iloc[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Add ideal region
        if len(df) > 1:
            best_acc = df['MC_Accuracy'].max()
            best_time = df['MC_Avg_Time'].min()
            axes[1,2].axhline(y=best_acc * 0.9, color='green', linestyle='--', alpha=0.5, label='High Performance')
            axes[1,2].axvline(x=best_time * 1.5, color='blue', linestyle='--', alpha=0.5, label='Fast Response')
            axes[1,2].legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f"model_comparison_plots_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Comparison plots saved: {plot_file}")

def _generate_enhanced_markdown_report(df: pd.DataFrame, model_results: List[Dict], output_dir: str, timestamp: str) -> str:
    """Generate comprehensive markdown comparison report with detailed analysis"""
    
    markdown_content = f"""# 🏥 Medical LLM Model Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Evaluation Framework:** Enhanced Medical Benchmark Suite  
**Models Evaluated:** {len(model_results)}

---

## Executive Summary

This comprehensive evaluation report analyzes the performance of fine-tuned medical language models across multiple dimensions including factual accuracy, clinical reasoning, and response quality. The evaluation framework uses model-specific prompt templates optimized for each architecture.

### Key Findings

"""
    
    # Add key findings based on data
    if len(df) > 1:
        best_mc_model = df.loc[df['MC_Accuracy'].idxmax(), 'Model'] if 'MC_Accuracy' in df.columns else "N/A"
        best_mc_acc = df['MC_Accuracy'].max() if 'MC_Accuracy' in df.columns else 0
        
        fastest_model = df.loc[df['MC_Avg_Time'].idxmin(), 'Model'] if 'MC_Avg_Time' in df.columns else "N/A"
        fastest_time = df['MC_Avg_Time'].min() if 'MC_Avg_Time' in df.columns else 0
        
        markdown_content += f"""
- ** Best Overall Accuracy:** {best_mc_model} ({best_mc_acc:.1%})
- ** Fastest Response:** {fastest_model} ({fastest_time:.2f}s avg)
- ** Model Types:** {', '.join(df['Model_Type'].unique())}
- ** Performance Range:** {df['MC_Accuracy'].min():.1%} - {df['MC_Accuracy'].max():.1%}
"""

    markdown_content += """

---

## Models Evaluated

"""
    
    for i, result in enumerate(model_results, 1):
        model_type = result.get('model_type', 'Unknown')
        is_enc_dec = result.get('is_encoder_decoder', False)
        arch_type = "Encoder-Decoder" if is_enc_dec else "Decoder-Only"
        
        # Get performance summary for this model
        eval_results = result.get('evaluation_results', {})
        mc_acc = eval_results.get('multiple_choice', {}).get('overall_accuracy', 0) * 100
        oe_score = eval_results.get('open_ended', {}).get('average_keyword_score', 0) * 100
        cr_acc = eval_results.get('clinical_reasoning', {}).get('diagnostic_accuracy', 0) * 100
        
        markdown_content += f"""
### {i}. **{result['model_name']}**

| Attribute | Value |
|-----------|-------|
| **Model Type** | `{model_type}` |
| **Architecture** | {arch_type} |
| **Base Model** | `{result.get('base_model', 'N/A')}` |
| **Multiple Choice Accuracy** | {mc_acc:.1f}% |
| **Open-Ended Keyword Score** | {oe_score:.1f}% |
| **Clinical Reasoning Accuracy** | {cr_acc:.1f}% |

**Model Path:** `{result['model_path']}`

"""

    markdown_content += """

---

## Performance Analysis

### Overall Performance Metrics

| Model | Architecture | MC Accuracy | OE Keywords | CR Accuracy | Avg Time | Performance Score* |
|-------|-------------|-------------|-------------|-------------|----------|-------------------|
"""
    
    for _, row in df.iterrows():
        arch = "Enc-Dec" if row.get('Is_Encoder_Decoder', False) else "Dec-Only"
        mc_acc = f"{row.get('MC_Accuracy', 0):.1%}" if 'MC_Accuracy' in row else "N/A"
        oe_score = f"{row.get('OE_Keyword_Score', 0):.1%}" if 'OE_Keyword_Score' in row else "N/A"
        cr_acc = f"{row.get('CR_Accuracy', 0):.1%}" if 'CR_Accuracy' in row else "N/A"
        avg_time = f"{row.get('MC_Avg_Time', 0):.2f}s" if 'MC_Avg_Time' in row else "N/A"
        
        # Calculate composite performance score
        perf_score = 0
        count = 0
        if 'MC_Accuracy' in row and not pd.isna(row['MC_Accuracy']):
            perf_score += row['MC_Accuracy']
            count += 1
        if 'OE_Keyword_Score' in row and not pd.isna(row['OE_Keyword_Score']):
            perf_score += row['OE_Keyword_Score']
            count += 1
        if 'CR_Accuracy' in row and not pd.isna(row['CR_Accuracy']):
            perf_score += row['CR_Accuracy']
            count += 1
        
        composite_score = f"{(perf_score/count):.1%}" if count > 0 else "N/A"
        
        markdown_content += f"| {row['Model']} | {arch} | {mc_acc} | {oe_score} | {cr_acc} | {avg_time} | {composite_score} |\n"
    
    markdown_content += f"""

*Performance Score = Average of MC Accuracy, OE Keywords, and CR Accuracy

### Category-wise Performance Analysis

"""
    
    # Add category analysis for each model
    for result in model_results:
        model_name = result['model_name']
        eval_results = result.get('evaluation_results', {})
        
        if 'multiple_choice' in eval_results:
            mc_results = eval_results['multiple_choice']
            category_perf = mc_results.get('category_performance', {})
            
            if category_perf:
                markdown_content += f"""
#### {model_name} - Category Performance

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
"""
                for category, perf in category_perf.items():
                    accuracy = perf['accuracy']
                    correct = perf['correct']
                    total = perf['total']
                    markdown_content += f"| {category} | {accuracy:.1%} | {correct}/{total} |\n"

    markdown_content += """

### Clinical Reasoning Analysis

"""
    
    # Add detailed clinical reasoning analysis
    for result in model_results:
        model_name = result['model_name']
        eval_results = result.get('evaluation_results', {})
        
        if 'clinical_reasoning' in eval_results:
            cr_results = eval_results['clinical_reasoning']
            detailed_results = cr_results.get('detailed_results', [])
            
            if detailed_results:
                markdown_content += f"""
#### {model_name} - Clinical Reasoning Performance

| Scenario | Correct Diagnosis | Key Features Identified | Response Quality |
|----------|------------------|-------------------------|------------------|
"""
                for result_item in detailed_results:
                    scenario_short = result_item['scenario'][:60] + "..." if len(result_item['scenario']) > 60 else result_item['scenario']
                    correct = "Y" if result_item['diagnosis_correct'] else "N"
                    features_count = len(result_item['key_features_mentioned'])
                    response_len = len(result_item['response'].split())
                    
                    markdown_content += f"| {scenario_short} | {correct} | {features_count} features | {response_len} words |\n"

    markdown_content += f"""

---

## Detailed Analysis

### Model Architecture Impact

**Decoder-Only Models:** {len(df[df.get('Is_Encoder_Decoder', pd.Series([False])) == False])} models
- Advantages: Better instruction following, natural conversation flow
- Challenges: Context length limitations, potential for repetition

**Encoder-Decoder Models:** {len(df[df.get('Is_Encoder_Decoder', pd.Series([False])) == True])} models  
- Advantages: Better structured output, task-specific optimization
- Challenges: Less natural conversation, prompt format sensitivity

### Prompt Template Optimization

Different model types used optimized prompt templates:

#### Mistral/Mixtral Models
```
<s>[INST] You are a medical expert. Answer this medical question accurately.

{{question}}
{{options}}

Provide the correct answer. [/INST]
```

#### GPT-2 Models  
```
Q: {{question}}
{{options}}
A:
```

#### Llama Models
```
### Question:
{{question}}
{{options}}

### Answer:
```

### Performance Insights

"""

    # Add performance insights based on the data
    if len(df) > 1 and 'MC_Accuracy' in df.columns:
        avg_accuracy = df['MC_Accuracy'].mean()
        std_accuracy = df['MC_Accuracy'].std()
        
        markdown_content += f"""
1. **Average Performance:** {avg_accuracy:.1%} ± {std_accuracy:.1%}
2. **Performance Consistency:** {'High' if std_accuracy < 0.1 else 'Moderate' if std_accuracy < 0.2 else 'Variable'}
3. **Speed vs Accuracy Trade-off:** Models with higher accuracy tend to have {'longer' if df['MC_Accuracy'].corr(df.get('MC_Avg_Time', pd.Series([0]))) > 0.3 else 'similar'} response times
"""

    markdown_content += f"""

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
- **Multiple Choice Questions:** {len(MedicalBenchmarks.get_clinical_questions())} questions across medical specialties
- **Open-ended Questions:** {len(MedicalBenchmarks.get_pharmacology_questions())} pharmacology and mechanism questions
- **Clinical Reasoning:** {len(MedicalBenchmarks.get_diagnostic_scenarios())} diagnostic scenarios

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

- **Detailed JSON Reports:** `evaluation_report_{{model_name}}_{{timestamp}}.json`
- **Comparison CSV:** `model_comparison_{{timestamp}}.csv` 
- **Visualization Dashboard:** `model_comparison_plots_{{timestamp}}.png`
- **This Report:** `detailed_medical_evaluation_report_{{timestamp}}.md`

---

*Report generated using Enhanced Medical LLM Evaluation Framework v2.0*  
*Evaluation completed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # Save enhanced markdown report
    markdown_file = os.path.join(output_dir, f"detailed_medical_evaluation_report_{timestamp}.md")
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info(f"Enhanced markdown report saved: {markdown_file}")
    return markdown_file

def run_comprehensive_evaluation(model_path: str, model_type: str = "auto", base_model: str = None, 
                                prompt_format: str = "auto", output_dir: str = "evaluation_results", 
                                deterministic: bool = True):
    """Run comprehensive evaluation suite for a single model with deterministic results"""
    
    logger.info("🏥 Starting Comprehensive Medical LLM Evaluation")
    logger.info("=" * 60)
    
    # Set deterministic behavior
    if deterministic:
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("🎯 DETERMINISTIC MODE: Results will be reproducible")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, model_type, base_model, prompt_format)
    
    # Print model info
    logger.info(f"Model Type: {evaluator.model_type}")
    logger.info(f"Architecture: {'Encoder-Decoder' if evaluator.is_encoder_decoder else 'Decoder-Only'}")
    logger.info(f"Prompt Templates: {list(evaluator.prompt_templates.keys())}")
    logger.info(f"Evaluation Mode: {'Deterministic (Reproducible)' if deterministic else 'Stochastic'}")
    
    # Run evaluations with deterministic settings
    evaluation_results = {}
    
    # 1. Multiple Choice Questions - DETERMINISTIC
    logger.info(" Running Multiple Choice Evaluation (Deterministic)...")
    mc_questions = MedicalBenchmarks.get_clinical_questions()
    evaluation_results['multiple_choice'] = evaluator.evaluate_multiple_choice(mc_questions, deterministic=deterministic)
    
    # 2. Open-ended Questions - DETERMINISTIC  
    logger.info(" Running Open-ended Evaluation (Controlled Randomness)...")
    open_questions = MedicalBenchmarks.get_pharmacology_questions()
    evaluation_results['open_ended'] = evaluator.evaluate_open_ended(open_questions, deterministic=deterministic)
    
    # 3. Clinical Reasoning - DETERMINISTIC
    logger.info(" Running Clinical Reasoning Evaluation (Deterministic)...")
    scenarios = MedicalBenchmarks.get_diagnostic_scenarios()
    evaluation_results['clinical_reasoning'] = evaluator.evaluate_clinical_reasoning(scenarios, deterministic=deterministic)
    
    # Add evaluation metadata
    evaluation_results['evaluation_metadata'] = {
        'deterministic_mode': deterministic,
        'random_seed': 42 if deterministic else None,
        'torch_seed': 42 if deterministic else None,
        'evaluation_timestamp': datetime.now().isoformat(),
        'reproducible': deterministic
    }
    
    # Generate report
    logger.info(" Generating Evaluation Report...")
    report_path = evaluator.generate_report(evaluation_results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {evaluator.model_name} ({evaluator.model_type})")
    print(f"Architecture: {'Encoder-Decoder' if evaluator.is_encoder_decoder else 'Decoder-Only'}")
    print(f"Mode: {'DETERMINISTIC (Reproducible)' if deterministic else 'STOCHASTIC'}")
    
    if 'multiple_choice' in evaluation_results:
        mc_acc = evaluation_results['multiple_choice']['overall_accuracy']
        mc_time = evaluation_results['multiple_choice']['average_time_per_question']
        print(f"📝 Multiple Choice Accuracy: {mc_acc:.2%} (Avg: {mc_time:.2f}s per question)")
        
        # Show category breakdown
        category_perf = evaluation_results['multiple_choice']['category_performance']
        for category, perf in category_perf.items():
            print(f"   └─ {category}: {perf['accuracy']:.1%} ({perf['correct']}/{perf['total']})")
    
    if 'open_ended' in evaluation_results:
        keyword_score = evaluation_results['open_ended']['average_keyword_score']
        oe_time = evaluation_results['open_ended']['average_time_per_question']
        semantic_score = evaluation_results['open_ended']['average_semantic_score']
        print(f" Open-ended Keyword Score: {keyword_score:.2%} (Avg: {oe_time:.2f}s per question)")
        if semantic_score > 0:
            print(f"   └─ Semantic Similarity: {semantic_score:.2%}")
    
    if 'clinical_reasoning' in evaluation_results:
        diag_acc = evaluation_results['clinical_reasoning']['diagnostic_accuracy']
        cr_time = evaluation_results['clinical_reasoning']['average_time_per_scenario']
        correct_diag = evaluation_results['clinical_reasoning']['correct_diagnoses']
        total_scenarios = evaluation_results['clinical_reasoning']['total_scenarios']
        print(f" Clinical Reasoning Accuracy: {diag_acc:.2%} ({correct_diag}/{total_scenarios} scenarios)")
        print(f"   └─ Average Time: {cr_time:.2f}s per scenario")
    
    # Calculate and display composite score
    scores = []
    if 'multiple_choice' in evaluation_results:
        scores.append(evaluation_results['multiple_choice']['overall_accuracy'])
    if 'open_ended' in evaluation_results:
        scores.append(evaluation_results['open_ended']['average_keyword_score'])
    if 'clinical_reasoning' in evaluation_results:
        scores.append(evaluation_results['clinical_reasoning']['diagnostic_accuracy'])
    
    if scores:
        composite_score = np.mean(scores)
        print(f"\n🎯 Composite Performance Score: {composite_score:.2%}")
    
    print(f"\n📁 Full report saved to: {report_path}")
    
    if deterministic:
        print(f"🔒 DETERMINISTIC EVALUATION: Results are reproducible with same model and seed")
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Medical LLM Evaluation with Mistral Support")
    parser.add_argument("--model-path", required=True, help="Path to the trained model")
    parser.add_argument("--model-type", choices=["auto", "gpt2", "llama", "mistral", "mixtral", "dialogue", "blenderbot", "t5", "alpaca", "vicuna"], 
                       default="auto", help="Model type")
    parser.add_argument("--base-model", help="Base model name (for PEFT models)")
    parser.add_argument("--prompt-format", choices=["auto", "instruction", "conversation", "qa_simple"], 
                       default="auto", help="Prompt format to use")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--compare-models", nargs="+", help="Paths to multiple models for comparison")
    parser.add_argument("--model-names", nargs="+", help="Custom names for the models")
    parser.add_argument("--model-types", nargs="+", help="Model types for comparison models")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic evaluation (default: True)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for deterministic evaluation")
    
    args = parser.parse_args()
    
    # Set deterministic behavior globally
    if args.deterministic:
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("🎯 Deterministic evaluation enabled - results will be reproducible")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare_models:
        # Multi-model comparison
        model_results = []
        model_paths = [args.model_path] + args.compare_models
        
        # Handle model names
        if args.model_names:
            if len(args.model_names) != len(model_paths):
                logger.error("Number of model names must match number of models")
                return
            model_names = args.model_names
        else:
            model_names = [os.path.basename(path) for path in model_paths]
        
        # Handle model types
        if args.model_types:
            if len(args.model_types) != len(model_paths):
                logger.error("Number of model types must match number of models")
                return
            model_types = args.model_types
        else:
            model_types = [args.model_type] * len(model_paths)
        
        for i, model_path in enumerate(model_paths):
            if not os.path.exists(model_path):
                logger.error(f"Model path not found: {model_path}")
                continue
            
            logger.info(f"\n🔄 Evaluating model {i+1}/{len(model_paths)}: {model_names[i]}")
            
            try:
                results = run_comprehensive_evaluation(
                    model_path=model_path,
                    model_type=model_types[i],
                    base_model=args.base_model,
                    prompt_format=args.prompt_format,
                    output_dir=args.output_dir,
                    deterministic=args.deterministic
                )
                
                model_result = {
                    'model_name': model_names[i],
                    'model_path': model_path,
                    'model_type': model_types[i],
                    'base_model': args.base_model,
                    'prompt_format': args.prompt_format,
                    'is_encoder_decoder': results.get('is_encoder_decoder', False),
                    'evaluation_results': results
                }
                model_results.append(model_result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
                continue
        
        if len(model_results) > 1:
            logger.info("\n📊 Creating comprehensive model comparison...")
            comparison_df, markdown_report = compare_models(model_results, args.output_dir)
            print(f"\n📋 Detailed comparison report saved to: {markdown_report}")
            print(f"📊 Comparison data saved to CSV")
            print(f"📈 Visualization dashboard created")
        
    else:
        # Single model evaluation
        if not os.path.exists(args.model_path):
            logger.error(f"Model path not found: {args.model_path}")
            return
        
        run_comprehensive_evaluation(
            model_path=args.model_path,
            model_type=args.model_type,
            base_model=args.base_model,
            prompt_format=args.prompt_format,
            output_dir=args.output_dir,
            deterministic=args.deterministic
        )

if __name__ == "__main__":
    main()