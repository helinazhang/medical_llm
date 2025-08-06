"""
Medical LLM Demo - Main class for medical question answering
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, LoraConfig
import warnings
import logging
from typing import Optional, Dict, Any
import time
import json
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)

class MedicalLLMDemo:
    """
    Medical AI Assistant using fine-tuned Llama model with LoRA adaptation
    """
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the Medical LLM Demo
        
        Args:
            model_path: Path to the fine-tuned model/LoRA adapters
            base_model: Base model identifier from HuggingFace
        """
        self.model_path = model_path
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Medical prompt template
        self.medical_prompt_template = """<s>[INST] You are a knowledgeable medical AI assistant. Provide accurate, evidence-based information while emphasizing that this is for educational purposes only and not a substitute for professional medical advice.

                                        Context: {context}
                                        Question: {question}

                                        Please provide a comprehensive but concise answer. [/INST]"""
        
        self.load_model()
    
    def get_quantization_config(self) -> BitsAndBytesConfig:
        """Get 4-bit quantization configuration"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self):
        """Load the model and tokenizer with quantization"""
        try:
            logging.info(f"Loading model from {self.base_model}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Quantization config
            bnb_config = self.get_quantization_config()
            
            # Load base model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Load LoRA adapters if they exist
            if os.path.exists(self.model_path):
                logging.info(f"Loading LoRA adapters from {self.model_path}")
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
            else:
                logging.warning(f"LoRA adapters not found at {self.model_path}. Using base model.")
            
            self.model.eval()
            logging.info("Model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_medical_response(
        self, 
        question: str, 
        context: str = "",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Generate medical response to a question
        
        Args:
            question: Medical question
            context: Additional context (patient info, etc.)
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated medical response
        """
        try:
            # Format prompt
            prompt = self.medical_prompt_template.format(
                context=context if context else "General medical inquiry",
                question=question
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after [/INST])
            if "[/INST]" in full_response:
                response = full_response.split("[/INST]")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            # Add disclaimer
            disclaimer = "\n\n **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for medical decisions."
            
            logging.info(f"Generated response in {generation_time:.2f}s")
            
            return response + disclaimer
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        info = {
            "base_model": self.base_model,
            "model_path": self.model_path,
            "device": str(self.device),
            "quantization": "4-bit NF4",
            "has_lora": os.path.exists(self.model_path)
        }
        
        if torch.cuda.is_available():
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
        
        return info
    
    def benchmark_model(self, test_questions: list, context: str = "") -> Dict[str, float]:
        """
        Benchmark model performance
        
        Args:
            test_questions: List of test questions
            context: Context for questions
            
        Returns:
            Performance metrics
        """
        times = []
        token_counts = []
        
        for question in test_questions:
            start_time = time.time()
            response = self.generate_medical_response(question, context)
            end_time = time.time()
            
            times.append(end_time - start_time)
            token_counts.append(len(self.tokenizer.encode(response)))
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        return {
            "average_response_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "total_questions": len(test_questions)
        }


def main():
    """Demo usage of MedicalLLMDemo"""
    # Example usage
    model_path = "./models/medical-llm-final"  # Adjust path as needed
    
    try:
        demo = MedicalLLMDemo(model_path)
        
        # Print model info
        info = demo.get_model_info()
        print("Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test questions
        test_questions = [
            "What are the symptoms of Type 2 diabetes?",
            "How does hypertension affect the cardiovascular system?",
            "What are the risk factors for heart disease?"
        ]
        
        print("\n" + "="*50)
        print("MEDICAL AI ASSISTANT DEMO")
        print("="*50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 50)
            
            response = demo.generate_medical_response(
                question=question,
                context="General medical inquiry"
            )
            
            print(f"Response: {response}")
            print("-" * 50)
        
        # Benchmark
        print("\nRunning benchmark...")
        benchmark_results = demo.benchmark_model(test_questions)
        
        print("\nBenchmark Results:")
        for key, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure you have the required model files and dependencies installed.")


if __name__ == "__main__":
    main()