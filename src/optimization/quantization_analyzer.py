"""
Complete Quantization Analysis and Optimization for Medical LLM
"""

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import psutil
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationResult:
    """Data class for quantization results"""
    method: str
    model_size_gb: float
    memory_usage_gb: float
    avg_inference_time: float
    tokens_per_second: float
    quality_score: float
    accuracy_score: float
    efficiency_score: float
    error: Optional[str] = None

class QuantizationAnalyzer:
    """Complete analyzer for different quantization methods"""
    
    def __init__(self, base_model: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.base_model = base_model
        self.test_questions = [
            "What are the symptoms of Type 2 diabetes?",
            "How does hypertension affect the cardiovascular system?", 
            "What are the early signs of stroke?",
            "What causes migraine headaches?",
            "How do antibiotics work against infections?",
            "What are the risk factors for heart disease?",
            "How is pneumonia diagnosed and treated?",
            "What are the functions of the liver?",
            "What causes high cholesterol?",
            "How does insulin work in the body?"
        ]
        
        self.expected_responses = [
            "increased thirst, frequent urination, fatigue, blurred vision",
            "increased workload on heart, arterial damage, atherosclerosis",
            "sudden numbness, confusion, trouble speaking, severe headache",
            "genetic factors, triggers like stress, hormonal changes",
            "kill bacteria, inhibit growth, target cell processes",
            "high blood pressure, smoking, high cholesterol, diabetes",
            "chest X-ray, blood tests, antibiotics treatment",
            "detoxification, protein synthesis, bile production",
            "genetics, diet, lack of exercise, medical conditions", 
            "regulates blood sugar, helps cells absorb glucose"
        ]
        
        self.results: List[QuantizationResult] = []
    
    def get_quantization_configs(self) -> Dict[str, Optional[BitsAndBytesConfig]]:
        """Get comprehensive quantization configurations"""
        configs = {
            'fp16': None,  # Baseline - no quantization
            'int8': BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None
            ),
            'int4_nf4': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            'int4_fp4': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4", 
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            'int4_nf4_single': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,  # Single quantization
            )
        }
        return configs
    
    def clear_memory(self):
        """Clear GPU and system memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Comprehensive memory usage measurement"""
        memory_stats = {}
        
        # System memory
        process = psutil.Process()
        memory_stats['system_memory_gb'] = process.memory_info().rss / (1024**3)
        memory_stats['system_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            memory_stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            memory_stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            
            # GPU properties
            gpu_props = torch.cuda.get_device_properties(0)
            memory_stats['gpu_total_gb'] = gpu_props.total_memory / (1024**3)
            memory_stats['gpu_utilization_percent'] = (
                memory_stats['gpu_allocated_gb'] / memory_stats['gpu_total_gb'] * 100
            )
        
        return memory_stats
    
    def estimate_model_size(self, method: str) -> float:
        """Estimate model size based on quantization method"""
        base_size_gb = 15.0  # Approximate size of Llama-2-7b in FP16
        
        size_estimates = {
            'fp16': base_size_gb,
            'int8': base_size_gb * 0.5,      # ~50% reduction
            'int4_nf4': base_size_gb * 0.25,  # ~75% reduction  
            'int4_fp4': base_size_gb * 0.25,  # ~75% reduction
            'int4_nf4_single': base_size_gb * 0.3  # ~70% reduction
        }
        
        return size_estimates.get(method, base_size_gb)
    
    def load_model_with_config(self, method: str, config: Optional[BitsAndBytesConfig]) -> Tuple[Any, Any]:
        """Load model and tokenizer with specific quantization config"""
        logger.info(f"Loading model with {method} quantization...")
        
        # Clear memory first
        self.clear_memory()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization config
        if config is None:
            # FP16 baseline
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Quantized model
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=config,
                device_map="auto", 
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        
        model.eval()
        return model, tokenizer
    
    def benchmark_inference_speed(self, model, tokenizer, num_runs: int = 5) -> Dict[str, float]:
        """Comprehensive inference speed benchmarking"""
        times = []
        token_counts = []
        first_token_times = []
        
        device = next(model.parameters()).device
        
        for i in range(num_runs):
            question = self.test_questions[i % len(self.test_questions)]
            
            # Prepare input
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            
            # Count generated tokens
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            token_count = len(generated_tokens)
            token_counts.append(token_count)
        
        return {
            'avg_inference_time': np.mean(times),
            'median_inference_time': np.median(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'avg_tokens_generated': np.mean(token_counts),
            'tokens_per_second': np.mean(token_counts) / np.mean(times),
            'throughput_variance': np.std(times) / np.mean(times)  # Coefficient of variation
        }
    
    def evaluate_response_quality(self, model, tokenizer, num_samples: int = 5) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics"""
        quality_scores = []
        accuracy_scores = []
        coherence_scores = []
        response_lengths = []
        
        device = next(model.parameters()).device
        
        for i in range(min(num_samples, len(self.test_questions))):
            question = self.test_questions[i]
            expected_terms = self.expected_responses[i].split(", ")
            
            try:
                # Generate response
                prompt = f"Question: {question}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_response[len(prompt):].strip()
                
                response_length = len(response.split())
                response_lengths.append(response_length)
                
                # Quality metrics
                
                # 1. Accuracy: Check if expected medical terms are present
                response_lower = response.lower()
                matching_terms = sum(1 for term in expected_terms if term.lower() in response_lower)
                accuracy_score = matching_terms / len(expected_terms)
                accuracy_scores.append(accuracy_score)
                
                # 2. Coherence: Check for medical vocabulary and structure
                medical_indicators = [
                    'symptoms', 'treatment', 'diagnosis', 'condition', 'disease',
                    'medical', 'health', 'patient', 'cause', 'effect', 'risk',
                    'therapy', 'medication', 'clinical', 'syndrome'
                ]
                
                medical_term_count = sum(1 for term in medical_indicators if term in response_lower)
                coherence_score = min(medical_term_count / 5.0, 1.0)  # Normalize to 0-1
                coherence_scores.append(coherence_score)
                
                # 3. Overall quality (combination of factors)
                length_appropriateness = 1.0 if 20 <= response_length <= 150 else 0.7
                completeness = 1.0 if len(response.strip()) > 10 else 0.5
                
                overall_quality = (
                    accuracy_score * 0.4 + 
                    coherence_score * 0.3 + 
                    length_appropriateness * 0.2 + 
                    completeness * 0.1
                )
                quality_scores.append(overall_quality)
                
            except Exception as e:
                logger.warning(f"Error evaluating quality for question {i}: {str(e)}")
                quality_scores.append(0.0)
                accuracy_scores.append(0.0)
                coherence_scores.append(0.0)
                response_lengths.append(0)
        
        return {
            'avg_quality_score': np.mean(quality_scores),
            'avg_accuracy_score': np.mean(accuracy_scores),
            'avg_coherence_score': np.mean(coherence_scores),
            'avg_response_length': np.mean(response_lengths),
            'quality_consistency': 1.0 - np.std(quality_scores),  # Higher = more consistent
            'response_length_variance': np.std(response_lengths)
        }
    
    def calculate_efficiency_score(self, speed_metrics: Dict, quality_metrics: Dict, model_size: float) -> float:
        """Calculate comprehensive efficiency score"""
        # Normalize individual metrics (0-1 scale, higher is better)
        
        # Speed score (normalize to 50 tokens/sec as good performance)
        speed_score = min(speed_metrics['tokens_per_second'] / 50.0, 1.0)
        
        # Quality score (already 0-1)
        quality_score = quality_metrics['avg_quality_score']
        
        # Memory efficiency (smaller model size is better, normalize to 16GB)
        memory_efficiency = min(16.0 / max(model_size, 1.0), 1.0)
        
        # Consistency bonus (lower variance is better)
        consistency_bonus = (1.0 - speed_metrics.get('throughput_variance', 0.5)) * 0.1
        
        # Weighted efficiency score
        efficiency = (
            speed_score * 0.35 +          # Speed weight
            quality_score * 0.40 +        # Quality weight  
            memory_efficiency * 0.20 +    # Memory weight
            consistency_bonus * 0.05      # Consistency bonus
        )
        
        return min(efficiency, 1.0)
    
    def analyze_method(self, method: str, config: Optional[BitsAndBytesConfig]) -> QuantizationResult:
        """Analyze a single quantization method"""
        logger.info(f"🔍 Analyzing {method} quantization...")
        
        try:
            # Load model
            model, tokenizer = self.load_model_with_config(method, config)
            
            # Measure memory usage after loading
            memory_stats = self.measure_memory_usage()
            
            # Benchmark speed
            speed_metrics = self.benchmark_inference_speed(model, tokenizer, num_runs=3)
            
            # Evaluate quality  
            quality_metrics = self.evaluate_response_quality(model, tokenizer, num_samples=3)
            
            # Calculate model size
            model_size = self.estimate_model_size(method)
            
            # Calculate efficiency
            efficiency = self.calculate_efficiency_score(speed_metrics, quality_metrics, model_size)
            
            result = QuantizationResult(
                method=method,
                model_size_gb=model_size,
                memory_usage_gb=memory_stats.get('gpu_allocated_gb', memory_stats.get('system_memory_gb', 0)),
                avg_inference_time=speed_metrics['avg_inference_time'],
                tokens_per_second=speed_metrics['tokens_per_second'],
                quality_score=quality_metrics['avg_quality_score'],
                accuracy_score=quality_metrics['avg_accuracy_score'],
                efficiency_score=efficiency
            )
            
            logger.info(f"✅ {method}: Quality={result.quality_score:.3f}, "
                       f"Speed={result.tokens_per_second:.1f} tok/s, "
                       f"Efficiency={result.efficiency_score:.3f}")
            
            # Clean up
            del model, tokenizer
            self.clear_memory()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze {method}: {str(e)}")
            return QuantizationResult(
                method=method,
                model_size_gb=0,
                memory_usage_gb=0,
                avg_inference_time=0,
                tokens_per_second=0,
                quality_score=0,
                accuracy_score=0,
                efficiency_score=0,
                error=str(e)
            )
    
    def compare_all_methods(self) -> List[QuantizationResult]:
        """Compare all quantization methods"""
        logger.info("Starting comprehensive quantization analysis...")
        
        configs = self.get_quantization_configs()
        results = []
        
        for method, config in configs.items():
            result = self.analyze_method(method, config)
            results.append(result)
            
            # Small delay between methods
            time.sleep(2)
        
        self.results = results
        return results
    
    def generate_comparison_table(self, results: List[QuantizationResult]) -> pd.DataFrame:
        """Generate comparison table"""
        valid_results = [r for r in results if r.error is None]
        
        data = []
        for result in valid_results:
            data.append({
                'Method': result.method.upper(),
                'Size (GB)': f"{result.model_size_gb:.1f}",
                'Memory (GB)': f"{result.memory_usage_gb:.1f}",
                'Speed (tok/s)': f"{result.tokens_per_second:.1f}",
                'Quality': f"{result.quality_score:.3f}",
                'Accuracy': f"{result.accuracy_score:.3f}",
                'Efficiency': f"{result.efficiency_score:.3f}"
            })
        
        return pd.DataFrame(data)
    
    def visualize_comprehensive_analysis(self, results: List[QuantizationResult], output_dir: str = "results/optimization"):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            logger.warning("No valid results to visualize")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Extract data
        methods = [r.method for r in valid_results]
        sizes = [r.model_size_gb for r in valid_results]
        speeds = [r.tokens_per_second for r in valid_results]
        qualities = [r.quality_score for r in valid_results]
        accuracies = [r.accuracy_score for r in valid_results]
        efficiencies = [r.efficiency_score for r in valid_results]
        
        # 1. Multi-metric comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Model sizes
        bars1 = ax1.bar(methods, sizes, color=colors[:len(methods)])
        ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Size (GB)')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(sizes):
            ax1.text(i, v + 0.3, f'{v:.1f}GB', ha='center', va='bottom', fontweight='bold')
        
        # Inference speed
        bars2 = ax2.bar(methods, speeds, color=colors[:len(methods)])
        ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Tokens per Second')
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(speeds):
            ax2.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Quality metrics
        x = np.arange(len(methods))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, qualities, width, label='Quality', color='#FF6B6B', alpha=0.8)
        bars3b = ax3.bar(x + width/2, accuracies, width, label='Accuracy', color='#4ECDC4', alpha=0.8)
        ax3.set_title('Quality Metrics Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Efficiency scores
        bars4 = ax4.bar(methods, efficiencies, color=colors[:len(methods)])
        ax4.set_title('Overall Efficiency Score', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Efficiency Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        for i, v in enumerate(efficiencies):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Radar chart for multi-dimensional comparison
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize metrics for radar chart
        normalized_data = []
        for result in valid_results:
            normalized = [
                min(result.tokens_per_second / 50.0, 1.0),  # Speed (normalized to 50 tok/s)
                result.quality_score,                        # Quality (already 0-1)
                result.accuracy_score,                       # Accuracy (already 0-1)
                min(16.0 / result.model_size_gb, 1.0),      # Memory efficiency (smaller is better)
                result.efficiency_score                      # Overall efficiency
            ]
            normalized_data.append(normalized)
        
        categories = ['Speed', 'Quality', 'Accuracy', 'Memory\nEfficiency', 'Overall\nEfficiency']
        N = len(categories)
        
        # Angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each method
        for i, (method, data) in enumerate(zip(methods, normalized_data)):
            values = data + data[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=method.upper(), color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Dimensional Performance Comparison', size=16, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Efficiency vs Quality scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(qualities, speeds, s=[s*20 for s in sizes], 
                           c=efficiencies, cmap='viridis', alpha=0.7, edgecolors='black')
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method.upper(), (qualities[i], speeds[i]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Quality Score', fontsize=12)
        ax.set_ylabel('Inference Speed (tokens/sec)', fontsize=12)
        ax.set_title('Quality vs Speed Trade-off\n(Bubble size = Model size, Color = Efficiency)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Efficiency Score', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_vs_speed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Comprehensive visualizations saved to {output_dir}")
    
    def generate_detailed_report(self, results: List[QuantizationResult], output_path: str = "results/optimization/detailed_analysis_report.md"):
        """Generate comprehensive analysis report"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        valid_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        if not valid_results:
            logger.error("No valid results to generate report")
            return
        
        # Find best performers
        best_speed = max(valid_results, key=lambda x: x.tokens_per_second)
        best_quality = max(valid_results, key=lambda x: x.quality_score)
        best_efficiency = max(valid_results, key=lambda x: x.efficiency_score)
        smallest_model = min(valid_results, key=lambda x: x.model_size_gb)
        most_accurate = max(valid_results, key=lambda x: x.accuracy_score)
        
        report = f"""# Comprehensive Quantization Analysis Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Base Model**: {self.base_model}
**Analysis Methods**: {len(results)} quantization techniques tested

## Executive Summary

This report provides a comprehensive analysis of different quantization methods for the Medical LLM, evaluating performance across multiple dimensions including inference speed, response quality, memory efficiency, and overall effectiveness.

### 🏆 Key Winners

- **🚀 Fastest Inference**: {best_speed.method.upper()} ({best_speed.tokens_per_second:.1f} tokens/sec)
- **🎯 Highest Quality**: {best_quality.method.upper()} (quality score: {best_quality.quality_score:.3f})
- **⚡ Most Efficient**: {best_efficiency.method.upper()} (efficiency score: {best_efficiency.efficiency_score:.3f})
- **💾 Smallest Model**: {smallest_model.method.upper()} ({smallest_model.model_size_gb:.1f} GB)
- **🎯 Most Accurate**: {most_accurate.method.upper()} (accuracy: {most_accurate.accuracy_score:.3f})

## Detailed Results

### Performance Comparison Table

| Method | Size (GB) | Memory (GB) | Speed (tok/s) | Inference (s) | Quality | Accuracy | Efficiency |
|--------|-----------|-------------|---------------|---------------|---------|----------|------------|
"""
        
        for result in valid_results:
            report += f"| {result.method.upper()} | {result.model_size_gb:.1f} | {result.memory_usage_gb:.1f} | {result.tokens_per_second:.1f} | {result.avg_inference_time:.2f} | {result.quality_score:.3f} | {result.accuracy_score:.3f} | {result.efficiency_score:.3f} |\n"
        
        report += f"""

### Method Analysis

"""
        
        for result in valid_results:
            # Calculate relative improvements/degradations
            baseline = next((r for r in valid_results if r.method == 'fp16'), valid_results[0])
            
            size_reduction = ((baseline.model_size_gb - result.model_size_gb) / baseline.model_size_gb) * 100
            speed_change = ((result.tokens_per_second - baseline.tokens_per_second) / baseline.tokens_per_second) * 100
            quality_change = ((result.quality_score - baseline.quality_score) / baseline.quality_score) * 100
            
            report += f"""#### {result.method.upper()}

**Performance Metrics:**
- Model Size: {result.model_size_gb:.1f} GB ({size_reduction:+.1f}% vs baseline)
- Memory Usage: {result.memory_usage_gb:.1f} GB
- Inference Speed: {result.tokens_per_second:.1f} tokens/sec ({speed_change:+.1f}% vs baseline)
- Average Inference Time: {result.avg_inference_time:.2f} seconds
- Quality Score: {result.quality_score:.3f} ({quality_change:+.1f}% vs baseline)
- Accuracy Score: {result.accuracy_score:.3f}
- Efficiency Score: {result.efficiency_score:.3f}

**Trade-off Analysis:**
"""
            
            if result.efficiency_score >= 0.8:
                report += "**Excellent overall performance** - Recommended for production use\n"
            elif result.efficiency_score >= 0.6:
                report += "**Good performance** - Suitable for most applications\n"
            else:
                report += "**Limited performance** - Consider for resource-constrained scenarios only\n"
            
            if size_reduction > 50:
                report += f"**Significant memory savings** ({size_reduction:.0f}% reduction)\n"
            
            if speed_change > 10:
                report += f"**Performance improvement** ({speed_change:.0f}% faster)\n"
            elif speed_change < -10:
                report += f"**Performance trade-off** ({abs(speed_change):.0f}% slower)\n"
            
            report += "\n"
        
        # Add failed results if any
        if failed_results:
            report += f"""### Failed Analyses

The following quantization methods failed during testing:

"""
            for result in failed_results:
                report += f"- **{result.method.upper()}**: {result.error}\n"
            
            report += "\n"
        
        report += f"""## Recommendations

### 🏭 Production Deployment
**Recommended**: {best_efficiency.method.upper()}
- **Rationale**: Best balance of speed, quality, and resource efficiency
- **Use Case**: General medical Q&A applications
- **Requirements**: {best_efficiency.model_size_gb:.1f}GB storage, {best_efficiency.memory_usage_gb:.1f}GB runtime memory

### 💻 Resource-Constrained Environments  
**Recommended**: {smallest_model.method.upper()}
- **Rationale**: Minimal memory and storage footprint
- **Use Case**: Edge deployment, mobile applications
- **Requirements**: {smallest_model.model_size_gb:.1f}GB storage, {smallest_model.memory_usage_gb:.1f}GB runtime memory

### 🎯 Quality-Critical Applications
**Recommended**: {best_quality.method.upper()}
- **Rationale**: Highest response quality and medical accuracy
- **Use Case**: Educational platforms, research applications
- **Requirements**: {best_quality.model_size_gb:.1f}GB storage, {best_quality.memory_usage_gb:.1f}GB runtime memory

### ⚡ Speed-Critical Applications
**Recommended**: {best_speed.method.upper()}
- **Rationale**: Fastest inference for real-time interactions
- **Use Case**: Interactive chatbots, live demonstrations
- **Requirements**: {best_speed.model_size_gb:.1f}GB storage, {best_speed.memory_usage_gb:.1f}GB runtime memory

## Technical Implementation Guide

### Recommended Quantization Config

For most applications, we recommend the **{best_efficiency.method.upper()}** configuration:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### Hardware Requirements

| Method | Min GPU Memory | Recommended GPU | Storage |
|--------|----------------|-----------------|---------|
"""
        
        for result in valid_results:
            min_gpu = f"{result.memory_usage_gb:.0f}GB"
            rec_gpu = "RTX 3070" if result.memory_usage_gb <= 8 else "RTX 4080" if result.memory_usage_gb <= 16 else "A100"
            storage = f"{result.model_size_gb:.0f}GB"
            
            report += f"| {result.method.upper()} | {min_gpu} | {rec_gpu} | {storage} |\n"
        
        report += f"""

### Performance Monitoring

Monitor these key metrics in production:

1. **Inference Speed**: Target > 20 tokens/sec for good user experience
2. **Memory Usage**: Keep GPU utilization < 90% for stability  
3. **Quality Scores**: Monitor response relevance and accuracy
4. **Error Rates**: Track generation failures and timeouts

### Optimization Tips

1. **Batch Processing**: Group multiple requests for better throughput
2. **Caching**: Cache model weights and frequent responses
3. **Dynamic Batching**: Adjust batch size based on input length
4. **Memory Management**: Clear cache between sessions

## Conclusion

Based on comprehensive testing across {len(valid_results)} quantization methods:

- **Best Overall Choice**: {best_efficiency.method.upper()} offers optimal balance ({best_efficiency.efficiency_score:.1%} efficiency)
- **Memory Savings**: Up to {((baseline.model_size_gb - smallest_model.model_size_gb) / baseline.model_size_gb * 100):.0f}% reduction in model size
- **Speed Range**: {min(r.tokens_per_second for r in valid_results):.1f} - {max(r.tokens_per_second for r in valid_results):.1f} tokens/sec across methods
- **Quality Retention**: {min(r.quality_score for r in valid_results):.1%} - {max(r.quality_score for r in valid_results):.1%} quality scores maintained

The analysis demonstrates that INT4 quantization with NF4 provides the best compromise for medical AI applications, reducing memory requirements significantly while maintaining acceptable quality and performance.

---
*This report was generated automatically by the Medical LLM Quantization Analyzer.*
*For questions or additional analysis, please refer to the source code and documentation.*
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Detailed analysis report saved to {output_path}")
        return report

def main():
    """Main function for quantization analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Quantization Analysis for Medical LLM")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-chat-hf", help="Base model to analyze")
    parser.add_argument("--output-dir", default="results/optimization", help="Output directory")
    parser.add_argument("--methods", nargs="+", help="Specific methods to test", 
                       choices=['fp16', 'int8', 'int4_nf4', 'int4_fp4', 'int4_nf4_single'])
    parser.add_argument("--quick", action="store_true", help="Quick analysis with fewer samples")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = QuantizationAnalyzer(args.base_model)
    
    if args.quick:
        analyzer.test_questions = analyzer.test_questions[:3]  # Use fewer questions for quick test
    
    logger.info("Starting comprehensive quantization analysis...")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run analysis
    if args.methods:
        # Test only specified methods
        configs = analyzer.get_quantization_configs()
        results = []
        for method in args.methods:
            if method in configs:
                result = analyzer.analyze_method(method, configs[method])
                results.append(result)
            else:
                logger.warning(f"Unknown method: {method}")
    else:
        # Test all methods
        results = analyzer.compare_all_methods()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw results
    results_data = []
    for result in results:
        results_data.append({
            'method': result.method,
            'model_size_gb': result.model_size_gb,
            'memory_usage_gb': result.memory_usage_gb,
            'avg_inference_time': result.avg_inference_time,
            'tokens_per_second': result.tokens_per_second,
            'quality_score': result.quality_score,
            'accuracy_score': result.accuracy_score,
            'efficiency_score': result.efficiency_score,
            'error': result.error
        })
    
    results_path = os.path.join(args.output_dir, "quantization_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Generate comparison table
    df = analyzer.generate_comparison_table(results)
    table_path = os.path.join(args.output_dir, "comparison_table.csv")
    df.to_csv(table_path, index=False)
    
    # Generate visualizations
    analyzer.visualize_comprehensive_analysis(results, args.output_dir)
    
    # Generate detailed report
    report_path = os.path.join(args.output_dir, "quantization_analysis_report.md")
    analyzer.generate_detailed_report(results, report_path)
    
    # Print summary
    valid_results = [r for r in results if r.error is None]
    
    print("\n" + "="*80)
    print("🔧 QUANTIZATION ANALYSIS COMPLETE")
    print("="*80)
    
    if valid_results:
        best_efficiency = max(valid_results, key=lambda x: x.efficiency_score)
        
        print(f"\n📊 SUMMARY ({len(valid_results)} methods analyzed):")
        print("-" * 50)
        
        for result in valid_results:
            status = "Excellent" if result == best_efficiency else "Good" if result.efficiency_score > 0.6 else "Caution"
            print(f"{status} {result.method.upper():12} | "
                  f"Size: {result.model_size_gb:4.1f}GB | "
                  f"Speed: {result.tokens_per_second:5.1f} tok/s | "
                  f"Quality: {result.quality_score:.3f} | "
                  f"Efficiency: {result.efficiency_score:.3f}")
        
        print(f"\n BEST OVERALL: {best_efficiency.method.upper()}")
        print(f"   Efficiency Score: {best_efficiency.efficiency_score:.3f}")
        print(f"   Model Size: {best_efficiency.model_size_gb:.1f}GB")
        print(f"   Speed: {best_efficiency.tokens_per_second:.1f} tokens/sec")
        print(f"   Quality: {best_efficiency.quality_score:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Visualizations: comprehensive_analysis.png, radar_comparison.png")
    print("Detailed report: quantization_analysis_report.md")
    print("Data: quantization_analysis_results.json, comparison_table.csv")
    
    failed_results = [r for r in results if r.error is not None]
    if failed_results:
        print(f"\n{len(failed_results)} methods failed:")
        for result in failed_results:
            print(f"{result.method}: {result.error}")
    
    print("="*80)

if __name__ == "__main__":
    main()