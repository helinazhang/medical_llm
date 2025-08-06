"""
Model monitoring and drift detection for Medical LLM
"""

import torch
import numpy as np
import json
import time
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import threading
import queue
import os 

# Try to import GPUtil, but handle gracefully if not available
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    print("GPUtil not available. GPU monitoring will use PyTorch only.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self, model_path: str, monitoring_window: int = 100):
        self.model_path = model_path
        self.monitoring_window = monitoring_window
        
        # Performance metrics storage
        self.metrics_history = defaultdict(deque)
        self.response_times = deque(maxlen=monitoring_window)
        self.memory_usage = deque(maxlen=monitoring_window)
        self.gpu_usage = deque(maxlen=monitoring_window)
        self.error_count = 0
        self.total_requests = 0
        
        # Drift detection
        self.baseline_metrics = None
        self.drift_threshold = 0.05  # 5% performance drop triggers alert
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        self._metrics_queue = queue.Queue()
        
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Model monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                gpu_info = None
                if torch.cuda.is_available():
                    try:
                        if GPU_UTIL_AVAILABLE:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu = gpus[0]
                                gpu_info = {
                                    'utilization': gpu.load * 100,
                                    'memory_used': gpu.memoryUsed,
                                    'memory_total': gpu.memoryTotal,
                                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                    'temperature': gpu.temperature
                                }
                        else:
                            # Fallback to PyTorch GPU monitoring
                            gpu_info = {
                                'memory_used': torch.cuda.memory_allocated() / (1024**2),  # MB
                                'memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**2),  # MB
                                'memory_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100,
                                'utilization': 'N/A',
                                'temperature': 'N/A'
                            }
                    except Exception as e:
                        logger.warning(f"GPU monitoring failed: {e}")
                        gpu_info = None
                
                # Store metrics
                timestamp = datetime.now()
                metrics = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_used_gb': memory_info.used / (1024**3),
                    'gpu_info': gpu_info
                }
                
                self._metrics_queue.put(metrics)
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def log_request(self, response_time: float, success: bool = True, error: str = None):
        """Log a model request"""
        self.total_requests += 1
        
        if success:
            self.response_times.append(response_time)
        else:
            self.error_count += 1
            if error:
                logger.warning(f"Request failed: {error}")
        
        # Collect current system metrics
        try:
            while not self._metrics_queue.empty():
                metrics = self._metrics_queue.get_nowait()
                self.memory_usage.append(metrics['memory_percent'])
                if metrics['gpu_info']:
                    self.gpu_usage.append(metrics['gpu_info']['memory_percent'])
        except queue.Empty:
            pass
    
    def evaluate_drift(self, reference_data: List[Dict], current_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model drift between reference and current data"""
        logger.info("Evaluating model drift...")
        
        if not reference_data or not current_data:
            return {"error": "Insufficient data for drift evaluation"}
        
        # Simple drift detection based on response length and time
        ref_lengths = []
        curr_lengths = []
        ref_times = []
        curr_times = []
        
        # For reference data (baseline)
        for item in reference_data:
            if 'response_length' in item:
                ref_lengths.append(item['response_length'])
            if 'response_time' in item:
                ref_times.append(item['response_time'])
        
        # For current data
        for item in current_data:
            if 'response_length' in item:
                curr_lengths.append(item['response_length'])
            if 'response_time' in item:
                curr_times.append(item['response_time'])
        
        drift_metrics = {}
        
        # Calculate drift in response length
        if ref_lengths and curr_lengths:
            ref_length_mean = np.mean(ref_lengths)
            curr_length_mean = np.mean(curr_lengths)
            length_drift = abs(curr_length_mean - ref_length_mean) / ref_length_mean
            drift_metrics['response_length_drift'] = length_drift
        
        # Calculate drift in response time
        if ref_times and curr_times:
            ref_time_mean = np.mean(ref_times)
            curr_time_mean = np.mean(curr_times)
            time_drift = abs(curr_time_mean - ref_time_mean) / ref_time_mean
            drift_metrics['response_time_drift'] = time_drift
        
        # Overall drift score
        if drift_metrics:
            overall_drift = np.mean(list(drift_metrics.values()))
            drift_metrics['overall_drift'] = overall_drift
            
            # Check if drift exceeds threshold
            if overall_drift > self.drift_threshold:
                drift_metrics['drift_detected'] = True
                logger.warning(f"Model drift detected! Drift score: {overall_drift:.3f}")
            else:
                drift_metrics['drift_detected'] = False
        
        return drift_metrics
    
    def trigger_retraining_alert(self, drift_score: float = None):
        """Trigger alert for model retraining"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'retraining_required',
            'drift_score': drift_score,
            'total_requests': self.total_requests,
            'error_rate': self.error_count / max(self.total_requests, 1),
            'avg_response_time': np.mean(self.response_times) if self.response_times else None
        }
        
        logger.critical("🚨 RETRAINING ALERT TRIGGERED!")
        logger.critical(f"Alert data: {json.dumps(alert_data, indent=2)}")
        
        # Save alert to file
        alerts_dir = Path("results/alerts")
        alerts_dir.mkdir(parents=True, exist_ok=True)
        
        alert_file = alerts_dir / f"alert_{int(time.time())}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        return alert_data
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        summary = {
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_requests, 1),
            'avg_response_time': np.mean(self.response_times) if self.response_times else None,
            'median_response_time': np.median(self.response_times) if self.response_times else None,
            'p95_response_time': np.percentile(self.response_times, 95) if self.response_times else None,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else None,
            'avg_gpu_usage': np.mean(self.gpu_usage) if self.gpu_usage else None,
            'monitoring_window': self.monitoring_window,
            'monitoring_active': self._monitoring
        }
        
        return summary
    
    def generate_monitoring_report(self, output_path: str = "results/monitoring_report.md"):
        """Generate monitoring report"""
        summary = self.get_performance_summary()
        
        report = f"""# Model Monitoring Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {self.model_path}

## Performance Summary

### Request Statistics
- **Total Requests**: {summary['total_requests']}
- **Error Count**: {summary['error_count']}
- **Error Rate**: {summary['error_rate']:.3%}

### Response Time Metrics
- **Average Response Time**: {summary['avg_response_time']:.3f}s
- **Median Response Time**: {summary['median_response_time']:.3f}s
- **95th Percentile**: {summary['p95_response_time']:.3f}s

### Resource Usage
- **Average Memory Usage**: {summary['avg_memory_usage']:.1f}%
- **Average GPU Usage**: {summary['avg_gpu_usage']:.1f}%

## Monitoring Status
- **Monitoring Active**: {summary['monitoring_active']}
- **Monitoring Window**: {summary['monitoring_window']} requests

## Recommendations

{"**High Error Rate**: Consider investigating model stability" if summary['error_rate'] > 0.05 else "✅ **Error Rate**: Within acceptable range"}

{"**Slow Response Time**: Consider model optimization" if summary.get('avg_response_time', 0) > 3.0 else "✅ **Response Time**: Within acceptable range"}

{"**High Memory Usage**: Monitor for memory leaks" if summary.get('avg_memory_usage', 0) > 80 else "✅ **Memory Usage**: Within acceptable range"}

---
*This report was generated automatically by the Medical LLM monitoring suite.*
"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Monitoring report saved to {output_path}")
        return report
    
    def visualize_metrics(self, output_dir: str = "results/monitoring"):
        """Create visualizations of monitoring metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Response time trend
        if self.response_times:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(list(self.response_times), marker='o', linewidth=1, markersize=3)
            ax.set_title('Response Time Trend', fontsize=14, fontweight='bold')
            ax.set_xlabel('Request Number')
            ax.set_ylabel('Response Time (seconds)')
            ax.grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(self.response_times)
            ax.axhline(y=avg_time, color='red', linestyle='--', 
                      label=f'Average: {avg_time:.3f}s')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'response_time_trend.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Memory usage trend
        if self.memory_usage:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # CPU Memory
            ax1.plot(list(self.memory_usage), color='blue', linewidth=2)
            ax1.set_title('Memory Usage Trend', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Memory Usage (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # GPU Memory
            if self.gpu_usage:
                ax2.plot(list(self.gpu_usage), color='green', linewidth=2)
                ax2.set_title('GPU Memory Usage Trend', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Time Points')
                ax2.set_ylabel('GPU Memory Usage (%)')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'memory_usage_trend.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Monitoring visualizations saved to {output_dir}")

def main():
    """Demo of model monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Monitoring Demo")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ModelMonitor(args.model_path)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some requests
    logger.info(f"Simulating requests for {args.duration} seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            # Simulate request
            request_start = time.time()
            time.sleep(np.random.uniform(0.5, 2.0))  # Simulate processing
            request_time = time.time() - request_start
            
            # Random success/failure
            success = np.random.random() > 0.05  # 95% success rate
            
            monitor.log_request(request_time, success)
            
            time.sleep(1)  # Wait between requests
    
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Generate report
        summary = monitor.get_performance_summary()
        print("\n" + "="*50)
        print("📊 MONITORING SUMMARY")
        print("="*50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        # Generate visualizations
        monitor.visualize_metrics()
        
        # Generate report
        monitor.generate_monitoring_report()
        
        print("Monitoring demo completed!")

if __name__ == "__main__":
    main()