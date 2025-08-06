"""
Download and prepare medical datasets for training
"""

import os
import json
import requests
import zipfile
import pandas as pd
from datasets import load_dataset
import argparse
import logging
from typing import Dict, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataDownloader:
    """Download and prepare medical datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_pubmedqa(self):
        """Download PubMedQA dataset"""
        logger.info("Downloading PubMedQA dataset...")
        
        try:
            # Load from HuggingFace datasets
            dataset = load_dataset("pubmed_qa", "pqa_labeled")
            
            # Convert to our format
            formatted_data = []
            for item in dataset['train']:
                formatted_item = {
                    "question": item['question'],
                    "context": " ".join(item['context']['contexts']) if 'context' in item else "",
                    "answer": item['final_decision'],
                    "long_answer": item.get('long_answer', ''),
                    "source": "PubMedQA"
                }
                formatted_data.append(formatted_item)
            
            # Save formatted data
            output_path = self.processed_dir / "pubmedqa_formatted.json"
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            logger.info(f"PubMedQA dataset saved to {output_path}")
            logger.info(f"Total samples: {len(formatted_data)}")
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Failed to download PubMedQA: {str(e)}")
            return self.create_sample_pubmedqa()
    
    def create_sample_pubmedqa(self):
        """Create sample PubMedQA data for testing"""
        logger.info("Creating sample PubMedQA dataset...")
        
        sample_data = [
            {
                "question": "What are the main symptoms of Type 2 diabetes?",
                "context": "Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar (glucose). With type 2 diabetes, your body either resists the effects of insulin or doesn't produce enough insulin to maintain normal glucose levels.",
                "answer": "The main symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
                "long_answer": "Type 2 diabetes symptoms develop slowly and may be subtle initially. The primary symptoms include polydipsia (increased thirst), polyuria (frequent urination), polyphagia (increased hunger), unexplained weight loss despite increased appetite, fatigue and weakness, blurred vision, slow-healing cuts and bruises, frequent infections particularly skin and urinary tract infections, and areas of darkened skin (acanthosis nigricans). Many people with type 2 diabetes have no symptoms initially, which is why regular screening is important for at-risk individuals.",
                "source": "Sample"
            },
            {
                "question": "How does hypertension affect the cardiovascular system?",
                "context": "Hypertension, commonly known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
                "answer": "Hypertension damages the cardiovascular system by increasing workload on the heart, damaging arterial walls, and promoting atherosclerosis.",
                "long_answer": "Hypertension affects the cardiovascular system through multiple mechanisms. It forces the heart to work harder to pump blood against increased resistance, leading to left ventricular hypertrophy and eventually heart failure. The elevated pressure damages the inner lining of arteries (endothelium), making them more susceptible to atherosclerotic plaque formation. This process narrows arteries and increases the risk of blood clots, potentially leading to heart attacks and strokes. Chronic hypertension also stiffens arteries, reducing their elasticity and further increasing cardiac workload. The kidneys, brain, and other organs supplied by damaged vessels may also suffer reduced blood flow and function.",
                "source": "Sample"
            },
            {
                "question": "What are the risk factors for coronary heart disease?",
                "context": "Coronary heart disease (CHD) is a disease in which plaque builds up inside the coronary arteries, which supply oxygen-rich blood to your heart muscle.",
                "answer": "Risk factors include high cholesterol, high blood pressure, smoking, diabetes, obesity, physical inactivity, unhealthy diet, age, gender, and family history.",
                "long_answer": "Coronary heart disease risk factors are classified as modifiable and non-modifiable. Modifiable risk factors include: high LDL cholesterol and low HDL cholesterol, hypertension, cigarette smoking, diabetes mellitus, obesity (particularly abdominal obesity), physical inactivity, unhealthy diet high in saturated fats and low in fruits and vegetables, excessive alcohol consumption, and chronic stress. Non-modifiable risk factors include: advancing age (men ≥45 years, women ≥55 years), male gender, family history of premature CHD, and certain genetic factors. Additional emerging risk factors include elevated C-reactive protein, homocysteine levels, and metabolic syndrome. The more risk factors present, the higher the likelihood of developing CHD.",
                "source": "Sample"
            },
            {
                "question": "What causes migraine headaches?",
                "context": "Migraine is a neurological disorder characterized by recurrent headaches that are moderate to severe.",
                "answer": "Migraines are caused by complex interactions involving genetics, brain chemistry changes, hormonal fluctuations, and various environmental triggers.",
                "long_answer": "Migraine headaches result from complex neurobiological processes involving genetic predisposition, neurotransmitter imbalances, and vascular changes in the brain. The exact mechanism involves activation of the trigeminal nerve system, leading to release of inflammatory substances around blood vessels in the brain. Common triggers include hormonal changes (particularly estrogen fluctuations in women), certain foods (aged cheeses, wine, chocolate, MSG), stress, sleep pattern changes, sensory stimuli (bright lights, loud sounds, strong smells), weather changes, and certain medications. Genetics play a significant role, with family history being a strong predictor. The condition involves dysfunction in brain areas that regulate pain and other sensory processing.",
                "source": "Sample"
            },
            {
                "question": "How do antibiotics work against bacterial infections?",
                "context": "Antibiotics are medications designed to fight bacterial infections in humans and animals.",
                "answer": "Antibiotics work by either killing bacteria or inhibiting their growth through various mechanisms targeting essential bacterial processes.",
                "long_answer": "Antibiotics combat bacterial infections through several distinct mechanisms. Bactericidal antibiotics kill bacteria directly, while bacteriostatic antibiotics inhibit bacterial growth and reproduction. The main mechanisms include: cell wall synthesis inhibition (penicillins, cephalosporins), protein synthesis inhibition (tetracyclines, aminoglycosides), DNA/RNA synthesis interference (fluoroquinolones, rifampin), cell membrane disruption (polymyxins), and metabolic pathway interference (sulfonamides, trimethoprim). Each class targets specific bacterial structures or processes that are essential for bacterial survival but different from human cellular processes, providing selective toxicity. However, bacteria can develop resistance through genetic mutations or acquisition of resistance genes, which is why appropriate antibiotic use and completing prescribed courses is crucial.",
                "source": "Sample"
            }
        ]
        
        # Save sample data
        output_path = self.processed_dir / "sample_medical_qa.json"
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample dataset created at {output_path}")
        logger.info(f"Total samples: {len(sample_data)}")
        
        return sample_data
    
    def download_medmcqa(self):
        """Download MedMCQA dataset (if available)"""
        logger.info("Attempting to download MedMCQA dataset...")
        
        try:
            # Try to load MedMCQA from HuggingFace
            dataset = load_dataset("medmcqa")
            
            # Process and save
            formatted_data = []
            for split in ['train', 'validation']:
                if split in dataset:
                    for item in dataset[split]:
                        formatted_item = {
                            "question": item['question'],
                            "options": [item['opa'], item['opb'], item['opc'], item['opd']],
                            "correct_answer": item['cop'],
                            "subject": item.get('subject_name', 'Medical'),
                            "source": "MedMCQA"
                        }
                        formatted_data.append(formatted_item)
            
            output_path = self.processed_dir / "medmcqa_formatted.json"
            with open(output_path, 'w') as f:
                json.dump(formatted_data[:1000], f, indent=2)  # Limit to 1000 samples
            
            logger.info(f"MedMCQA dataset saved to {output_path}")
            return formatted_data[:1000]
            
        except Exception as e:
            logger.warning(f"Could not download MedMCQA: {str(e)}")
            return self.create_sample_mcqa()
    
    def create_sample_mcqa(self):
        """Create sample medical MCQ data"""
        sample_mcqa = [
            {
                "question": "Which of the following is the most common cause of hypertension?",
                "options": [
                    "Primary (essential) hypertension",
                    "Kidney disease",
                    "Thyroid disorders", 
                    "Medication side effects"
                ],
                "correct_answer": 0,
                "subject": "Cardiology",
                "source": "Sample"
            },
            {
                "question": "What is the normal range for fasting blood glucose?",
                "options": [
                    "70-100 mg/dL",
                    "100-140 mg/dL",
                    "140-180 mg/dL",
                    "180-220 mg/dL"
                ],
                "correct_answer": 0,
                "subject": "Endocrinology",
                "source": "Sample"
            }
        ]
        
        output_path = self.processed_dir / "sample_medical_mcqa.json"
        with open(output_path, 'w') as f:
            json.dump(sample_mcqa, f, indent=2)
        
        logger.info(f"Sample MCQA dataset created at {output_path}")
        return sample_mcqa
    
    def create_evaluation_dataset(self):
        """Create evaluation dataset"""
        logger.info("Creating evaluation dataset...")
        
        eval_data = [
            {
                "question": "What are the early warning signs of a heart attack?",
                "answer": "Early warning signs include chest pain or discomfort, shortness of breath, pain in arms, back, neck, jaw, or stomach, cold sweat, nausea, and lightheadedness.",
                "context": "Cardiovascular emergency",
                "category": "Emergency Medicine"
            },
            {
                "question": "How is Type 1 diabetes different from Type 2 diabetes?",
                "answer": "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin, typically diagnosed in children and young adults. Type 2 diabetes involves insulin resistance and relative insulin deficiency, usually develops in adults, and is often associated with obesity.",
                "context": "Diabetes management",
                "category": "Endocrinology"
            },
            {
                "question": "What are the main functions of the liver?",
                "answer": "The liver performs over 500 functions including detoxification, protein synthesis, bile production, glucose storage, fat metabolism, blood clotting factor production, and immune system support.",
                "context": "Hepatology",
                "category": "Gastroenterology"
            }
        ]
        
        eval_path = self.data_dir / "evaluation" / "medical_eval_set.json"
        eval_path.parent.mkdir(exist_ok=True)
        
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        logger.info(f"Evaluation dataset created at {eval_path}")
        return eval_data
    
    def download_all(self):
        """Download all available datasets"""
        logger.info("Starting data download process...")
        
        datasets = {}
        
        # Download PubMedQA
        datasets['pubmedqa'] = self.download_pubmedqa()
        
        # Download MedMCQA
        datasets['medmcqa'] = self.download_medmcqa()
        
        # Create evaluation set
        datasets['evaluation'] = self.create_evaluation_dataset()
        
        # Create summary
        summary = {
            "datasets": {
                name: len(data) for name, data in datasets.items()
            },
            "total_samples": sum(len(data) for data in datasets.values()),
            "download_date": pd.Timestamp.now().isoformat()
        }
        
        summary_path = self.data_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Data download completed!")
        logger.info(f"Summary saved to {summary_path}")
        
        for name, count in summary["datasets"].items():
            logger.info(f"  {name}: {count} samples")
        
        return datasets

def main():
    parser = argparse.ArgumentParser(description="Download medical datasets")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--dataset", choices=["pubmedqa", "medmcqa", "all"], default="all", help="Dataset to download")
    parser.add_argument("--create-sample", action="store_true", help="Create sample dataset for testing")
    
    args = parser.parse_args()
    
    downloader = MedicalDataDownloader(args.data_dir)
    
    if args.create_sample:
        # Create sample data immediately
        logger.info("Creating sample medical dataset...")
        sample_data = downloader.create_sample_pubmedqa()
        
        # Also create evaluation data
        eval_data = downloader.create_evaluation_dataset()
        
        print("\n" + "="*50)
        print("📚 SAMPLE DATASET CREATION COMPLETE")
        print("="*50)
        print(f"📖 Training samples: {len(sample_data)}")
        print(f"🧪 Evaluation samples: {len(eval_data)}")
        print(f"📁 Data directory: {args.data_dir}")
        print("\n💡 You can now train with:")
        print("   python src/training/train_medical_llm.py --create-sample")
        print("="*50)
        return
    
    if args.dataset == "pubmedqa":
        downloader.download_pubmedqa()
    elif args.dataset == "medmcqa":
        downloader.download_medmcqa()
    else:
        downloader.download_all()

if __name__ == "__main__":
    main()