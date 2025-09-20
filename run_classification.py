#!/usr/bin/env python3
"""
Main runner script for AI vs Human Music Classification
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add ml_pipeline to path
sys.path.append(str(Path(__file__).parent / "ml_pipeline"))

from ml_pipeline.main_pipeline import MusicClassificationPipeline
from ml_pipeline.model_comparison import ModelComparator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="AI vs Human Music Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single model training
  python run_classification.py --mode single --epochs 10

  # Compare multiple models
  python run_classification.py --mode compare --epochs 5

  # Quick test with fewer samples
  python run_classification.py --mode single --samples-per-class 10 --epochs 3
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["single", "compare"], default="single",
                       help="Mode: 'single' for single model training, 'compare' for model comparison")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="ashhadahsan/ai-vs-human-music-dataset",
                       help="HuggingFace dataset name")
    parser.add_argument("--samples-per-class", type=int, default=25,
                       help="Number of samples per class")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base",
                       help="Base Wav2Vec2 model name (for single mode)")
    parser.add_argument("--freeze-feature-extractor", action="store_true",
                       help="Freeze the feature extractor during training")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                       help="Learning rate")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    
    # Comparison mode arguments
    parser.add_argument("--compare-models", nargs='+', 
                       choices=['wav2vec2-base', 'wav2vec2-base-frozen', 'wav2vec2-large', 'wav2vec2-large-frozen'],
                       help="Specific models to compare (for compare mode)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("AI vs HUMAN MUSIC CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    try:
        if args.mode == "single":
            # Single model training
            print(f"Training single model: {args.model}")
            
            pipeline = MusicClassificationPipeline(
                dataset_name=args.dataset,
                samples_per_class=args.samples_per_class,
                model_name=args.model,
                output_dir=args.output_dir,
                freeze_feature_extractor=args.freeze_feature_extractor
            )
            
            # Custom training arguments
            training_args = {
                'num_train_epochs': args.epochs,
                'per_device_train_batch_size': args.batch_size,
                'per_device_eval_batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'warmup_steps': 100,
                'weight_decay': 0.01,
                'logging_steps': 10,
                'eval_steps': 50,
                'save_steps': 100,
                'evaluation_strategy': "steps",
                'save_strategy': "steps",
                'load_best_model_at_end': True,
                'metric_for_best_model': "eval_accuracy",
                'greater_is_better': True,
                'save_total_limit': 3,
                'dataloader_num_workers': 2,
                'fp16': False  # Disable for compatibility
            }
            
            # Run pipeline
            results = pipeline.run_complete_pipeline()
            
            # Print results
            print("\n" + "="*50)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"Model: {args.model}")
            print(f"Final accuracy: {results['evaluation_results']['accuracy']:.4f}")
            print(f"Results directory: {results['output_directory']}")
            
            # Print key metrics
            eval_results = results['evaluation_results']
            if 'classification_report' in eval_results:
                report = eval_results['classification_report']
                print(f"\nDetailed Metrics:")
                print(f"  Precision: {report.get('weighted avg', {}).get('precision', 0):.4f}")
                print(f"  Recall: {report.get('weighted avg', {}).get('recall', 0):.4f}")
                print(f"  F1 Score: {report.get('weighted avg', {}).get('f1-score', 0):.4f}")
            
            if 'roc_auc' in eval_results:
                print(f"  ROC AUC: {eval_results['roc_auc']:.4f}")
            
            print("="*50)
        
        elif args.mode == "compare":
            # Model comparison
            print("Running model comparison...")
            
            comparator = ModelComparator(
                dataset_name=args.dataset,
                samples_per_class=args.samples_per_class,
                output_dir=args.output_dir
            )
            
            # Run comparison
            results = comparator.run_comparison(
                epochs=args.epochs,
                batch_size=args.batch_size,
                selected_configs=args.compare_models
            )
            
            # Print summary
            summary = results['comparison_summary']
            
            print("\n" + "="*50)
            print("MODEL COMPARISON COMPLETED!")
            print("="*50)
            print(f"Total experiments: {summary['total_experiments']}")
            print(f"Successful: {summary['successful_experiments']}")
            print(f"Failed: {summary['failed_experiments']}")
            
            if summary['best_model']:
                print(f"\nBest Model: {summary['best_model']['model_name']}")
                print(f"  Accuracy: {summary['best_model']['accuracy']:.4f}")
                print(f"  F1 Score: {summary['best_model']['f1']:.4f}")
                print(f"  Precision: {summary['best_model']['precision']:.4f}")
                print(f"  Recall: {summary['best_model']['recall']:.4f}")
            
            print(f"\nResults saved to: {args.output_dir}")
            print("Check the comparison_report.md for detailed results")
            print("="*50)
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
