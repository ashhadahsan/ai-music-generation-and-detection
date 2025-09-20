"""
Model comparison utilities for testing different Wav2Vec2 configurations
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

from main_pipeline import MusicClassificationPipeline
from evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare different model configurations"""
    
    def __init__(self, 
                 dataset_name: str = "ashhadahsan/ai-vs-human-music-dataset",
                 samples_per_class: int = 25,
                 output_dir: str = "./comparison_results"):
        """
        Initialize model comparator
        
        Args:
            dataset_name: HuggingFace dataset name
            samples_per_class: Number of samples per class
            output_dir: Output directory for comparison results
        """
        self.dataset_name = dataset_name
        self.samples_per_class = samples_per_class
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations to test
        self.model_configs = [
            {
                'name': 'wav2vec2-base',
                'model_name': 'facebook/wav2vec2-base',
                'freeze_feature_extractor': False,
                'description': 'Wav2Vec2 Base - Full fine-tuning'
            },
            {
                'name': 'wav2vec2-base-frozen',
                'model_name': 'facebook/wav2vec2-base',
                'freeze_feature_extractor': True,
                'description': 'Wav2Vec2 Base - Frozen feature extractor'
            },
            {
                'name': 'wav2vec2-large',
                'model_name': 'facebook/wav2vec2-large',
                'freeze_feature_extractor': False,
                'description': 'Wav2Vec2 Large - Full fine-tuning'
            },
            {
                'name': 'wav2vec2-large-frozen',
                'model_name': 'facebook/wav2vec2-large',
                'freeze_feature_extractor': True,
                'description': 'Wav2Vec2 Large - Frozen feature extractor'
            }
        ]
        
        logger.info(f"Initialized ModelComparator with {len(self.model_configs)} configurations")
    
    def run_single_experiment(self, 
                            config: Dict,
                            epochs: int = 5,
                            batch_size: int = 2) -> Dict:
        """
        Run a single experiment with given configuration
        
        Args:
            config: Model configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Experiment results
        """
        logger.info(f"Running experiment: {config['name']}")
        logger.info(f"Description: {config['description']}")
        
        # Create pipeline for this configuration
        pipeline = MusicClassificationPipeline(
            dataset_name=self.dataset_name,
            samples_per_class=self.samples_per_class,
            model_name=config['model_name'],
            output_dir=str(self.output_dir / config['name']),
            freeze_feature_extractor=config['freeze_feature_extractor']
        )
        
        # Custom training arguments for faster comparison
        training_args = {
            'num_train_epochs': epochs,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': 3e-5,
            'warmup_steps': 50,
            'weight_decay': 0.01,
            'logging_steps': 5,
            'eval_steps': 25,
            'save_steps': 50,
            'evaluation_strategy': "steps",
            'save_strategy': "steps",
            'load_best_model_at_end': True,
            'metric_for_best_model': "eval_accuracy",
            'greater_is_better': True,
            'save_total_limit': 2,
            'dataloader_num_workers': 2,
            'fp16': False  # Disable for compatibility
        }
        
        try:
            # Load data once
            dataset_dict = pipeline.load_data()
            
            # Train model
            training_results = pipeline.train_model(dataset_dict, training_args)
            
            # Evaluate model
            evaluation_results = pipeline.evaluate_model(
                training_results['trainer'], 
                dataset_dict
            )
            
            # Compile results
            experiment_results = {
                'config': config,
                'training_args': training_args,
                'evaluation_results': evaluation_results,
                'training_history': training_results['training_history'],
                'success': True
            }
            
            logger.info(f"Experiment {config['name']} completed successfully")
            logger.info(f"Final accuracy: {evaluation_results['accuracy']:.4f}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment {config['name']} failed: {e}")
            return {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    def run_comparison(self, 
                      epochs: int = 5,
                      batch_size: int = 2,
                      selected_configs: Optional[List[str]] = None) -> Dict:
        """
        Run comparison across multiple model configurations
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            selected_configs: List of config names to run (None for all)
            
        Returns:
            Comparison results
        """
        logger.info("Starting model comparison...")
        
        # Filter configurations if specified
        configs_to_run = self.model_configs
        if selected_configs:
            configs_to_run = [
                config for config in self.model_configs 
                if config['name'] in selected_configs
            ]
        
        logger.info(f"Running {len(configs_to_run)} model configurations")
        
        all_results = []
        
        for config in configs_to_run:
            result = self.run_single_experiment(config, epochs, batch_size)
            all_results.append(result)
        
        # Create comparison summary
        comparison_summary = self.create_comparison_summary(all_results)
        
        # Save results
        self.save_comparison_results(all_results, comparison_summary)
        
        # Create comparison plots
        self.create_comparison_plots(all_results)
        
        logger.info("Model comparison completed")
        
        return {
            'individual_results': all_results,
            'comparison_summary': comparison_summary
        }
    
    def create_comparison_summary(self, results: List[Dict]) -> Dict:
        """
        Create a summary of comparison results
        
        Args:
            results: List of individual experiment results
            
        Returns:
            Comparison summary
        """
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        summary = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(failed_results),
            'model_performance': [],
            'best_model': None,
            'worst_model': None
        }
        
        if successful_results:
            # Extract performance metrics
            for result in successful_results:
                config = result['config']
                eval_results = result['evaluation_results']
                
                performance = {
                    'model_name': config['name'],
                    'description': config['description'],
                    'accuracy': eval_results.get('accuracy', 0),
                    'f1': eval_results.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0),
                    'precision': eval_results.get('classification_report', {}).get('weighted avg', {}).get('precision', 0),
                    'recall': eval_results.get('classification_report', {}).get('weighted avg', {}).get('recall', 0)
                }
                
                # Add ROC AUC if available
                if 'roc_auc' in eval_results:
                    performance['roc_auc'] = eval_results['roc_auc']
                
                summary['model_performance'].append(performance)
            
            # Sort by accuracy
            summary['model_performance'].sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Find best and worst models
            if summary['model_performance']:
                summary['best_model'] = summary['model_performance'][0]
                summary['worst_model'] = summary['model_performance'][-1]
        
        # Add failed experiments info
        if failed_results:
            summary['failed_models'] = [
                {
                    'model_name': r['config']['name'],
                    'error': r.get('error', 'Unknown error')
                }
                for r in failed_results
            ]
        
        return summary
    
    def save_comparison_results(self, 
                              results: List[Dict], 
                              summary: Dict) -> None:
        """
        Save comparison results to files
        
        Args:
            results: Individual experiment results
            summary: Comparison summary
        """
        # Save detailed results
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        with open(self.output_dir / "comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create human-readable report
        self.create_comparison_report(summary)
        
        logger.info(f"Comparison results saved to {self.output_dir}")
    
    def create_comparison_report(self, summary: Dict) -> None:
        """
        Create a human-readable comparison report
        
        Args:
            summary: Comparison summary
        """
        report_path = self.output_dir / "comparison_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"**Dataset**: {self.dataset_name}\n")
            f.write(f"**Samples per class**: {self.samples_per_class}\n")
            f.write(f"**Total experiments**: {summary['total_experiments']}\n")
            f.write(f"**Successful experiments**: {summary['successful_experiments']}\n")
            f.write(f"**Failed experiments**: {summary['failed_experiments']}\n\n")
            
            if summary['best_model']:
                f.write("## Best Performing Model\n\n")
                f.write(f"**Model**: {summary['best_model']['model_name']}\n")
                f.write(f"**Description**: {summary['best_model']['description']}\n")
                f.write(f"**Accuracy**: {summary['best_model']['accuracy']:.4f}\n")
                f.write(f"**F1 Score**: {summary['best_model']['f1']:.4f}\n")
                f.write(f"**Precision**: {summary['best_model']['precision']:.4f}\n")
                f.write(f"**Recall**: {summary['best_model']['recall']:.4f}\n")
                if 'roc_auc' in summary['best_model']:
                    f.write(f"**ROC AUC**: {summary['best_model']['roc_auc']:.4f}\n")
                f.write("\n")
            
            f.write("## All Model Performance\n\n")
            f.write("| Model | Description | Accuracy | F1 | Precision | Recall | ROC AUC |\n")
            f.write("|-------|-------------|----------|----|-----------|--------|----------|\n")
            
            for model in summary['model_performance']:
                roc_auc = model.get('roc_auc', 'N/A')
                if isinstance(roc_auc, float):
                    roc_auc = f"{roc_auc:.4f}"
                
                f.write(f"| {model['model_name']} | {model['description']} | "
                       f"{model['accuracy']:.4f} | {model['f1']:.4f} | "
                       f"{model['precision']:.4f} | {model['recall']:.4f} | {roc_auc} |\n")
            
            if summary.get('failed_models'):
                f.write("\n## Failed Experiments\n\n")
                for failed in summary['failed_models']:
                    f.write(f"- **{failed['model_name']}**: {failed['error']}\n")
        
        logger.info(f"Comparison report saved to {report_path}")
    
    def create_comparison_plots(self, results: List[Dict]) -> None:
        """
        Create comparison plots
        
        Args:
            results: Individual experiment results
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            logger.warning("No successful results to plot")
            return
        
        # Create evaluator for plotting
        evaluator = ModelEvaluator(
            class_labels=["ai_generated", "human_created"],
            output_dir=str(self.output_dir)
        )
        
        # Extract metrics for comparison
        model_names = []
        accuracies = []
        f1_scores = []
        
        for result in successful_results:
            config = result['config']
            eval_results = result['evaluation_results']
            
            model_names.append(config['name'])
            accuracies.append(eval_results.get('accuracy', 0))
            
            f1 = eval_results.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0)
            f1_scores.append(f1)
        
        # Create accuracy comparison plot
        evaluator.compare_models(
            [{'model_name': name, 'accuracy': acc} for name, acc in zip(model_names, accuracies)],
            metric='accuracy',
            title='Model Accuracy Comparison',
            save_path=str(self.output_dir / 'accuracy_comparison.png')
        )
        
        # Create F1 score comparison plot
        evaluator.compare_models(
            [{'model_name': name, 'f1': f1} for name, f1 in zip(model_names, f1_scores)],
            metric='f1',
            title='Model F1 Score Comparison',
            save_path=str(self.output_dir / 'f1_comparison.png')
        )


def main():
    """Main function for model comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare different Wav2Vec2 configurations")
    
    parser.add_argument("--dataset", type=str, default="ashhadahsan/ai-vs-human-music-dataset",
                       help="HuggingFace dataset name")
    parser.add_argument("--samples-per-class", type=int, default=25,
                       help="Number of samples per class")
    parser.add_argument("--output-dir", type=str, default="./comparison_results",
                       help="Output directory for comparison results")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs for comparison")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--models", nargs='+', 
                       choices=['wav2vec2-base', 'wav2vec2-base-frozen', 'wav2vec2-large', 'wav2vec2-large-frozen'],
                       help="Specific models to compare (default: all)")
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(
        dataset_name=args.dataset,
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir
    )
    
    # Run comparison
    results = comparator.run_comparison(
        epochs=args.epochs,
        batch_size=args.batch_size,
        selected_configs=args.models
    )
    
    # Print summary
    summary = results['comparison_summary']
    
    print("\n" + "="*60)
    print("MODEL COMPARISON COMPLETED!")
    print("="*60)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {summary['failed_experiments']}")
    
    if summary['best_model']:
        print(f"\nBest Model: {summary['best_model']['model_name']}")
        print(f"Accuracy: {summary['best_model']['accuracy']:.4f}")
        print(f"F1 Score: {summary['best_model']['f1']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
