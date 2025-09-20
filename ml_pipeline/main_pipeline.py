"""
Main pipeline for AI vs Human Music Classification
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np

from data_loader import MusicDataLoader
from wav2vec2_classifier import MusicClassificationTrainer
from evaluator import ModelEvaluator
from audio_preprocessor import AudioPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MusicClassificationPipeline:
    """Complete pipeline for music classification"""
    
    def __init__(self,
                 dataset_name: str = "ashhadahsan/ai-vs-human-music-dataset",
                 samples_per_class: int = 25,
                 model_name: str = "facebook/wav2vec2-base",
                 output_dir: str = "./results",
                 freeze_feature_extractor: bool = False):
        """
        Initialize the complete pipeline
        
        Args:
            dataset_name: HuggingFace dataset name
            samples_per_class: Number of samples per class
            model_name: Base Wav2Vec2 model name
            output_dir: Output directory for results
            freeze_feature_extractor: Whether to freeze feature extractor
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = MusicDataLoader(
            dataset_name=dataset_name,
            samples_per_class=samples_per_class
        )
        
        self.trainer = MusicClassificationTrainer(
            model_name=model_name,
            output_dir=str(self.output_dir / "models"),
            freeze_feature_extractor=freeze_feature_extractor
        )
        
        self.evaluator = ModelEvaluator(
            class_labels=self.data_loader.get_class_labels(),
            output_dir=str(self.output_dir / "evaluation")
        )
        
        self.audio_preprocessor = AudioPreprocessor(model_name=model_name)
        
        logger.info("Music Classification Pipeline initialized")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Samples per class: {samples_per_class}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self) -> Dict:
        """
        Load and prepare the dataset
        
        Returns:
            Dataset dictionary
        """
        logger.info("Loading and preparing dataset...")
        
        dataset_dict = self.data_loader.prepare_dataset_for_training()
        
        logger.info("Dataset loaded successfully")
        return dataset_dict
    
    def train_model(self, 
                   dataset_dict: Dict,
                   training_args: Optional[Dict] = None) -> Dict:
        """
        Train the model
        
        Args:
            dataset_dict: Prepared dataset dictionary
            training_args: Training arguments (optional)
            
        Returns:
            Training results
        """
        logger.info("Starting model training...")
        
        # Setup training arguments
        if training_args is None:
            training_args = {
                'num_train_epochs': 10,
                'per_device_train_batch_size': 4,
                'per_device_eval_batch_size': 4,
                'learning_rate': 3e-5,
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
                'fp16': torch.cuda.is_available()
            }
        
        # Train the model
        trained_trainer = self.trainer.train(dataset_dict, training_args)
        
        # Get training history
        training_history = trained_trainer.state.log_history
        
        logger.info("Model training completed")
        
        return {
            'trainer': trained_trainer,
            'training_history': training_history
        }
    
    def evaluate_model(self, 
                      trainer,
                      dataset_dict: Dict) -> Dict:
        """
        Evaluate the trained model
        
        Args:
            trainer: Trained trainer object
            dataset_dict: Dataset dictionary
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model...")
        
        # Get test predictions
        test_predictions = trainer.predict(dataset_dict['test'])
        
        # Extract labels and predictions
        y_true = [pred.label_ids for pred in test_predictions]
        y_pred = [pred.predictions.argmax() for pred in test_predictions]
        y_prob = np.array([pred.predictions for pred in test_predictions])
        
        # Evaluate predictions
        evaluation_results = self.evaluator.evaluate_predictions(
            y_true, y_pred, y_prob, "Wav2Vec2 Music Classifier"
        )
        
        # Save results
        self.evaluator.save_results(evaluation_results, "wav2vec2_results.json")
        
        # Generate and save report
        report = self.evaluator.generate_report(
            evaluation_results, 
            str(self.output_dir / "evaluation" / "evaluation_report.md")
        )
        
        logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def create_visualizations(self, 
                            evaluation_results: Dict,
                            training_history: List[Dict]) -> None:
        """
        Create visualization plots
        
        Args:
            evaluation_results: Evaluation results
            training_history: Training history
        """
        logger.info("Creating visualizations...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        if 'confusion_matrix' in evaluation_results:
            cm = np.array(evaluation_results['confusion_matrix'])
            self.evaluator.plot_confusion_matrix(
                cm,
                "Wav2Vec2 Music Classification - Confusion Matrix",
                str(plots_dir / "confusion_matrix.png")
            )
        
        # ROC curve
        if 'roc_curve' in evaluation_results:
            roc_data = evaluation_results['roc_curve']
            self.evaluator.plot_roc_curve(
                roc_data['fpr'],
                roc_data['tpr'],
                roc_data['auc'],
                "Wav2Vec2 Music Classification - ROC Curve",
                str(plots_dir / "roc_curve.png")
            )
        
        # Precision-Recall curve
        if 'pr_curve' in evaluation_results:
            pr_data = evaluation_results['pr_curve']
            self.evaluator.plot_precision_recall_curve(
                pr_data['precision'],
                pr_data['recall'],
                pr_data['average_precision'],
                "Wav2Vec2 Music Classification - Precision-Recall Curve",
                str(plots_dir / "precision_recall_curve.png")
            )
        
        # Training history
        if training_history:
            # Extract training metrics
            history_dict = {
                'train_loss': [],
                'eval_loss': [],
                'train_accuracy': [],
                'eval_accuracy': []
            }
            
            for log in training_history:
                for key in history_dict.keys():
                    if key in log:
                        history_dict[key].append(log[key])
            
            self.evaluator.plot_training_history(
                history_dict,
                "Wav2Vec2 Music Classification - Training History",
                str(plots_dir / "training_history.png")
            )
        
        logger.info(f"Visualizations saved to {plots_dir}")
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete pipeline
        
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete music classification pipeline...")
        
        try:
            # Step 1: Load data
            dataset_dict = self.load_data()
            
            # Step 2: Train model
            training_results = self.train_model(dataset_dict)
            trainer = training_results['trainer']
            training_history = training_results['training_history']
            
            # Step 3: Evaluate model
            evaluation_results = self.evaluate_model(trainer, dataset_dict)
            
            # Step 4: Create visualizations
            self.create_visualizations(evaluation_results, training_history)
            
            # Compile final results
            final_results = {
                'dataset_info': {
                    'total_samples': len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test']),
                    'train_samples': len(dataset_dict['train']),
                    'val_samples': len(dataset_dict['validation']),
                    'test_samples': len(dataset_dict['test']),
                    'classes': self.data_loader.get_class_labels()
                },
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'model_path': str(self.trainer.output_dir),
                'output_directory': str(self.output_dir)
            }
            
            # Save final results
            import json
            with open(self.output_dir / "final_results.json", 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info("Complete pipeline finished successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="AI vs Human Music Classification Pipeline")
    
    parser.add_argument("--dataset", type=str, default="ashhadahsan/ai-vs-human-music-dataset",
                       help="HuggingFace dataset name")
    parser.add_argument("--samples-per-class", type=int, default=25,
                       help="Number of samples per class")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base",
                       help="Base Wav2Vec2 model name")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--freeze-feature-extractor", action="store_true",
                       help="Freeze the feature extractor during training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize pipeline
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
        'fp16': torch.cuda.is_available()
    }
    
    # Run pipeline
    results = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Final accuracy: {results['evaluation_results']['accuracy']:.4f}")
    print(f"Results directory: {results['output_directory']}")
    print("="*50)


if __name__ == "__main__":
    main()
