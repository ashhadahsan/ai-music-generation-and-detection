"""
Evaluation utilities for music classification models
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, 
                 class_labels: List[str],
                 output_dir: str = "./evaluation_results"):
        """
        Initialize evaluator
        
        Args:
            class_labels: List of class label names
            output_dir: Directory to save evaluation results
        """
        self.class_labels = class_labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelEvaluator with classes: {class_labels}")
    
    def evaluate_predictions(self, 
                           y_true: List[int],
                           y_pred: List[int],
                           y_prob: Optional[np.ndarray] = None,
                           model_name: str = "Model") -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive evaluation of model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Basic metrics
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_labels,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate additional metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        }
        
        # ROC and PR curves if probabilities are provided
        if y_prob is not None:
            try:
                # ROC curve (assuming binary classification)
                if len(self.class_labels) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                    avg_precision = average_precision_score(y_true, y_prob[:, 1])
                    
                    metrics.update({
                        'roc_auc': roc_auc,
                        'average_precision': avg_precision,
                        'roc_curve': {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': roc_auc
                        },
                        'pr_curve': {
                            'precision': precision.tolist(),
                            'recall': recall.tolist(),
                            'average_precision': avg_precision
                        }
                    })
            except Exception as e:
                logger.warning(f"Could not compute ROC/PR curves: {e}")
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=self.class_labels,
                   yticklabels=self.class_labels)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, 
                      fpr: List[float],
                      tpr: List[float],
                      roc_auc: float,
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: ROC AUC score
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, 
                                  precision: List[float],
                                  recall: List[float],
                                  avg_precision: float,
                                  title: str = "Precision-Recall Curve",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            avg_precision: Average precision score
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            title: str = "Training History",
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history.get('train_loss', []), label='Training Loss')
        axes[0].plot(history.get('eval_loss', []), label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history.get('train_accuracy', []), label='Training Accuracy')
        axes[1].plot(history.get('eval_accuracy', []), label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, 
                      results: List[Dict[str, Union[float, Dict]]],
                      metric: str = 'accuracy',
                      title: str = "Model Comparison",
                      save_path: Optional[str] = None) -> None:
        """
        Compare multiple models
        
        Args:
            results: List of evaluation results from different models
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the plot
        """
        model_names = [result['model_name'] for result in results]
        metric_values = [result.get(metric, 0) for result in results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, 
                       results: Dict[str, Union[float, Dict]],
                       save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: Evaluation results
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        model_name = results.get('model_name', 'Model')
        
        report = f"""
# Model Evaluation Report: {model_name}

## Overview
- **Model**: {model_name}
- **Classes**: {', '.join(self.class_labels)}
- **Accuracy**: {results.get('accuracy', 0):.4f}

## Detailed Metrics
"""
        
        # Classification report
        if 'classification_report' in results:
            report += "\n### Classification Report\n"
            report += f"```\n{classification_report([0, 1], [0, 1], target_names=self.class_labels)}\n```\n"
        
        # ROC AUC
        if 'roc_auc' in results:
            report += f"\n### ROC Analysis\n"
            report += f"- **ROC AUC**: {results['roc_auc']:.4f}\n"
        
        # Average Precision
        if 'average_precision' in results:
            report += f"- **Average Precision**: {results['average_precision']:.4f}\n"
        
        # Confusion Matrix
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            report += f"\n### Confusion Matrix\n"
            report += f"```\n{cm}\n```\n"
            
            # Add interpretation
            if len(self.class_labels) == 2:
                tn, fp, fn, tp = cm.ravel()
                report += f"- **True Positives**: {tp}\n"
                report += f"- **True Negatives**: {tn}\n"
                report += f"- **False Positives**: {fp}\n"
                report += f"- **False Negatives**: {fn}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
    
    def save_results(self, 
                    results: Dict[str, Union[float, Dict]],
                    filename: str = "evaluation_results.json") -> None:
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results
            filename: Output filename
        """
        save_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def load_results(self, 
                    filename: str = "evaluation_results.json") -> Dict[str, Union[float, Dict]]:
        """
        Load evaluation results from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded results
        """
        load_path = self.output_dir / filename
        
        with open(load_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {load_path}")
        return results


def main():
    """Test the evaluator"""
    logging.basicConfig(level=logging.INFO)
    
    # Test evaluator
    evaluator = ModelEvaluator(["ai_generated", "human_created"])
    
    # Dummy data for testing
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6], 
                      [0.4, 0.6], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    
    # Evaluate
    results = evaluator.evaluate_predictions(y_true, y_pred, y_prob, "Test Model")
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
