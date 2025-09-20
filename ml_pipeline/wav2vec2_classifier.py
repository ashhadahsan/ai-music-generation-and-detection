"""
Wav2Vec2-based classifier for AI vs Human music classification
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
    AutoFeatureExtractor,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Wav2Vec2Classifier(nn.Module):
    """Custom Wav2Vec2 classifier with additional classification head"""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        freeze_feature_extractor: bool = False,
    ):
        """
        Initialize Wav2Vec2 classifier

        Args:
            model_name: Base Wav2Vec2 model name
            num_labels: Number of classification labels
            dropout_rate: Dropout rate for classification head
            freeze_feature_extractor: Whether to freeze the feature extractor
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load base Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Set dropout rate after initialization (compatibility fix)
        if hasattr(self.wav2vec2, "classifier_dropout"):
            self.wav2vec2.classifier_dropout = dropout_rate

        # Freeze feature extractor if requested
        if freeze_feature_extractor:
            self.wav2vec2.freeze_feature_extractor()
            logger.info("Feature extractor frozen")

        logger.info(f"Initialized Wav2Vec2Classifier with {num_labels} labels")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        """
        Forward pass

        Args:
            input_values: Audio input values
            attention_mask: Attention mask
            labels: Ground truth labels

        Returns:
            Model output with logits and loss
        """
        return self.wav2vec2(
            input_values=input_values, attention_mask=attention_mask, labels=labels
        )


class MusicClassificationTrainer:
    """Trainer class for music classification with Wav2Vec2"""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_labels: int = 2,
        output_dir: str = "./models/wav2vec2_music_classifier",
        freeze_feature_extractor: bool = False,
    ):
        """
        Initialize trainer

        Args:
            model_name: Base Wav2Vec2 model name
            num_labels: Number of classification labels
            output_dir: Directory to save trained models
            freeze_feature_extractor: Whether to freeze the feature extractor
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = Wav2Vec2Classifier(
            model_name=model_name,
            num_labels=num_labels,
            freeze_feature_extractor=freeze_feature_extractor,
        )

        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        logger.info(f"Initialized MusicClassificationTrainer")
        logger.info(f"Output directory: {self.output_dir}")

    def prepare_dataset(self, dataset_dict):
        """
        Prepare dataset for training

        Args:
            dataset_dict: HuggingFace dataset dictionary

        Returns:
            Preprocessed dataset dictionary
        """

        def preprocess_function(examples):
            """Preprocess batch of examples"""
            # Get audio arrays and labels
            audio_arrays = examples["audio"]
            labels = examples["label"]

            # Process audio through feature extractor
            inputs = self.processor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            return {
                "input_values": inputs.input_values,
                "attention_mask": inputs.attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        # Apply preprocessing to all splits
        processed_dataset = dataset_dict.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
        )

        return processed_dataset

    def setup_training_args(
        self,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        learning_rate: float = 3e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_accuracy",
        greater_is_better: bool = True,
        save_total_limit: int = 3,
        dataloader_num_workers: int = 4,
        fp16: bool = True,
    ) -> TrainingArguments:
        """
        Setup training arguments

        Returns:
            TrainingArguments object
        """
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
            dataloader_num_workers=dataloader_num_workers,
            fp16=fp16,
            gradient_checkpointing=True,  # Fix gradient checkpointing warning
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # Disable wandb/tensorboard for now
        )

        return training_args

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        import evaluate

        # Load metrics
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Compute metrics
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
        }

    def train(
        self, dataset_dict, training_args: Optional[TrainingArguments] = None
    ) -> Trainer:
        """
        Train the model

        Args:
            dataset_dict: Preprocessed dataset dictionary
            training_args: Training arguments (optional)

        Returns:
            Trained trainer object
        """
        # Setup training arguments if not provided
        if training_args is None:
            training_args = self.setup_training_args()

        # Prepare dataset
        processed_dataset = self.prepare_dataset(dataset_dict)

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        trainer.save_model()
        self.processor.save_pretrained(self.output_dir)

        logger.info(f"Training completed. Model saved to {self.output_dir}")

        return trainer

    def evaluate(self, trainer: Trainer, test_dataset=None) -> Dict[str, float]:
        """
        Evaluate the trained model

        Args:
            trainer: Trained trainer object
            test_dataset: Test dataset (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        if test_dataset is None:
            logger.warning("No test dataset provided for evaluation")
            return {}

        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)

        logger.info("Test Results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value:.4f}")

        return test_results

    def predict(
        self, trainer: Trainer, audio_paths: List[str], class_labels: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Make predictions on new audio files

        Args:
            trainer: Trained trainer object
            audio_paths: List of audio file paths
            class_labels: List of class label names

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for audio_path in audio_paths:
            try:
                # Load and preprocess audio
                import librosa

                audio, sr = librosa.load(audio_path, sr=16000)

                # Prepare input
                inputs = self.processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # Make prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class_id].item()

                prediction = {
                    "audio_path": audio_path,
                    "predicted_class": class_labels[predicted_class_id],
                    "confidence": confidence,
                    "probabilities": {
                        class_labels[i]: probabilities[0][i].item()
                        for i in range(len(class_labels))
                    },
                }

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error predicting {audio_path}: {e}")
                continue

        return predictions


def main():
    """Test the classifier"""
    logging.basicConfig(level=logging.INFO)

    # Test classifier initialization
    classifier = Wav2Vec2Classifier()
    print(f"Model initialized with {classifier.num_labels} labels")

    # Test trainer initialization
    trainer = MusicClassificationTrainer()
    print(f"Trainer initialized with output directory: {trainer.output_dir}")


if __name__ == "__main__":
    main()
