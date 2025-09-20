#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from random import randint
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.56.0")

require_version(
    "datasets>=1.14.0",
    "To fix: pip install -r examples/pytorch/audio-classification/requirements.txt",
)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A file containing the training audio paths and labels."},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "A file containing the validation audio paths and labels."},
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    label_column_name: str = field(
        default="label",
        metadata={
            "help": "The name of the dataset column containing the labels. Defaults to 'label'"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={
            "help": "Audio clips will be randomly cut to this length during training if the value is set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from the Hub"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to generate an attention mask in the feature extractor."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to freeze the feature extractor layers of the model."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder` "
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`. "
                "Only make use of `--freeze_feature_encoder`."
            )


class MusicClassificationTrainer:
    """Enhanced trainer class using the working Hugging Face audio classification approach"""

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
        self.output_dir = output_dir

        # Initialize model arguments
        self.model_args = ModelArguments(
            model_name_or_path=model_name,
            freeze_feature_encoder=freeze_feature_extractor,
            attention_mask=True,
        )

        # Initialize data arguments
        self.data_args = DataTrainingArguments(
            audio_column_name="audio",
            label_column_name="label_text",  # Use label_text to match existing data loader
            max_length_seconds=20,
        )

        logger.info(f"Initialized MusicClassificationTrainer")
        logger.info(f"Model: {model_name}")
        logger.info(f"Output directory: {output_dir}")

    def prepare_dataset(self, dataset_dict):
        """
        Prepare dataset for training using the working Hugging Face approach

        Args:
            dataset_dict: HuggingFace dataset dictionary

        Returns:
            Preprocessed dataset dictionary
        """
        # Initialize feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_args.feature_extractor_name
            or self.model_args.model_name_or_path,
            return_attention_mask=self.model_args.attention_mask,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        # Convert label_text to numeric labels if needed
        def convert_labels(example):
            if "label_text" in example and "label" not in example:
                # Convert label_text to numeric label
                label_mapping = {"ai_generated": 0, "human_created": 1}
                example["label"] = label_mapping.get(example["label_text"], 0)
            return example

        # Apply label conversion to all splits
        for split_name in dataset_dict.keys():
            dataset_dict[split_name] = dataset_dict[split_name].map(convert_labels)

        # Cast audio column to proper format
        dataset_dict = dataset_dict.cast_column(
            self.data_args.audio_column_name,
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
        )

        model_input_name = feature_extractor.model_input_names[0]

        def train_transforms(batch):
            """Apply train_transforms across a batch."""
            subsampled_wavs = []
            for audio in batch[self.data_args.audio_column_name]:
                wav = random_subsample(
                    audio["array"],
                    max_length=self.data_args.max_length_seconds,
                    sample_rate=feature_extractor.sampling_rate,
                )
                subsampled_wavs.append(wav)
            inputs = feature_extractor(
                subsampled_wavs, sampling_rate=feature_extractor.sampling_rate
            )
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[self.data_args.label_column_name])
            return output_batch

        def val_transforms(batch):
            """Apply val_transforms across a batch."""
            wavs = [audio["array"] for audio in batch[self.data_args.audio_column_name]]
            inputs = feature_extractor(
                wavs, sampling_rate=feature_extractor.sampling_rate
            )
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[self.data_args.label_column_name])
            return output_batch

        # Apply transforms
        if "train" in dataset_dict:
            dataset_dict["train"].set_transform(
                train_transforms, output_all_columns=False
            )
        if "validation" in dataset_dict:
            dataset_dict["validation"].set_transform(
                val_transforms, output_all_columns=False
            )

        return dataset_dict, feature_extractor

    def setup_model(self, dataset_dict, feature_extractor):
        """Setup the model with proper configuration"""
        # Prepare label mappings - use predefined labels since we converted them
        labels = ["ai_generated", "human_created"]
        label2id, id2label = {}, {}
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        config = AutoConfig.from_pretrained(
            self.model_args.config_name or self.model_args.model_name_or_path,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="audio-classification",
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        model = AutoModelForAudioClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
            ignore_mismatched_sizes=self.model_args.ignore_mismatched_sizes,
        )

        # Freeze the convolutional waveform encoder if requested
        if self.model_args.freeze_feature_encoder:
            model.freeze_feature_encoder()

        return model, feature_extractor

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Load accuracy metric
        metric = evaluate.load("accuracy", cache_dir=self.model_args.cache_dir)
        return metric.compute(predictions=predictions, references=labels)

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
        fp16: bool = False,
    ) -> TrainingArguments:
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
            dataloader_num_workers=dataloader_num_workers,
            fp16=fp16,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
        )

    def train(self, dataset_dict, training_args: Optional[TrainingArguments] = None):
        """
        Train the model using the working Hugging Face approach

        Args:
            dataset_dict: HuggingFace dataset dictionary
            training_args: Training arguments (optional)

        Returns:
            Trained trainer object
        """
        # Set seed for reproducibility
        set_seed(42)

        # Setup training arguments if not provided
        if training_args is None:
            training_args = self.setup_training_args()

        # Prepare dataset
        processed_dataset, feature_extractor = self.prepare_dataset(dataset_dict)

        # Setup model
        model, feature_extractor = self.setup_model(
            processed_dataset, feature_extractor
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=(
                processed_dataset["train"] if "train" in processed_dataset else None
            ),
            eval_dataset=(
                processed_dataset["validation"]
                if "validation" in processed_dataset
                else None
            ),
            compute_metrics=self.compute_metrics,
            processing_class=feature_extractor,
        )

        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save the model and feature extractor
        trainer.save_model()
        feature_extractor.save_pretrained(self.output_dir)

        # Log and save metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        logger.info(f"Training completed. Model saved to {self.output_dir}")

        return trainer

    def evaluate(self, trainer, test_dataset=None):
        """Evaluate the trained model"""
        if test_dataset is None:
            logger.warning("No test dataset provided for evaluation")
            return {}

        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)

        logger.info("Test Results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value:.4f}")

        return test_results

    def predict(self, trainer, audio_paths, class_labels):
        """Make predictions on new audio files"""
        predictions = []

        for audio_path in audio_paths:
            try:
                # Load and preprocess audio
                import librosa

                audio, sr = librosa.load(audio_path, sr=16000)

                # Prepare input
                inputs = trainer.processing_class(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # Make prediction
                with torch.no_grad():
                    outputs = trainer.model(**inputs)
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

    # Test trainer initialization
    trainer = MusicClassificationTrainer()
    print(f"Trainer initialized with output directory: {trainer.output_dir}")


if __name__ == "__main__":
    main()
