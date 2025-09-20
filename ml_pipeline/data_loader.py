"""
Data loader for AI vs Human Music Classification Dataset
"""

import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MusicDataLoader:
    """Data loader for the AI vs Human music classification dataset"""

    def __init__(
        self,
        dataset_name: str = "ashhadahsan/ai-vs-human-music-with-audio",
        samples_per_class: int = 25,
        test_split: float = 0.2,
        val_split: float = 0.2,
        random_seed: int = 42,
    ):
        """
        Initialize the data loader

        Args:
            dataset_name: HuggingFace dataset name
            samples_per_class: Number of samples to load per class
            test_split: Fraction of data for testing
            val_split: Fraction of remaining data for validation
            random_seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.samples_per_class = samples_per_class
        self.test_split = test_split
        self.val_split = val_split
        self.random_seed = random_seed

        # Set random seeds for reproducibility
        random.seed(random_seed)

    def load_dataset_from_hf(self) -> pd.DataFrame:
        """Load the dataset from HuggingFace Hub"""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            # Load dataset with audio data
            dataset = load_dataset(self.dataset_name)

            # Convert to pandas DataFrame
            df = dataset["train"].to_pandas()
            logger.info(f"Loaded {len(df)} total samples")

            # Check if audio data is embedded
            if "audio" in df.columns:
                logger.info("✅ Audio data found in dataset")
                # Check a sample to see the audio format
                sample_audio = df["audio"].iloc[0]
                if isinstance(sample_audio, dict):
                    logger.info(f"Audio format: {sample_audio.keys()}")
                else:
                    logger.info(f"Audio data type: {type(sample_audio)}")
            else:
                logger.warning("⚠️ No audio data found in dataset")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset from HuggingFace: {e}")
            raise

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by sampling equal numbers from each class"""

        # Get class counts
        class_counts = df["label_text"].value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")

        # Sample from each class
        balanced_samples = []

        for label_text in df["label_text"].unique():
            class_data = df[df["label_text"] == label_text]

            # Sample up to samples_per_class
            n_samples = min(self.samples_per_class, len(class_data))
            sampled_data = class_data.sample(n=n_samples, random_state=self.random_seed)
            balanced_samples.append(sampled_data)

            logger.info(f"Sampled {n_samples} samples from class: {label_text}")

        # Combine all samples
        balanced_df = pd.concat(balanced_samples, ignore_index=True)

        # Shuffle the dataset
        balanced_df = balanced_df.sample(
            frac=1, random_state=self.random_seed
        ).reset_index(drop=True)

        logger.info(f"Balanced dataset size: {len(balanced_df)}")
        logger.info(
            f"Final class distribution: {balanced_df['label_text'].value_counts().to_dict()}"
        )

        return balanced_df

    def create_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, and test splits"""

        # First split: test vs train+val
        test_size = int(len(df) * self.test_split)
        train_val_size = len(df) - test_size

        # Stratified split to maintain class balance
        test_df = (
            df.groupby("label_text")
            .apply(
                lambda x: x.sample(
                    n=int(len(x) * self.test_split), random_state=self.random_seed
                )
            )
            .reset_index(drop=True)
        )

        train_val_df = df.drop(test_df.index).reset_index(drop=True)

        # Second split: train vs val from remaining data
        val_size = int(len(train_val_df) * self.val_split / (1 - self.test_split))

        val_df = (
            train_val_df.groupby("label_text")
            .apply(
                lambda x: x.sample(
                    n=int(len(x) * self.val_split / (1 - self.test_split)),
                    random_state=self.random_seed,
                )
            )
            .reset_index(drop=True)
        )

        train_df = train_val_df.drop(val_df.index).reset_index(drop=True)

        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")

        # Log class distribution for each split
        for split_name, split_df in [
            ("Train", train_df),
            ("Validation", val_df),
            ("Test", test_df),
        ]:
            logger.info(
                f"{split_name} class distribution: {split_df['label_text'].value_counts().to_dict()}"
            )

        return train_df, val_df, test_df

    def prepare_dataset_for_training(self) -> DatasetDict:
        """Prepare the complete dataset for training"""

        # Load and balance dataset
        df = self.load_dataset_from_hf()
        balanced_df = self.balance_dataset(df)

        # Create splits
        train_df, val_df, test_df = self.create_splits(balanced_df)

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Create dataset dict
        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        logger.info("Dataset preparation completed successfully")
        return dataset_dict

    def load_audio_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Load audio data from Hugging Face cloud dataset"""
        import numpy as np
        import librosa
        import io

        def process_audio_data(example):
            """Process audio data from HuggingFace dataset"""
            try:
                # Check if audio data is already present in the dataset
                if "audio" in example and example["audio"] is not None:
                    # Audio data is already loaded from HuggingFace
                    audio_data = example["audio"]

                    # Convert HuggingFace audio format to Wav2Vec2 format
                    if isinstance(audio_data, dict):
                        if "array" in audio_data and "sampling_rate" in audio_data:
                            # Audio is already in correct format
                            return example
                        elif "bytes" in audio_data:
                            # Convert from bytes to array using librosa
                            try:
                                # Load audio from bytes
                                audio_bytes = audio_data["bytes"]
                                audio_array, sr = librosa.load(
                                    io.BytesIO(audio_bytes), sr=16000
                                )

                                # Ensure audio_array is a numpy array with proper dtype
                                audio_array = np.array(audio_array, dtype=np.float32)

                                return {
                                    **example,
                                    "audio": {
                                        "array": audio_array,
                                        "sampling_rate": sr,
                                    },
                                }
                            except Exception as e:
                                logger.warning(f"Failed to load audio from bytes: {e}")
                                # Create dummy audio data
                                dummy_audio = np.zeros(16000, dtype=np.float32)
                                return {
                                    **example,
                                    "audio": {
                                        "array": dummy_audio,
                                        "sampling_rate": 16000,
                                    },
                                }
                        else:
                            logger.warning(
                                f"Unknown audio format: {list(audio_data.keys())}"
                            )
                            # Create dummy audio data
                            dummy_audio = np.zeros(16000, dtype=np.float32)
                            return {
                                **example,
                                "audio": {"array": dummy_audio, "sampling_rate": 16000},
                            }
                    else:
                        logger.warning(f"Audio data is not a dict: {type(audio_data)}")
                        # Create dummy audio data
                        dummy_audio = np.zeros(16000, dtype=np.float32)
                        return {
                            **example,
                            "audio": {"array": dummy_audio, "sampling_rate": 16000},
                        }
                else:
                    # No audio data found - create dummy data
                    logger.warning(
                        f"No audio data found for {example.get('file_path', 'unknown')}"
                    )
                    dummy_audio = np.zeros(
                        16000, dtype=np.float32
                    )  # 1 second of silence
                    return {
                        **example,
                        "audio": {"array": dummy_audio, "sampling_rate": 16000},
                    }
            except Exception as e:
                logger.warning(
                    f"Failed to process audio for {example.get('file_path', 'unknown')}: {e}"
                )
                # Create dummy audio data for failed loads
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                return {
                    **example,
                    "audio": {"array": dummy_audio, "sampling_rate": 16000},
                }

        # Apply audio processing to all splits
        logger.info("Processing audio data from HuggingFace dataset...")

        processed_dataset = DatasetDict()
        for split_name, split_dataset in dataset_dict.items():
            logger.info(f"Processing audio for {split_name} split...")
            processed_split = split_dataset.map(
                process_audio_data,
                desc=f"Processing audio for {split_name}",
                num_proc=1,  # Use single process to avoid issues
            )

            # Post-process to ensure audio arrays are numpy arrays
            def fix_audio_array(example):
                if "audio" in example and isinstance(example["audio"], dict):
                    if "array" in example["audio"]:
                        # Convert list back to numpy array if needed
                        audio_array = example["audio"]["array"]
                        if isinstance(audio_array, list):
                            example["audio"]["array"] = np.array(
                                audio_array, dtype=np.float32
                            )
                        elif not isinstance(audio_array, np.ndarray):
                            example["audio"]["array"] = np.array(
                                audio_array, dtype=np.float32
                            )
                return example

            processed_dataset[split_name] = processed_split.map(
                fix_audio_array,
                desc=f"Fixing audio arrays for {split_name}",
                num_proc=1,
            )

        logger.info("Audio processing completed")
        return processed_dataset

    def get_class_labels(self) -> List[str]:
        """Get the list of class labels"""
        return ["ai_generated", "human_created"]

    def get_num_classes(self) -> int:
        """Get the number of classes"""
        return len(self.get_class_labels())


def main():
    """Test the data loader"""
    logging.basicConfig(level=logging.INFO)

    # Initialize data loader
    loader = MusicDataLoader(samples_per_class=25)

    # Load and prepare dataset
    dataset_dict = loader.prepare_dataset_for_training()

    # Print dataset info
    print("\nDataset Summary:")
    print(f"Number of classes: {loader.get_num_classes()}")
    print(f"Class labels: {loader.get_class_labels()}")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")

    # Show sample data
    print("\nSample training data:")
    print(dataset_dict["train"][0])


if __name__ == "__main__":
    main()
