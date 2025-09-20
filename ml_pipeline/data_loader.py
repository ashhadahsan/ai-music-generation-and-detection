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
    
    def __init__(self, 
                 dataset_name: str = "ashhadahsan/ai-vs-human-music-dataset",
                 samples_per_class: int = 25,
                 test_split: float = 0.2,
                 val_split: float = 0.2,
                 random_seed: int = 42):
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
            dataset = load_dataset(self.dataset_name)
            
            # Convert to pandas DataFrame
            df = dataset['train'].to_pandas()
            logger.info(f"Loaded {len(df)} total samples")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset from HuggingFace: {e}")
            raise
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by sampling equal numbers from each class"""
        
        # Get class counts
        class_counts = df['label_text'].value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        # Sample from each class
        balanced_samples = []
        
        for label_text in df['label_text'].unique():
            class_data = df[df['label_text'] == label_text]
            
            # Sample up to samples_per_class
            n_samples = min(self.samples_per_class, len(class_data))
            sampled_data = class_data.sample(n=n_samples, random_state=self.random_seed)
            balanced_samples.append(sampled_data)
            
            logger.info(f"Sampled {n_samples} samples from class: {label_text}")
        
        # Combine all samples
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        logger.info(f"Balanced dataset size: {len(balanced_df)}")
        logger.info(f"Final class distribution: {balanced_df['label_text'].value_counts().to_dict()}")
        
        return balanced_df
    
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, and test splits"""
        
        # First split: test vs train+val
        test_size = int(len(df) * self.test_split)
        train_val_size = len(df) - test_size
        
        # Stratified split to maintain class balance
        test_df = df.groupby('label_text').apply(
            lambda x: x.sample(n=int(len(x) * self.test_split), random_state=self.random_seed)
        ).reset_index(drop=True)
        
        train_val_df = df.drop(test_df.index).reset_index(drop=True)
        
        # Second split: train vs val from remaining data
        val_size = int(len(train_val_df) * self.val_split / (1 - self.test_split))
        
        val_df = train_val_df.groupby('label_text').apply(
            lambda x: x.sample(n=int(len(x) * self.val_split / (1 - self.test_split)), 
                             random_state=self.random_seed)
        ).reset_index(drop=True)
        
        train_df = train_val_df.drop(val_df.index).reset_index(drop=True)
        
        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples") 
        logger.info(f"  Test: {len(test_df)} samples")
        
        # Log class distribution for each split
        for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            logger.info(f"{split_name} class distribution: {split_df['label_text'].value_counts().to_dict()}")
        
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
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        logger.info("Dataset preparation completed successfully")
        return dataset_dict
    
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
    print(dataset_dict['train'][0])


if __name__ == "__main__":
    main()
