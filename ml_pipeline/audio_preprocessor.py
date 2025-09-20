"""
Audio preprocessing utilities for Wav2Vec2 fine-tuning
"""

import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing class for Wav2Vec2 models"""
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base",
                 target_sr: int = 16000,
                 max_duration: float = 30.0,
                 min_duration: float = 1.0):
        """
        Initialize audio preprocessor
        
        Args:
            model_name: Wav2Vec2 model name
            target_sr: Target sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
        """
        self.model_name = model_name
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            sampling_rate=target_sr
        )
        
        logger.info(f"Initialized AudioPreprocessor with model: {model_name}")
        logger.info(f"Target sample rate: {target_sr} Hz")
        logger.info(f"Duration range: {min_duration}s - {max_duration}s")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio with librosa
            audio_array, sr = librosa.load(audio_path, sr=self.target_sr)
            
            logger.debug(f"Loaded audio: {audio_path}")
            logger.debug(f"Original shape: {audio_array.shape}, Sample rate: {sr}")
            
            return audio_array, sr
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def trim_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Trim audio to specified duration range
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Trimmed audio array
        """
        duration = len(audio) / sr
        
        if duration > self.max_duration:
            # Trim to max duration
            max_samples = int(self.max_duration * sr)
            audio = audio[:max_samples]
            logger.debug(f"Trimmed audio from {duration:.2f}s to {self.max_duration}s")
            
        elif duration < self.min_duration:
            # Pad to min duration
            min_samples = int(self.min_duration * sr)
            if len(audio) < min_samples:
                padding = min_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
                logger.debug(f"Padded audio from {duration:.2f}s to {self.min_duration}s")
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio array
        """
        # Avoid division by zero
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        return audio
    
    def add_noise_augmentation(self, 
                              audio: np.ndarray, 
                              noise_factor: float = 0.005) -> np.ndarray:
        """
        Add Gaussian noise for data augmentation
        
        Args:
            audio: Audio array
            noise_factor: Noise strength factor
            
        Returns:
            Audio with added noise
        """
        noise = np.random.normal(0, noise_factor, audio.shape)
        return audio + noise
    
    def time_shift_augmentation(self, 
                               audio: np.ndarray, 
                               sr: int,
                               shift_factor: float = 0.2) -> np.ndarray:
        """
        Apply time shift augmentation
        
        Args:
            audio: Audio array
            sr: Sample rate
            shift_factor: Maximum shift as fraction of audio length
            
        Returns:
            Time-shifted audio
        """
        shift_samples = int(len(audio) * shift_factor * np.random.uniform(-1, 1))
        
        if shift_samples > 0:
            # Shift right, pad left
            audio = np.pad(audio, (shift_samples, 0), mode='constant')[:-shift_samples]
        elif shift_samples < 0:
            # Shift left, pad right
            audio = np.pad(audio, (0, -shift_samples), mode='constant')[-shift_samples:]
            
        return audio
    
    def preprocess_audio(self, 
                        audio_path: str,
                        apply_augmentation: bool = False) -> Dict[str, Union[np.ndarray, int]]:
        """
        Complete audio preprocessing pipeline
        
        Args:
            audio_path: Path to audio file
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Trim to duration range
        audio = self.trim_audio(audio, sr)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Apply augmentation if requested
        if apply_augmentation:
            audio = self.add_noise_augmentation(audio)
            audio = self.time_shift_augmentation(audio, sr)
        
        # Ensure audio is in correct format for Wav2Vec2
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'duration': len(audio) / sr
        }
    
    def batch_preprocess(self, 
                        audio_paths: List[str],
                        apply_augmentation: bool = False) -> List[Dict[str, Union[np.ndarray, int]]]:
        """
        Preprocess multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            List of preprocessed audio dictionaries
        """
        processed_audios = []
        
        for audio_path in audio_paths:
            try:
                processed = self.preprocess_audio(audio_path, apply_augmentation)
                processed_audios.append(processed)
            except Exception as e:
                logger.error(f"Error preprocessing {audio_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_audios)}/{len(audio_paths)} audio files")
        return processed_audios
    
    def prepare_for_wav2vec2(self, 
                            audio: np.ndarray,
                            sr: int) -> torch.Tensor:
        """
        Prepare audio for Wav2Vec2 model input
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Tensor ready for Wav2Vec2
        """
        # Use feature extractor to prepare input
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt", 
            padding=True
        )
        
        return inputs.input_values.squeeze(0)  # Remove batch dimension
    
    def create_audio_features(self, 
                             audio_data: Dict[str, Union[np.ndarray, int]]) -> Dict[str, torch.Tensor]:
        """
        Create audio features for training
        
        Args:
            audio_data: Preprocessed audio data
            
        Returns:
            Dictionary with audio features
        """
        audio = audio_data['audio']
        sr = audio_data['sample_rate']
        
        # Prepare for Wav2Vec2
        input_values = self.prepare_for_wav2vec2(audio, sr)
        
        return {
            'input_values': input_values,
            'attention_mask': torch.ones_like(input_values)
        }


def test_preprocessor():
    """Test the audio preprocessor"""
    import tempfile
    import os
    
    # Create a dummy audio file for testing
    duration = 5  # seconds
    sr = 16000
    t = np.linspace(0, duration, sr * duration)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sr)
        temp_path = tmp_file.name
    
    try:
        # Test preprocessor
        preprocessor = AudioPreprocessor()
        
        # Test preprocessing
        processed = preprocessor.preprocess_audio(temp_path)
        print(f"Processed audio shape: {processed['audio'].shape}")
        print(f"Processed audio duration: {processed['duration']:.2f}s")
        
        # Test feature creation
        features = preprocessor.create_audio_features(processed)
        print(f"Input values shape: {features['input_values'].shape}")
        print(f"Attention mask shape: {features['attention_mask'].shape}")
        
    finally:
        # Clean up
        os.unlink(temp_path)


if __name__ == "__main__":
    test_preprocessor()
