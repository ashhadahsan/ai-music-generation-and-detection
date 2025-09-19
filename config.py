"""
Configuration file for AI Music Dataset Generator using Pydantic Settings
Controls audio duration, number of samples, and other settings
"""

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path
import os


class AudioSettings(BaseModel):
    """Audio configuration settings"""

    duration_seconds: int = Field(
        default=30, description="Duration for both AI and human music samples"
    )
    max_duration_seconds: int = Field(
        default=300, description="Maximum duration for YouTube videos (5 minutes)"
    )
    min_duration_seconds: int = Field(
        default=10, description="Minimum duration for YouTube videos"
    )
    audio_format: str = Field(default="mp3", description="Audio format")
    audio_quality: str = Field(default="192K", description="Audio quality")


class AIGenerationSettings(BaseModel):
    """AI music generation settings"""

    num_tracks: int = Field(default=2, description="Number of AI tracks to generate")
    delay_between_generations: int = Field(
        default=5, description="Seconds to wait between generations"
    )
    model_version: str = Field(
        default="stereo-large", description="MusicGen model version"
    )
    temperature: float = Field(default=1.0, description="Generation temperature")
    top_k: int = Field(default=250, description="Top-k sampling parameter")
    top_p: float = Field(default=0, description="Top-p sampling parameter")
    classifier_free_guidance: int = Field(
        default=3, description="Classifier-free guidance"
    )
    normalization_strategy: str = Field(
        default="loudness", description="Audio normalization strategy"
    )


class HumanExtractionSettings(BaseModel):
    """Human music extraction settings"""

    max_results_per_term: int = Field(
        default=10, description="Max YouTube results per search term"
    )
    delay_between_extractions: int = Field(
        default=2, description="Seconds to wait between extractions"
    )
    delay_between_searches: int = Field(
        default=1, description="Seconds to wait between searches"
    )
    default_search_terms: List[str] = Field(
        default=[
            "classical",
            "jazz",
            "blues",
            "folk",
            "acoustic",
            "piano",
            "guitar",
            "orchestral",
            "chamber music",
            "solo instrumental",
        ],
        description="Default search terms for music extraction",
    )


class DatasetSettings(BaseModel):
    """Dataset configuration settings"""

    output_dir: str = Field(
        default="ai_music_dataset", description="Output directory for dataset"
    )
    ai_subdir: str = Field(
        default="ai_generated", description="Subdirectory for AI-generated music"
    )
    human_subdir: str = Field(
        default="human_created", description="Subdirectory for human-created music"
    )
    metadata_filename: str = Field(
        default="dataset_metadata.json", description="Main metadata filename"
    )
    extraction_metadata_filename: str = Field(
        default="extraction_metadata.json", description="Extraction metadata filename"
    )


class HuggingFaceSettings(BaseModel):
    """Hugging Face settings"""

    default_private: bool = Field(
        default=True, description="Make datasets private by default"
    )
    default_push_audio: bool = Field(
        default=True, description="Push audio files by default"
    )
    dataset_description: str = Field(
        default="AI vs Human Music Classification Dataset",
        description="Default dataset description",
    )


class LoggingSettings(BaseModel):
    """Logging configuration"""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(levelname)s - %(message)s", description="Log format"
    )


class RateLimitingSettings(BaseModel):
    """Rate limiting settings"""

    youtube_search_delay: int = Field(
        default=1, description="Seconds between YouTube searches"
    )
    youtube_download_delay: int = Field(
        default=2, description="Seconds between YouTube downloads"
    )
    ai_generation_delay: int = Field(
        default=5, description="Seconds between AI generations"
    )


class Config(BaseSettings):
    """Main configuration class using Pydantic Settings"""

    # Nested settings
    audio: AudioSettings = Field(default_factory=AudioSettings)
    ai_generation: AIGenerationSettings = Field(default_factory=AIGenerationSettings)
    human_extraction: HumanExtractionSettings = Field(
        default_factory=HumanExtractionSettings
    )
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    rate_limiting: RateLimitingSettings = Field(default_factory=RateLimitingSettings)

    class Config:
        env_prefix = (
            "AI_MUSIC_"  # Environment variables will be prefixed with AI_MUSIC_
        )
        env_nested_delimiter = "__"  # Use double underscore for nested settings
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

    @model_validator(mode="after")
    def apply_legacy_overrides(self):
        """Apply legacy environment variable overrides"""
        # Legacy environment variable mappings
        legacy_mappings = {
            "AI_MUSIC_DURATION": ("audio", "duration_seconds"),
            "AI_MUSIC_NUM_TRACKS": ("ai_generation", "num_tracks"),
            "AI_MUSIC_MAX_RESULTS": ("human_extraction", "max_results_per_term"),
            "AI_MUSIC_OUTPUT_DIR": ("dataset", "output_dir"),
        }

        for env_var, (section, field) in legacy_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert to appropriate type
                    if field in [
                        "duration_seconds",
                        "num_tracks",
                        "max_results_per_term",
                    ]:
                        value = int(value)
                    elif field in ["temperature", "top_p"]:
                        value = float(value)

                    section_obj = getattr(self, section)
                    setattr(section_obj, field, value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid {env_var} value: {value} - {e}")

        return self


# Global config instance
_config = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


# Quick access functions for backward compatibility
def get_audio_duration() -> int:
    """Get audio duration in seconds"""
    return get_config().audio.duration_seconds


def get_max_duration() -> int:
    """Get maximum duration for YouTube videos"""
    return get_config().audio.max_duration_seconds


def get_min_duration() -> int:
    """Get minimum duration for YouTube videos"""
    return get_config().audio.min_duration_seconds


def get_num_ai_tracks() -> int:
    """Get number of AI tracks to generate"""
    return get_config().ai_generation.num_tracks


def get_max_youtube_results() -> int:
    """Get maximum YouTube results per search term"""
    return get_config().human_extraction.max_results_per_term


def get_output_dir() -> str:
    """Get output directory"""
    return get_config().dataset.output_dir


def get_search_terms() -> List[str]:
    """Get default search terms"""
    return get_config().human_extraction.default_search_terms


def get_ai_generation_delay() -> int:
    """Get delay between AI generations"""
    return get_config().rate_limiting.ai_generation_delay


def get_youtube_download_delay() -> int:
    """Get delay between YouTube downloads"""
    return get_config().rate_limiting.youtube_download_delay


def get_youtube_search_delay() -> int:
    """Get delay between YouTube searches"""
    return get_config().rate_limiting.youtube_search_delay


def print_config():
    """Print current configuration"""
    config = get_config()
    print("\n" + "=" * 50)
    print("CURRENT CONFIGURATION")
    print("=" * 50)
    print(f"Audio Duration: {config.audio.duration_seconds} seconds")
    print(f"Max Duration: {config.audio.max_duration_seconds} seconds")
    print(f"Min Duration: {config.audio.min_duration_seconds} seconds")
    print(f"AI Tracks to Generate: {config.ai_generation.num_tracks}")
    print(
        f"Max YouTube Results per Term: {config.human_extraction.max_results_per_term}"
    )
    print(f"Output Directory: {config.dataset.output_dir}")
    print(f"AI Generation Delay: {config.rate_limiting.ai_generation_delay} seconds")
    print(
        f"YouTube Download Delay: {config.rate_limiting.youtube_download_delay} seconds"
    )
    print(f"YouTube Search Delay: {config.rate_limiting.youtube_search_delay} seconds")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
