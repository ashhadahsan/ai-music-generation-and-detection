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
        default=1800, description="Maximum duration for YouTube videos (30 minutes)"
    )
    min_duration_seconds: int = Field(
        default=10, description="Minimum duration for YouTube videos"
    )
    audio_format: str = Field(default="mp3", description="Audio format")
    audio_quality: str = Field(default="192K", description="Audio quality")


class AIGenerationSettings(BaseModel):
    """AI music generation settings"""

    num_tracks: int = Field(default=25, description="Number of AI tracks to generate")
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
    max_total_samples: int = Field(
        default=25, description="Maximum total human samples to extract"
    )
    delay_between_extractions: int = Field(
        default=2, description="Seconds to wait between extractions"
    )
    delay_between_searches: int = Field(
        default=1, description="Seconds to wait between searches"
    )
    default_search_terms: List[str] = Field(
        default=[
            "classical short",
            "jazz instrumental short",
            "piano solo short",
            "guitar instrumental short",
            "acoustic guitar short",
            "violin solo short",
            "saxophone solo short",
            "flute solo short",
            "cello solo short",
            "trumpet solo short",
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
    auto_upload: bool = Field(
        default=True,
        description="Automatically upload to Hugging Face after generation",
    )
    dataset_name: str = Field(
        default="", description="Custom dataset name (empty for auto-generated)"
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


class ExtractionModeSettings(BaseModel):
    """Extraction mode settings"""

    mode: str = Field(
        default="both", description="Extraction mode: 'ai', 'human', or 'both'"
    )
    custom_search_terms: List[str] = Field(
        default=[],
        description="Custom search terms for human extraction (overrides default if provided)",
    )


class Config(BaseSettings):
    """Main configuration class using Pydantic Settings"""

    # API Keys
    replicate_api_token: str = Field(default="", description="Replicate API token")
    hugging_face_api_key: str = Field(default="", description="Hugging Face API key")
    free_sound_api_key: str = Field(default="", description="FreeSound API key")
    free_sound_secret_key: str = Field(default="", description="FreeSound secret key")

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
    extraction_mode: ExtractionModeSettings = Field(
        default_factory=ExtractionModeSettings
    )

    class Config:
        env_prefix = (
            "AI_MUSIC_"  # Environment variables will be prefixed with AI_MUSIC_
        )
        env_nested_delimiter = "__"  # Use double underscore for nested settings
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields that aren't defined

    @model_validator(mode="after")
    def apply_legacy_overrides(self):
        """Apply legacy environment variable overrides"""
        # Legacy environment variable mappings
        legacy_mappings = {
            "AI_MUSIC_DURATION": ("audio", "duration_seconds"),
            "AI_MUSIC_NUM_TRACKS": ("ai_generation", "num_tracks"),
            "AI_MUSIC_MAX_RESULTS": ("human_extraction", "max_results_per_term"),
            "AI_MUSIC_OUTPUT_DIR": ("dataset", "output_dir"),
            "AI_MUSIC_EXTRACTION_MODE": ("extraction_mode", "mode"),
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
                    elif field == "mode":
                        # Validate mode value
                        if value not in ["ai", "human", "both"]:
                            print(
                                f"Warning: Invalid extraction mode '{value}'. Using 'both'."
                            )
                            value = "both"

                    section_obj = getattr(self, section)
                    setattr(section_obj, field, value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid {env_var} value: {value} - {e}")

        return self


# Create a global config instance for easy access
config = Config()
