"""
AI Music Dataset Generator using Replicate Meta MusicGen
Creates labeled dataset for AI vs Human music classification
"""

import os
import json
import time
import requests
import replicate
import yt_dlp
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import HfApi, login
import subprocess
import sys
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    get_config,
    get_audio_duration,
    get_max_duration,
    get_min_duration,
    get_num_ai_tracks,
    get_max_youtube_results,
    get_output_dir,
    get_search_terms,
    get_ai_generation_delay,
    get_youtube_download_delay,
    get_youtube_search_delay,
)

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MusicDatasetGenerator:
    def __init__(self, replicate_token, output_dir=None):
        """
        Initialize the music dataset generator

        Args:
            replicate_token (str): Your Replicate API token
            output_dir (str): Directory to save generated music and metadata (uses config if None)
        """
        self.client = replicate.Client(api_token=replicate_token)
        self.config = get_config()

        # Use config output_dir if not provided
        if output_dir is None:
            output_dir = get_output_dir()

        self.output_dir = Path(output_dir)
        self.ai_dir = self.output_dir / self.config["dataset"]["ai_subdir"]
        self.human_dir = self.output_dir / self.config["dataset"]["human_subdir"]
        self.metadata_file = (
            self.output_dir / self.config["dataset"]["metadata_filename"]
        )

        # Create directories
        self.ai_dir.mkdir(parents=True, exist_ok=True)
        self.human_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata or create new
        self.metadata = self.load_metadata()

        # Check if yt-dlp is installed
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import yt_dlp

            logger.info("yt-dlp is available")
        except ImportError:
            logger.error("yt-dlp not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            logger.info("yt-dlp installed successfully")

    def load_metadata(self):
        """Load existing metadata or create new structure"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {
            "ai_generated": [],
            "human_created": [],
            "generation_stats": {
                "total_generated": 0,
                "successful_generations": 0,
                "failed_generations": 0,
            },
        }

    def save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def generate_music_prompts(self, num_prompts=50):
        """
        Generate diverse music prompts for AI generation

        Args:
            num_prompts (int): Number of prompts to generate

        Returns:
            list: List of music generation prompts
        """
        # Genre-based prompts
        genres = [
            "pop",
            "rock",
            "jazz",
            "classical",
            "electronic",
            "hip-hop",
            "country",
            "blues",
            "reggae",
            "folk",
            "ambient",
            "techno",
            "r&b",
            "soul",
            "funk",
            "disco",
            "house",
            "trance",
        ]

        # Mood/emotion prompts
        moods = [
            "upbeat and energetic",
            "calm and peaceful",
            "dark and mysterious",
            "romantic and dreamy",
            "aggressive and intense",
            "melancholic and sad",
            "cheerful and bright",
            "atmospheric and ethereal",
            "groovy and rhythmic",
        ]

        # Instrument combinations
        instruments = [
            "piano and strings",
            "guitar and drums",
            "synthesizer and bass",
            "violin and cello",
            "saxophone and piano",
            "electric guitar solo",
            "orchestral arrangement",
            "acoustic guitar",
            "electronic beats",
        ]

        # Tempo descriptions
        tempos = [
            "slow tempo",
            "medium tempo",
            "fast tempo",
            "variable tempo",
            "120 BPM",
            "80 BPM",
            "140 BPM",
            "uptempo",
        ]

        prompts = []

        # Generate diverse combinations
        for i in range(num_prompts):
            # Choose random elements
            import random

            genre = random.choice(genres)
            mood = random.choice(moods)
            instrument = random.choice(instruments)
            tempo = random.choice(tempos)

            # Create varied prompt structures
            prompt_templates = [
                f"A {mood} {genre} song with {instrument}, {tempo}",
                f"{genre.capitalize()} music that is {mood}, featuring {instrument}",
                f"Instrumental {genre} track, {mood} mood, {tempo}",
                f"Create a {tempo} {genre} piece with {instrument}, very {mood}",
                f"{mood.capitalize()} {genre} composition using {instrument}",
            ]

            prompt = random.choice(prompt_templates)
            prompts.append(prompt)

        return prompts

    def generate_ai_music(self, prompt, filename_prefix="ai_track", duration=None):
        """
        Generate AI music using Replicate Meta MusicGen

        Args:
            prompt (str): Text prompt for music generation
            filename_prefix (str): Prefix for the generated file
            duration (int): Duration in seconds (uses config if None)

        Returns:
            dict: Generation result with file path and metadata
        """
        try:
            # Use config duration if not provided
            if duration is None:
                duration = get_audio_duration()

            logger.info(
                f"Generating music with prompt: '{prompt}' (duration: {duration}s)"
            )

            # Get AI generation config
            ai_config = self.config["ai_generation"]

            # Call Meta MusicGen model
            output = self.client.run(
                "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
                input={
                    "top_k": ai_config["top_k"],
                    "top_p": ai_config["top_p"],
                    "prompt": prompt,
                    "duration": duration,
                    "temperature": ai_config["temperature"],
                    "continuation": False,
                    "model_version": ai_config["model_version"],
                    "output_format": self.config["audio"]["audio_format"],
                    "continuation_start": 0,
                    "multi_band_diffusion": False,
                    "normalization_strategy": ai_config["normalization_strategy"],
                    "classifier_free_guidance": ai_config["classifier_free_guidance"],
                },
            )

            # Save the audio file directly from the output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.mp3"
            file_path = self.ai_dir / filename

            # Write the file to disk
            with open(file_path, "wb") as file:
                file.write(output.read())

            # Create metadata entry
            metadata_entry = {
                "filename": filename,
                "file_path": str(file_path),
                "prompt": prompt,
                "duration": duration,
                "generation_timestamp": timestamp,
                "model": "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
                "label": "ai_generated",
            }

            self.metadata["ai_generated"].append(metadata_entry)
            self.metadata["generation_stats"]["successful_generations"] += 1

            logger.info(f"Successfully generated and saved: {filename}")
            return metadata_entry

        except Exception as e:
            logger.error(f"Error generating music: {str(e)}")
            self.metadata["generation_stats"]["failed_generations"] += 1
            return None

    def batch_generate_ai_music(self, num_tracks=None, delay_between=None):
        """
        Generate multiple AI music tracks

        Args:
            num_tracks (int): Number of tracks to generate (uses config if None)
            delay_between (int): Delay in seconds between generations (uses config if None)
        """
        # Use config values if not provided
        if num_tracks is None:
            num_tracks = get_num_ai_tracks()
        if delay_between is None:
            delay_between = get_ai_generation_delay()

        logger.info(f"Starting batch generation of {num_tracks} AI music tracks")

        prompts = self.generate_music_prompts(num_tracks)
        successful_generations = 0

        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Generating track {i}/{num_tracks}")

            result = self.generate_ai_music(
                prompt=prompt, filename_prefix=f"ai_track_{i:03d}"
            )

            if result:
                successful_generations += 1

            # Update total counter
            self.metadata["generation_stats"]["total_generated"] += 1

            # Save metadata periodically
            if i % 5 == 0:
                self.save_metadata()
                logger.info(
                    f"Progress: {successful_generations}/{i} successful generations"
                )

            # Rate limiting
            if i < len(prompts):
                logger.info(
                    f"Waiting {delay_between} seconds before next generation..."
                )
                time.sleep(delay_between)

        # Final save
        self.save_metadata()

        logger.info(f"Batch generation complete!")
        logger.info(f"Successful: {successful_generations}/{num_tracks}")
        logger.info(f"Total AI tracks: {len(self.metadata['ai_generated'])}")

    def search_youtube_music(self, search_terms, max_results=None):
        """
        Search YouTube for music videos and return URLs

        Args:
            search_terms (list): List of search terms for music
            max_results (int): Maximum number of results per search term (uses config if None)

        Returns:
            list: List of YouTube URLs
        """
        # Use config value if not provided
        if max_results is None:
            max_results = get_max_youtube_results()

        logger.info(f"Searching YouTube for music with {len(search_terms)} terms")

        all_urls = []

        for search_term in search_terms:
            try:
                logger.info(f"Searching for: {search_term}")

                # Use yt-dlp to search YouTube
                search_query = f"{search_term} music"

                # Create a temporary yt-dlp instance for searching
                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "extract_flat": True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Search for videos using the correct syntax
                    search_results = ydl.extract_info(
                        f"ytsearch{max_results}:{search_query}", download=False
                    )

                    if search_results and "entries" in search_results:
                        for entry in search_results["entries"]:
                            if entry:
                                # Get URL from different possible fields
                                url = (
                                    entry.get("webpage_url")
                                    or entry.get("url")
                                    or entry.get("id")
                                )
                                title = entry.get("title", "Unknown")
                                duration = entry.get("duration", 0)

                                # If we have an ID but no full URL, construct it
                                if not url and entry.get("id"):
                                    url = (
                                        f"https://www.youtube.com/watch?v={entry['id']}"
                                    )

                                if url:
                                    # Filter for reasonable duration using config
                                    min_duration = get_min_duration()
                                    max_duration = get_max_duration()
                                    if min_duration <= duration <= max_duration:
                                        all_urls.append(url)
                                        logger.info(f"Found: {title} ({duration}s)")
                                    else:
                                        logger.info(
                                            f"Skipped {title} - duration {duration}s (outside {min_duration}-{max_duration}s range)"
                                        )
                                else:
                                    logger.warning(f"No URL found for: {title}")

            except Exception as e:
                logger.error(f"Error searching for {search_term}: {str(e)}")

            # Rate limiting between searches using config
            time.sleep(get_youtube_search_delay())

        logger.info(f"Found {len(all_urls)} total YouTube URLs")
        return all_urls

    def extract_from_youtube(self, urls, max_duration=None):
        """
        Extract music from YouTube URLs using yt-dlp

        Args:
            urls (list): List of YouTube URLs
            max_duration (int): Maximum duration in seconds to extract (uses config if None)
        """
        # Use config value if not provided
        if max_duration is None:
            max_duration = get_audio_duration()

        logger.info(f"Starting YouTube extraction for {len(urls)} URLs")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.human_dir / "%(title)s.%(ext)s"),
            "extractaudio": True,
            "audioformat": self.config["audio"]["audio_format"],
            "audioquality": self.config["audio"]["audio_quality"],
            "noplaylist": True,
            "max_duration": max_duration,
            "writesubtitles": False,
            "writeautomaticsub": False,
        }

        successful_extractions = 0

        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"Extracting {i}/{len(urls)}: {url}")

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Get video info first
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title", f"youtube_track_{i}")
                    duration = info.get("duration", 0)

                    # Skip if too long
                    if duration > max_duration:
                        logger.warning(
                            f"Skipping {title} - duration {duration}s > {max_duration}s"
                        )
                        continue

                    # Download the audio
                    ydl.download([url])

                    # Find the downloaded file
                    downloaded_files = list(self.human_dir.glob(f"{title}*"))
                    if downloaded_files:
                        downloaded_file = downloaded_files[0]

                        # Rename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_filename = f"human_youtube_{i:03d}_{timestamp}.mp3"
                        new_path = self.human_dir / new_filename
                        downloaded_file.rename(new_path)

                        # Create metadata entry
                        metadata_entry = {
                            "filename": new_filename,
                            "file_path": str(new_path),
                            "source": "youtube",
                            "source_url": url,
                            "original_title": title,
                            "duration": duration,
                            "extraction_timestamp": timestamp,
                            "label": "human_created",
                        }

                        self.metadata["human_created"].append(metadata_entry)
                        self.metadata["generation_stats"]["successful_generations"] += 1
                        successful_extractions += 1

                        logger.info(f"Successfully extracted: {new_filename}")

            except Exception as e:
                logger.error(f"Error extracting from {url}: {str(e)}")
                self.metadata["generation_stats"]["failed_generations"] += 1

            self.metadata["generation_stats"]["total_generated"] += 1

            # Rate limiting using config
            if i < len(urls):
                delay = get_youtube_download_delay()
                logger.info(f"Waiting {delay} seconds before next extraction...")
                time.sleep(delay)

        self.save_metadata()
        logger.info(
            f"YouTube extraction complete! {successful_extractions}/{len(urls)} successful"
        )

    def extract_human_music_samples(
        self, search_terms=None, max_results=None, max_duration=None
    ):
        """
        Extract human-created music samples using YouTube search and extraction

        Args:
            search_terms (list): List of search terms for music (uses config if None)
            max_results (int): Maximum number of results per search term (uses config if None)
            max_duration (int): Maximum duration in seconds to extract (uses config if None)
        """
        logger.info("Starting human music extraction using YouTube search")

        # Use config values if not provided
        if search_terms is None:
            search_terms = get_search_terms()
        if max_results is None:
            max_results = get_max_youtube_results()
        if max_duration is None:
            max_duration = get_audio_duration()

        # Search for YouTube URLs
        youtube_urls = self.search_youtube_music(search_terms, max_results)

        if youtube_urls:
            # Extract music from found URLs
            self.extract_from_youtube(youtube_urls, max_duration)
        else:
            logger.warning("No YouTube URLs found for extraction")

        logger.info("Human music extraction completed")

    def get_sample_search_terms(self):
        """Get sample search terms for music extraction"""
        return get_search_terms()

    def create_training_csv(self, save_local=True):
        """Create a CSV file for training with file paths and labels"""
        data = []

        # Add AI generated tracks
        for entry in self.metadata["ai_generated"]:
            data.append(
                {
                    "file_path": entry["file_path"],
                    "filename": entry["filename"],
                    "label": 0,  # 0 for AI-generated
                    "label_text": "ai_generated",
                    "prompt": entry.get("prompt", ""),
                    "duration": entry.get("duration", 30),
                    "model": entry.get("model", ""),
                    "generation_timestamp": entry.get("generation_timestamp", ""),
                }
            )

        # Add human created tracks
        for entry in self.metadata["human_created"]:
            data.append(
                {
                    "file_path": entry["file_path"],
                    "filename": entry["filename"],
                    "label": 1,  # 1 for human-created
                    "label_text": "human_created",
                    "prompt": "",
                    "duration": entry.get("duration", 30),
                    "model": "",
                    "generation_timestamp": entry.get("extraction_timestamp", ""),
                    "source": entry.get("source", ""),
                    "original_title": entry.get("original_title", ""),
                }
            )

        # Create DataFrame
        df = pd.DataFrame(data)

        if save_local:
            # Save CSV locally with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"dataset_{timestamp}.csv"
            df.to_csv(csv_path, index=False)

            # Also save a latest version
            latest_csv_path = self.output_dir / "dataset_latest.csv"
            df.to_csv(latest_csv_path, index=False)

            logger.info(f"Training CSV created: {csv_path}")
            logger.info(f"Latest CSV saved: {latest_csv_path}")

        logger.info(f"Dataset summary:")
        logger.info(f"  AI-generated tracks: {len(self.metadata['ai_generated'])}")
        logger.info(f"  Human-created tracks: {len(self.metadata['human_created'])}")
        logger.info(f"  Total tracks: {len(data)}")

        return df

    def upload_to_huggingface(
        self, dataset_name=None, private=True, push_audio_files=True
    ):
        """
        Upload dataset to Hugging Face Hub

        Args:
            dataset_name (str): Name for the dataset on Hugging Face (default: auto-generated)
            private (bool): Whether to make the dataset private
            push_audio_files (bool): Whether to upload audio files (requires git-lfs)

        Returns:
            str: URL of the uploaded dataset
        """
        try:
            # Get Hugging Face API key
            hf_token = os.getenv("HUGGING_FACE_API_KEY")
            if not hf_token:
                logger.error("HUGGING_FACE_API_KEY not found in environment variables")
                return None

            # Login to Hugging Face
            login(token=hf_token)
            api = HfApi()

            # Generate dataset name if not provided
            if not dataset_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"ai-music-dataset-{timestamp}"

            # Create the dataset
            logger.info(f"Creating Hugging Face dataset: {dataset_name}")

            # Create dataset metadata
            dataset_info = {
                "dataset_name": dataset_name,
                "description": "AI vs Human Music Classification Dataset",
                "created_by": "AI Music Dataset Generator",
                "total_samples": len(self.metadata["ai_generated"])
                + len(self.metadata["human_created"]),
                "ai_generated_samples": len(self.metadata["ai_generated"]),
                "human_created_samples": len(self.metadata["human_created"]),
                "creation_timestamp": datetime.now().isoformat(),
                "model_used": "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            }

            # Create dataset directory structure
            hf_dataset_dir = self.output_dir / "huggingface_dataset"
            hf_dataset_dir.mkdir(exist_ok=True)

            # Save dataset info
            with open(hf_dataset_dir / "dataset_info.json", "w") as f:
                json.dump(dataset_info, f, indent=2)

            # Create the CSV for Hugging Face
            df = self.create_training_csv(save_local=False)
            csv_path = hf_dataset_dir / "dataset.csv"
            df.to_csv(csv_path, index=False)

            # Create README.md
            readme_content = f"""---
license: mit
task_categories:
- audio-classification
- music-generation
language:
- en
tags:
- music
- ai-generated
- classification
- audio
size_categories:
- 1K<n<10K
---

# AI vs Human Music Classification Dataset

## Dataset Description

This dataset contains music samples for training AI vs Human music classification models.

- **Total Samples**: {dataset_info['total_samples']}
- **AI Generated**: {dataset_info['ai_generated_samples']} samples
- **Human Created**: {dataset_info['human_created_samples']} samples
- **Model Used**: {dataset_info['model_used']}
- **Created**: {dataset_info['creation_timestamp']}

## Dataset Structure

- `dataset.csv`: Main dataset file with file paths and labels
- `ai_generated/`: Directory containing AI-generated music samples
- `human_created/`: Directory containing human-created music samples
- `dataset_info.json`: Dataset metadata and statistics

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_name}")

# Access the data
train_data = dataset['train']
```

## Labels

- `0`: AI-generated music
- `1`: Human-created music

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{ai_music_dataset_{timestamp},
  title={{AI vs Human Music Classification Dataset}},
  author={{AI Music Dataset Generator}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{dataset_name}}}
}}
```
"""

            with open(hf_dataset_dir / "README.md", "w") as f:
                f.write(readme_content)

            # Copy audio files if requested
            if push_audio_files:
                logger.info("Copying audio files to Hugging Face dataset directory...")

                # Copy AI generated files
                ai_hf_dir = hf_dataset_dir / "ai_generated"
                ai_hf_dir.mkdir(exist_ok=True)

                for entry in self.metadata["ai_generated"]:
                    src_path = Path(entry["file_path"])
                    if src_path.exists():
                        dst_path = ai_hf_dir / entry["filename"]
                        import shutil

                        shutil.copy2(src_path, dst_path)

                # Copy human created files
                human_hf_dir = hf_dataset_dir / "human_created"
                human_hf_dir.mkdir(exist_ok=True)

                for entry in self.metadata["human_created"]:
                    src_path = Path(entry["file_path"])
                    if src_path.exists():
                        dst_path = human_hf_dir / entry["filename"]
                        import shutil

                        shutil.copy2(src_path, dst_path)

            # Create Hugging Face dataset
            try:
                # Create repository
                api.create_repo(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    private=private,
                    exist_ok=True,
                )

                # Upload files
                logger.info("Uploading files to Hugging Face...")
                api.upload_folder(
                    folder_path=str(hf_dataset_dir),
                    repo_id=dataset_name,
                    repo_type="dataset",
                    commit_message=f"Upload AI music dataset - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )

                dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
                logger.info(f"Dataset successfully uploaded to: {dataset_url}")

                return dataset_url

            except Exception as e:
                logger.error(f"Error uploading to Hugging Face: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in Hugging Face upload: {str(e)}")
            return None

    def print_dataset_stats(self):
        """Print dataset statistics"""
        stats = self.metadata["generation_stats"]

        print("\n" + "=" * 50)
        print("MUSIC DATASET STATISTICS")
        print("=" * 50)
        print(f"AI-Generated Tracks: {len(self.metadata['ai_generated'])}")
        print(f"Human-Created Tracks: {len(self.metadata['human_created'])}")
        print(
            f"Total Tracks: {len(self.metadata['ai_generated']) + len(self.metadata['human_created'])}"
        )
        print(f"\nGeneration Statistics:")
        print(f"  Total Attempts: {stats['total_generated']}")
        print(f"  Successful: {stats['successful_generations']}")
        print(f"  Failed: {stats['failed_generations']}")
        print(
            f"  Success Rate: {stats['successful_generations']/max(stats['total_generated'],1)*100:.1f}%"
        )
        print(f"\nDataset Directory: {self.output_dir}")
        print("=" * 50)


def main():
    """Main function to demonstrate usage"""

    # Set your Replicate API token
    REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

    if not REPLICATE_TOKEN:
        print("ERROR: Please set your REPLICATE_API_TOKEN environment variable")
        print("Get your token from: https://replicate.com/account")
        return

    # Initialize generator
    generator = MusicDatasetGenerator(
        replicate_token=REPLICATE_TOKEN, output_dir="ai_music_dataset"
    )

    print("AI Music Dataset Generator")
    print("=" * 50)
    print("1. Generate AI music samples only")
    print("2. Extract human music samples only (YouTube search)")
    print("3. Full dataset generation (AI + Human)")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice in ["1", "3"]:
        # Generate AI music dataset
        print("Starting AI music generation...")
        generator.batch_generate_ai_music()

    if choice in ["2", "3"]:
        # Extract human music samples using YouTube search
        print("\n🎵 Human Music Extraction (YouTube Search)")
        print("Enter music search terms (comma-separated):")
        print("Examples: classical, jazz, piano, guitar, orchestral")

        search_input = input("Search terms: ").strip()
        if search_input:
            search_terms = [term.strip() for term in search_input.split(",")]
            max_results_input = input(
                f"Max results per term (default {get_max_youtube_results()}): "
            ).strip()
            max_results = int(max_results_input) if max_results_input else None

            print(f"\n🔍 Searching and extracting human music...")
            generator.extract_human_music_samples(
                search_terms=search_terms, max_results=max_results
            )
        else:
            print("Using default search terms...")
            generator.extract_human_music_samples()

    # Create training CSV
    df = generator.create_training_csv()

    # Print statistics
    generator.print_dataset_stats()

    # Ask about Hugging Face upload
    hf_choice = input("\nUpload to Hugging Face? (y/n): ").strip().lower()
    if hf_choice in ["y", "yes"]:
        hf_token = os.getenv("HUGGING_FACE_API_KEY")
        if not hf_token:
            print("HUGGING_FACE_API_KEY not found in environment variables")
            print("Please set your Hugging Face API key to upload the dataset")
        else:
            dataset_name = input(
                "Enter dataset name (or press Enter for auto-generated): "
            ).strip()
            if not dataset_name:
                dataset_name = None

            private = input("Make dataset private? (y/n, default: y): ").strip().lower()
            private = private not in ["n", "no"]

            push_audio = (
                input("Upload audio files? (y/n, default: y): ").strip().lower()
            )
            push_audio = push_audio not in ["n", "no"]

            print("Uploading to Hugging Face...")
            dataset_url = generator.upload_to_huggingface(
                dataset_name=dataset_name, private=private, push_audio_files=push_audio
            )

            if dataset_url:
                print(f"✅ Dataset uploaded successfully: {dataset_url}")
            else:
                print("❌ Failed to upload dataset to Hugging Face")

    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Check the 'ai_music_dataset' folder for all generated files")
    print("2. Use the generated CSV file for training your classifier")
    print("3. The dataset is ready for AI vs Human music classification!")
    print("\nFor Replicate API token: https://replicate.com/account")
    print("For Hugging Face API key: https://huggingface.co/settings/tokens")
