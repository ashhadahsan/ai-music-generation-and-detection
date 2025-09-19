"""
Simple Music Extractor - YouTube Only
Extracts music from YouTube URLs using yt-dlp for training AI vs Human music classifier
"""

import os
import json
import time
import yt_dlp
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv
import subprocess
import sys
import requests
import re

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleMusicExtractor:
    def __init__(self, output_dir="music_dataset"):
        """
        Initialize the simple music extractor (YouTube only)

        Args:
            output_dir (str): Directory to save extracted music and metadata
        """
        self.output_dir = Path(output_dir)
        self.human_dir = self.output_dir / "human_created"
        self.metadata_file = self.output_dir / "extraction_metadata.json"

        # Create directories
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
            "human_created": [],
            "extraction_stats": {
                "total_attempts": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
            },
        }

    def save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def extract_from_youtube(self, urls, max_duration=30):
        """
        Extract music from YouTube URLs using yt-dlp

        Args:
            urls (list): List of YouTube URLs
            max_duration (int): Maximum duration in seconds to extract
        """
        logger.info(f"Starting YouTube extraction for {len(urls)} URLs")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.human_dir / "%(title)s.%(ext)s"),
            "extractaudio": True,
            "audioformat": "mp3",
            "audioquality": "192K",
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
                        self.metadata["extraction_stats"]["successful_extractions"] += 1
                        successful_extractions += 1

                        logger.info(f"Successfully extracted: {new_filename}")

            except Exception as e:
                logger.error(f"Error extracting from {url}: {str(e)}")
                self.metadata["extraction_stats"]["failed_extractions"] += 1

            self.metadata["extraction_stats"]["total_attempts"] += 1

            # Rate limiting
            if i < len(urls):
                logger.info("Waiting 2 seconds before next extraction...")
                time.sleep(2)

        self.save_metadata()
        logger.info(
            f"YouTube extraction complete! {successful_extractions}/{len(urls)} successful"
        )

    def search_youtube_music(self, search_terms, max_results=10):
        """
        Search YouTube for music videos and return URLs

        Args:
            search_terms (list): List of search terms for music
            max_results (int): Maximum number of results per search term

        Returns:
            list: List of YouTube URLs
        """
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
                                    # Filter for reasonable duration (10 seconds to 5 minutes)
                                    if 10 <= duration <= 300:
                                        all_urls.append(url)
                                        logger.info(f"Found: {title} ({duration}s)")
                                    else:
                                        logger.info(
                                            f"Skipped {title} - duration {duration}s"
                                        )
                                else:
                                    logger.warning(f"No URL found for: {title}")

            except Exception as e:
                logger.error(f"Error searching for {search_term}: {str(e)}")

            # Rate limiting between searches
            time.sleep(1)

        logger.info(f"Found {len(all_urls)} total YouTube URLs")
        return all_urls

    def get_sample_youtube_urls(self):
        """Get sample YouTube URLs for testing (you should replace with your own)"""
        return [
            # Add your YouTube music URLs here
            # Example: "https://www.youtube.com/watch?v=VIDEO_ID",
        ]

    def print_extraction_stats(self):
        """Print extraction statistics"""
        stats = self.metadata["extraction_stats"]

        print("\n" + "=" * 50)
        print("MUSIC EXTRACTION STATISTICS")
        print("=" * 50)
        print(f"Human-Created Tracks: {len(self.metadata['human_created'])}")
        print(f"\nExtraction Statistics:")
        print(f"  Total Attempts: {stats['total_attempts']}")
        print(f"  Successful: {stats['successful_extractions']}")
        print(f"  Failed: {stats['failed_extractions']}")
        if stats["total_attempts"] > 0:
            print(
                f"  Success Rate: {stats['successful_extractions']/stats['total_attempts']*100:.1f}%"
            )
        print(f"\nExtraction Directory: {self.output_dir}")
        print("=" * 50)
