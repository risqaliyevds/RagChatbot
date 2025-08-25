#!/usr/bin/env python3
"""
NLTK Data Downloader for Offline Deployment
===========================================

This script downloads all necessary NLTK data that might be needed by unstructured
and other document processing libraries. Run this on a machine with internet access,
then copy the nltk_data folder to your offline server.
"""

import os
import sys
import nltk
from pathlib import Path

def download_nltk_data():
    """Download all necessary NLTK data for offline use"""
    
    print("🚀 Starting NLTK data download for offline deployment...")
    
    # Set custom download directory
    download_dir = Path("./nltk_data")
    download_dir.mkdir(exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.insert(0, str(download_dir))
    
    # List of NLTK data packages that might be needed
    packages = [
        'punkt',           # Sentence tokenizer
        'punkt_tab',       # New punkt tokenizer data (for newer NLTK versions)
        'stopwords',       # Stop words
        'wordnet',         # WordNet
        'averaged_perceptron_tagger',  # POS tagger
        'vader_lexicon',   # Sentiment analysis
        'brown',           # Brown corpus
        'omw-1.4',         # Open Multilingual Wordnet
        'universal_tagset', # Universal tagset
    ]
    
    print(f"📁 Downloading NLTK data to: {download_dir.absolute()}")
    
    downloaded = []
    failed = []
    
    for package in packages:
        try:
            print(f"📦 Downloading {package}...")
            nltk.download(package, download_dir=str(download_dir), quiet=False)
            downloaded.append(package)
            print(f"✅ Successfully downloaded {package}")
        except Exception as e:
            print(f"❌ Failed to download {package}: {e}")
            failed.append(package)
    
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successfully downloaded: {len(downloaded)} packages")
    for pkg in downloaded:
        print(f"   - {pkg}")
    
    if failed:
        print(f"\n❌ Failed to download: {len(failed)} packages")
        for pkg in failed:
            print(f"   - {pkg}")
    
    print(f"\n📁 NLTK data downloaded to: {download_dir.absolute()}")
    print("\n🔧 DEPLOYMENT INSTRUCTIONS:")
    print("="*60)
    print("1. Copy the entire 'nltk_data' folder to your offline server")
    print("2. Place it in your project directory or set NLTK_DATA environment variable")
    print("3. Add this to your application startup code:")
    print()
    print("   import nltk")
    print("   import os")
    print("   nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')")
    print("   nltk.data.path.insert(0, nltk_data_path)")
    print()
    print("4. Alternative: Set environment variable NLTK_DATA=/path/to/nltk_data")
    print()
    print("🚀 Your application should now work offline!")


if __name__ == "__main__":
    try:
        download_nltk_data()
    except KeyboardInterrupt:
        print("\n\n⏸️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1) 