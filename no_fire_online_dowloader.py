#!pip install bing-image-downloader

from bing_image_downloader import downloader

# Create dataset folders
downloader.download("forest no fire", limit=100, output_dir="fire_dataset/train/no_fire", adult_filter_off=True, force_replace=False, timeout=60)
downloader.download("city street", limit=100, output_dir="fire_dataset/train/no_fire", adult_filter_off=True, force_replace=False, timeout=60)