#!/usr/bin/env python3

"""
Extract URLs column from combined CSV and split into separate columns.

Each comma-separated URL value gets its own column.
"""

import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed.")
    print("Install with: pip install pandas")
    sys.exit(1)


def extract_urls(input_file, output_file):
    """
    Extract and split URLs using Pandas.

    Args:
        input_file: Path to combined CSV file
        output_file: Path to output CSV with split URLs
    """
    print(f"Reading data from: {input_file}")
    print("=" * 60)

    # Read the CSV
    df = pd.read_csv(input_file)

    print(f"Total rows: {len(df):,}")
    print(f"Extracting 'urls' column...")

    # Extract the urls column and split by comma
    urls_series = df['urls'].str.split(', ')

    # Find the maximum number of URLs in any row
    max_urls = urls_series.apply(lambda x: len(x) if isinstance(x, list) else 0).max()

    print(f"Maximum URLs in a single row: {max_urls}")

    # Create a dataframe with split URLs
    # Each URL gets its own column: url_1, url_2, url_3, etc.
    url_df = pd.DataFrame(
        urls_series.tolist(),
        columns=[f'url_{i+1}' for i in range(max_urls)]
    )

    print(f"Created {max_urls} URL columns")
    print(f"Writing to: {output_file}")

    # Write to CSV
    url_df.to_csv(output_file, index=False)

    # Get file size
    output_size = os.path.getsize(output_file) / (1024 * 1024)

    print("=" * 60)
    print(f"✓ Successfully extracted URLs")
    print(f"  Rows: {len(url_df):,}")
    print(f"  Columns: {len(url_df.columns)}")
    print(f"  Output size: {output_size:.2f} MB")
    print(f"  Output file: {output_file}")


def main():
    # CONFIGURE THESE PATHS based on your repository structure
    input_file = Path("data/combined_data.csv")  # UPDATE THIS PATH
    output_file = Path("urls_extracted.csv")  # UPDATE THIS PATH

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        print("\nPlease update the paths in this script:")
        print("  - input_file: Path to your combined_data.csv")
        print("  - output_file: Where to save the extracted URLs")
        sys.exit(1)

    # Check if output file already exists
    if output_file.exists():
        response = input(f"\n⚠️  Output file '{output_file}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        print()

    # Extract and split URLs
    try:
        extract_urls(input_file, output_file)
    except Exception as e:
        print(f"\n✗ Failed to extract URLs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
