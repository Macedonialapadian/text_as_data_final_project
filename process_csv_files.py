#!/usr/bin/env python3
"""
Process CSV files to keep only the first 8 columns.
High-performance implementation using Polars for optimal speed and memory efficiency.
"""

import os
import sys
from pathlib import Path

try:
    import polars as pl
    USE_POLARS = True
except ImportError:
    print("Warning: Polars not found. Install with: pip install polars")
    print("Falling back to pandas...")
    try:
        import pandas as pd
        USE_POLARS = False
    except ImportError:
        print("Error: Neither polars nor pandas is installed.")
        print("Install one with: pip install polars  (recommended)")
        print("              or: pip install pandas")
        sys.exit(1)


def process_csv_polars(file_path, num_columns=8):
    """
    Process CSV using Polars (fastest, most memory-efficient).

    Args:
        file_path: Path to the CSV file
        num_columns: Number of columns to keep (default: 8)
    """
    print(f"Processing with Polars: {file_path}")

    temp_file = f"{file_path}.tmp"

    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"  File size: {file_size / (1024*1024):.2f} MB")

        # First, read just the header to get column names
        lazy_df = pl.scan_csv(file_path)

        # Get the first N column names
        columns = lazy_df.collect_schema().names()[:num_columns]

        # Read CSV in streaming mode selecting only the first N columns
        # This is memory efficient for large files
        df = pl.scan_csv(file_path).select(columns)

        # Collect and write to temp file using streaming
        df.collect(streaming=True).write_csv(temp_file)

        # Get row count efficiently
        final_df = pl.read_csv(temp_file, n_rows=1)  # Just read first row to get structure
        row_count = pl.read_csv(temp_file).height  # Then get full count

        print(f"  Processed {row_count} rows")

        # Replace original file
        os.replace(temp_file, file_path)
        print(f"  ✓ Successfully updated {file_path}")

    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


def process_csv_pandas(file_path, num_columns=8, chunk_size=50000):
    """
    Process CSV using Pandas with chunking (fallback option).

    Args:
        file_path: Path to the CSV file
        num_columns: Number of columns to keep (default: 8)
        chunk_size: Rows per chunk (default: 50000)
    """
    print(f"Processing with Pandas: {file_path}")

    temp_file = f"{file_path}.tmp"

    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"  File size: {file_size / (1024*1024):.2f} MB")

        # Process in chunks
        first_chunk = True
        total_rows = 0

        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            # Keep only first N columns
            chunk_trimmed = chunk.iloc[:, :num_columns]

            # Write to temp file
            if first_chunk:
                chunk_trimmed.to_csv(temp_file, index=False, mode='w')
                first_chunk = False
            else:
                chunk_trimmed.to_csv(temp_file, index=False, mode='a', header=False)

            total_rows += len(chunk)
            print(f"  Processed {total_rows} rows...", end='\r')

        print(f"  Processed {total_rows} rows... Done!")

        # Replace original file
        os.replace(temp_file, file_path)
        print(f"  ✓ Successfully updated {file_path}")

    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


def process_csv_file(file_path, num_columns=8):
    """
    Process a single CSV file using the best available library.
    """
    if USE_POLARS:
        process_csv_polars(file_path, num_columns)
    else:
        process_csv_pandas(file_path, num_columns)


def main():
    # Path to the xdata directory
    xdata_dir = Path("dataverse_files/xdata")

    if not xdata_dir.exists():
        print(f"Error: Directory '{xdata_dir}' does not exist.")
        print("Please ensure the CSV files are in the correct location.")
        sys.exit(1)

    # Find all CSV files
    csv_files = sorted(xdata_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{xdata_dir}'")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) to process")
    print(f"Using: {'Polars (high-performance)' if USE_POLARS else 'Pandas (standard)'}")
    print("=" * 60)

    # Process each file one by one
    success_count = 0
    failed_files = []

    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {csv_file.name}")
        try:
            process_csv_file(csv_file, num_columns=8)
            success_count += 1
        except Exception as e:
            failed_files.append(csv_file.name)
            print(f"Failed to process {csv_file.name}, continuing with next file...")
            continue

    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(csv_files)} files")

    if failed_files:
        print(f"\nFailed files:")
        for filename in failed_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
