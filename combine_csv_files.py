#!/usr/bin/env python3
"""
Combine all CSV files in dataverse_files/xdata/ into a single CSV file.
Handles large files efficiently by processing one at a time.
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


def combine_csv_polars(csv_files, output_file):
    """
    Combine CSV files using Polars (fastest, most memory-efficient).

    Args:
        csv_files: List of CSV file paths
        output_file: Output file path
    """
    print(f"Combining {len(csv_files)} CSV files using Polars...")
    print(f"Output file: {output_file}")
    print("=" * 60)

    total_rows = 0

    # Open output file for writing
    with open(output_file, 'w', newline='') as f_out:
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] Reading {csv_file.name}...", end=" ")

            try:
                # Read CSV
                df = pl.read_csv(csv_file)
                row_count = df.height
                total_rows += row_count

                print(f"({row_count:,} rows)")

                # Write to output file
                if i == 1:
                    # First file: write with header
                    df.write_csv(f_out)
                else:
                    # Subsequent files: append without header by converting to CSV string
                    csv_string = df.write_csv()
                    # Skip the header line and write the rest
                    lines = csv_string.strip().split('\n')
                    f_out.write('\n'.join(lines[1:]) + '\n')

                # Clean up memory
                del df

            except Exception as e:
                print(f"  ✗ Error reading {csv_file.name}: {e}")
                raise

    print("=" * 60)
    print(f"✓ Successfully combined {len(csv_files)} files")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Output: {output_file}")


def combine_csv_pandas(csv_files, output_file):
    """
    Combine CSV files using Pandas with chunked processing.

    Args:
        csv_files: List of CSV file paths
        output_file: Output file path
    """
    print(f"Combining {len(csv_files)} CSV files using Pandas...")
    print(f"Output file: {output_file}")
    print("=" * 60)

    total_rows = 0
    first_file = True

    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Reading {csv_file.name}...", end=" ")

        try:
            # Read CSV
            df = pd.read_csv(csv_file, low_memory=False)
            row_count = len(df)
            total_rows += row_count

            print(f"({row_count:,} rows)")

            # Write to output file
            if first_file:
                # First file: create new file with header
                df.to_csv(output_file, index=False, mode='w')
                first_file = False
            else:
                # Subsequent files: append without header
                df.to_csv(output_file, index=False, mode='a', header=False)

            # Clean up memory
            del df

        except Exception as e:
            print(f"  ✗ Error reading {csv_file.name}: {e}")
            raise

    print("=" * 60)
    print(f"✓ Successfully combined {len(csv_files)} files")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Output: {output_file}")


def main():
    # Path to the xdata directory
    xdata_dir = Path("dataverse_files/xdata")

    if not xdata_dir.exists():
        print(f"Error: Directory '{xdata_dir}' does not exist.")
        sys.exit(1)

    # Find all CSV files
    csv_files = sorted(xdata_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{xdata_dir}'")
        sys.exit(1)

    # Output file path
    output_file = Path("dataverse_files/combined_data.csv")

    # Check if output file already exists
    if output_file.exists():
        response = input(f"\n⚠️  Output file '{output_file}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        print()

    print(f"Found {len(csv_files)} CSV file(s) to combine")
    print(f"Using: {'Polars (high-performance)' if USE_POLARS else 'Pandas (standard)'}")
    print()

    # Combine files
    try:
        if USE_POLARS:
            combine_csv_polars(csv_files, output_file)
        else:
            combine_csv_pandas(csv_files, output_file)
    except Exception as e:
        print(f"\n✗ Failed to combine CSV files: {e}")
        sys.exit(1)

    # Show output file size
    output_size = output_file.stat().st_size / (1024 * 1024)
    print(f"  Output size: {output_size:.2f} MB")


if __name__ == "__main__":
    main()
