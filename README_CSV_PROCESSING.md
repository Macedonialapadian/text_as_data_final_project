# CSV Processing Scripts - High Performance Edition

Two scripts for processing and combining CSV files in `dataverse_files/xdata/` using optimized libraries.

## Scripts

1. **`process_csv_files.py`** - Trim CSV files to first 8 columns
2. **`combine_csv_files.py`** - Combine all processed CSV files into one

## Features

- **Polars (Recommended)**: Lightning-fast Rust-based engine, 5-10x faster than pandas
- **Streaming Mode**: Processes files in streaming mode for minimal memory usage
- **Automatic Fallback**: Falls back to pandas if Polars not available
- **One-by-One Processing**: Handles each file sequentially to avoid RAM overflow
- **Safe**: Creates temporary files and only replaces originals after successful processing
- **Progress Tracking**: Shows file size, row count, and processing status

## Prerequisites

- Python 3.7 or higher
- CSV files in: `dataverse_files/xdata/`

## Installation

**Option 1: Polars (Recommended - Fastest)**
```bash
pip install polars
```

**Option 2: Pandas (Fallback)**
```bash
pip install pandas
```

**Quick Install**
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Process CSV Files (Trim to 8 columns)

1. **Install dependencies**:
   ```bash
   pip install polars
   ```

2. **Ensure your CSV files are in the correct location**:
   ```bash
   # Create the directory if needed
   mkdir -p dataverse_files/xdata

   # Place your CSV files there
   # (Download or move them to dataverse_files/xdata/)
   ```

3. **Run the processing script**:
   ```bash
   python3 process_csv_files.py
   ```

   This will trim all CSV files to keep only the first 8 columns.

### Step 2: Combine All CSV Files (Optional)

After processing, you can combine all CSV files into one:

```bash
python3 combine_csv_files.py
```

This will:
- Read all CSV files from `dataverse_files/xdata/`
- Combine them into a single file: `dataverse_files/combined_data.csv`
- Process files one-by-one to minimize memory usage
- Show progress and total row count

## What It Does

1. Scans `dataverse_files/xdata/` for all CSV files
2. For each CSV file:
   - Uses **Polars streaming mode** or **Pandas chunking** (memory efficient)
   - Keeps only the first 8 columns
   - Writes to a temporary file
   - Replaces the original file after successful processing
3. Processes files one-by-one to avoid RAM overflow
4. Shows progress and summary statistics

## Example Output

**With Polars (Recommended):**
```
Found 3 CSV file(s) to process
Using: Polars (high-performance)
============================================================

[1/3] data1.csv
Processing with Polars: dataverse_files/xdata/data1.csv
  File size: 125.45 MB
  Processed 250000 rows
  ✓ Successfully updated dataverse_files/xdata/data1.csv

[2/3] data2.csv
Processing with Polars: dataverse_files/xdata/data2.csv
  File size: 98.32 MB
  Processed 180000 rows
  ✓ Successfully updated dataverse_files/xdata/data2.csv

============================================================
Processing complete!
Successfully processed: 3/3 files
```

**With Pandas (Fallback):**
```
Found 3 CSV file(s) to process
Using: Pandas (standard)
============================================================

[1/3] data1.csv
Processing with Pandas: dataverse_files/xdata/data1.csv
  File size: 125.45 MB
  Processed 250000 rows... Done!
  ✓ Successfully updated dataverse_files/xdata/data1.csv
```

## Safety

- Original files are only replaced after successful processing
- If an error occurs, the temporary file is cleaned up
- Original file remains untouched if processing fails

## Performance Comparison

For large CSV files (100+ MB):

| Library | Speed | Memory Usage | Best For |
|---------|-------|--------------|----------|
| **Polars** | 5-10x faster | Very Low | Large files, best overall |
| Pandas | Standard | Medium | Compatibility, familiar API |

## Why Polars?

- **Written in Rust**: Native performance
- **Lazy Evaluation**: Only processes what's needed
- **Streaming**: Handles files larger than RAM
- **Parallel Processing**: Multi-threaded by default
- **Zero-Copy**: Minimal memory overhead

## Troubleshooting

**"Neither polars nor pandas is installed"**
- Install Polars: `pip install polars` (recommended)
- Or install Pandas: `pip install pandas`

**"Directory 'dataverse_files/xdata' does not exist"**
- Create the directory: `mkdir -p dataverse_files/xdata`
- Move your CSV files into it

**"No CSV files found"**
- Check that your files have `.csv` extension
- Verify they're in the correct directory

**Out of memory errors**
- Polars streaming mode should handle very large files
- If using pandas, the script processes in 50,000-row chunks
- Process files one at a time (script does this automatically)
