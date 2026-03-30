import os
import glob
import pandas as pd
import argparse

def convert_csvs_to_parquet(base_dir):
    """
    Recursively finds all CSV files in base_dir,
    reads them as strings, converts to Parquet,
    and removes the original CSV file.
    """
    # The glob pattern below will search all subdirectories
    search_pattern = os.path.join(base_dir, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in '{base_dir}'. "
              "Check that parsing is complete or verify directory.")
        return
        
    print(f"Found {len(csv_files)} CSV files to convert to Parquet.")
    
    success_count = 0
    error_count = 0
    
    for i, csv_path in enumerate(csv_files, 1):
        try:
            # Read CSV with str dtype to preserve exact CMS formatting easily
            df = pd.read_csv(csv_path, dtype=str)
            
            # Change extension
            parquet_path = os.path.splitext(csv_path)[0] + '.parquet'
            
            # Write out Parquet file
            # engine='pyarrow' or 'fastparquet' can be specified if needed
            df.to_parquet(parquet_path, index=False)
            
            # If the write succeeds safely without throwing an exception, remove original CSV
            os.remove(csv_path)
            
            success_count += 1
            if success_count % 100 == 0:
                print(f"  -> Successfully converted {success_count} files...")
                
        except Exception as e:
            print(f"❌ Error converting '{csv_path}': {e}")
            error_count += 1
            
    print("\n✅ Parquet Conversion complete!")
    print(f"Total CSVs converted and deleted: {success_count}")
    if error_count > 0:
        print(f"Total failures: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert nested CSV files into Parquet files, replacing them.")
    parser.add_argument("--dir", type=str, default="partitioned_data", help="Target dir to walk recursively.")
    args = parser.parse_args()
    
    convert_csvs_to_parquet(args.dir)
