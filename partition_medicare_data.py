import os
import glob
import argparse
import pandas as pd

def partition_and_sort(input_dir, output_dir, chunksize=250_000):
    """
    Reads all CSVs in input_dir in manageable chunks.
    Partitions rows into output_dir/STATE/PROVIDER_TYPE.csv.
    After partitioning is completely finished, iterates over all 
    generated files and sorts them individually by HCPCS_Cd.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'")
        return

    print("Step 1 of 2: Partitioning data by State and Provider Type...")
    
    # Track the partition files we create so we know what to sort later.
    partition_files = set()

    for file in csv_files:
        print(f"  -> Processing {file} in chunks...")
        # Read dataset as string to prevent pandas from guessing types and throwing mixed-type warnings
        for chunk in pd.read_csv(file, chunksize=chunksize, dtype=str):
            
            # Fill NaNs in the partitioning columns to avoid groupby dropping them
            chunk['Rndrng_Prvdr_State_Abrvtn'] = chunk['Rndrng_Prvdr_State_Abrvtn'].fillna('UNKNOWN_STATE')
            chunk['Rndrng_Prvdr_Type'] = chunk['Rndrng_Prvdr_Type'].fillna('UNKNOWN_TYPE')
            
            # Provider types often contain spaces, slashes, or other special characters. 
            # We sanitize them to use as safe filename paths.
            safe_states = chunk['Rndrng_Prvdr_State_Abrvtn'].str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
            safe_provs  = chunk['Rndrng_Prvdr_Type'].str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
            
            # Group the chunk dataframe natively
            grouped = chunk.groupby([safe_states, safe_provs])
            
            for (state, prov), group_df in grouped:
                # Nest directory structure like: output_dir / MD / Hospitalist.csv
                state_dir = os.path.join(output_dir, state)
                os.makedirs(state_dir, exist_ok=True)
                
                out_path = os.path.join(state_dir, f"{prov}.csv")
                partition_files.add(out_path)
                
                # If the file doesn't exist yet, we add headers
                header = not os.path.exists(out_path)
                
                # Append rows to the correct state/provider file
                group_df.to_csv(out_path, mode='a', index=False, header=header)

    print(f"\nStep 2 of 2: Sorting each of the {len(partition_files)} generated partitions by HCPCS_Cd...")
    
    # Since each file only contains data for one State and Provider,
    # they are small enough to be loaded into memory and sorted instantly.
    for i, p_file in enumerate(partition_files, 1):
        if i % 100 == 0:
            print(f"  -> Sorted {i}/{len(partition_files)} files...")
            
        try:
            # Read the partition
            df = pd.read_csv(p_file, dtype=str)
            
            # Sort by visit code (HCPCS_Cd)
            # na_position='last' ensures any empty entries go to the bottom
            df = df.sort_values(by='HCPCS_Cd', na_position='last')
            
            # Overwrite the partition file with the sorted version
            df.to_csv(p_file, index=False)
            
        except Exception as e:
            print(f"Error sorting {p_file}: {e}")

    print("\n✅ Partitioning and sorting complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition Medicare data by State & Provider, then sort by Visit ID.")
    parser.add_argument("--input-dir", type=str, default="data", help="Directory containing the raw CSV pulls")
    parser.add_argument("--output-dir", type=str, default="partitioned_data", help="Directory to save the partitioned tree")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Number of rows to load sequentially to prevent out-of-memory errors")
    
    args = parser.parse_args()
    
    partition_and_sort(args.input_dir, args.output_dir, chunksize=args.chunksize)
