import os
import requests
import csv
import time
import argparse

# Define the years and their corresponding endpoints/UUIDs
DATASETS = {
    2023: "0e9f2f2b-7bf9-451a-912c-e02e654dd725",
    2022: "e650987d-01b7-4f09-b75e-b0b075afbf98",
    2021: "31dc2c47-f297-4948-bfb4-075e1bec3a02",
    2020: "c957b49e-1323-49e7-8678-c09da387551d",
    2019: "867b8ac7-ccb7-4cc9-873d-b24340d89e32",
    2018: "fb6d9fe8-38c1-4d24-83d4-0b7b291000b2",
    2017: "85bf3c9c-2244-490d-ad7d-c34e4c28f8ea",
    2016: "7918e22a-fbfb-4a07-9f59-f8aab2b757d4",
    2015: "f8cdb11a-d5f7-4fbe-aac4-05abc8ee2c83",
    2014: "f63b48ae-946e-48f7-9f56-327a68da4e0b",
    2013: "ad5e7548-98ab-4325-af4b-b2a7099b9351"
}

BASE_API_URL = "https://data.cms.gov/data-api/v1/dataset/{uuid}/data"

def fetch_and_save_data(year, uuid, output_dir="data", batch_size=5000, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"medicare_physician_practitioners_{year}.csv")
    
    url = BASE_API_URL.format(uuid=uuid)
    offset = 0
    records_fetched = 0
    
    print(f"Starting data pull for year {year}...")
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = None
        
        while True:
            params = {
                "size": batch_size,
                "offset": offset
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for year {year} at offset {offset}: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # If no data is returned, we've hit the end
            if not data:
                break
                
            # Initialize CSV writer with headers from the first batch
            if writer is None:
                headers = list(data[0].keys())
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
            writer.writerows(data)
            
            records_fetched += len(data)
            offset += batch_size
            
            print(f"Year {year}: Fetched {records_fetched} records so far...")
            
            # For testing purposes, stop early if limit is set
            if limit and records_fetched >= limit:
                print(f"Reached limit of {limit} records for year {year}.")
                break
            
            # If we received less data than our requested batch size, it's the last page
            if len(data) < batch_size:
                break
                
    print(f"Completed year {year}. Total records fetched: {records_fetched}. Saved to {output_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Medicare Provider data from CMS API.")
    parser.add_argument("--year", type=int, help="Specific year to pull (e.g. 2023). If omitted, pulls all years.")
    parser.add_argument("--limit", type=int, help="Limit number of records per year (useful for testing).")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size per API request.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory to save CSVs.")
    
    args = parser.parse_args()
    
    years_to_pull = {args.year: DATASETS[args.year]} if args.year and args.year in DATASETS else DATASETS
    
    for year, uuid in years_to_pull.items():
        fetch_and_save_data(year, uuid, output_dir=args.output_dir, batch_size=args.batch_size, limit=args.limit)
