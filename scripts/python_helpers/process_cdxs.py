import os
import argparse
import boto3
import warcio
import json
import re
import gzip
import pandas as pd
from multiprocessing import Pool, Manager, cpu_count

class CDXProcessor:
    def __init__(self, bucket_name, cdx_file_paths, processor_id, output_dir):
        self.bucket_name = bucket_name
        self.output_dir = output_dir
        self.cdx_file_paths = cdx_file_paths
        self.cdx_file_idx = 0
        self.processor_id = processor_id
        self.file_handle = self.get_cdx_file_handle()

    def get_cdx_file_handle(self):
        cdx_path = self.cdx_file_paths[self.cdx_file_idx]
        s3 = boto3.client('s3')
        local_cdx_path = os.path.join(self.output_dir, f'cdx_data_{self.processor_id}_{self.cdx_file_idx}.gz')
        s3.download_file(self.bucket_name, cdx_path, local_cdx_path)
        self.file_handle = gzip.open(local_cdx_path, 'rb')
        return self.file_handle

    def get_next_pdf_entry(self):
        pdf_entry = None
        while not pdf_entry:
            cdx_line = self.file_handle.readline()
            if not cdx_line:
                if self.cdx_file_idx < len(self.cdx_file_paths) - 1:
                    self.cdx_file_idx += 1
                    self.get_cdx_file_handle()
                else:
                    return None
            try:
                cdx_line_string = cdx_line.decode().partition(' ')[2].partition(' ')[2]
                data = json.loads(cdx_line_string)
            except Exception:
                continue  # Skip lines that are not valid JSON
            if ((data.get('mime') == 'application/pdf') or (".pdf" in data.get('url'))) and data.get('status') == '200':
                pdf_entry = {
                    'url': data.get('url'),
                    'filename': data.get('filename'),
                    'digest': data.get('digest'),
                    'offset': data.get('offset'),
                    'length': data.get('length'),
                }
        return pdf_entry

    def close(self):
        self.file_handle.close()

def process_cdx_batch(args):
    bucket, cdx_file_paths, processor_id, output_dir = args
    processor = CDXProcessor(bucket, cdx_file_paths, processor_id, output_dir)
    entries = []
    while True:
        pdf_entry = processor.get_next_pdf_entry()
        if not pdf_entry:
            break
        entries.append(pdf_entry)
    processor.close()
    return entries

def main():
    parser = argparse.ArgumentParser(description='Process CDX files from S3.')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--cdx_file_paths', required=True, help='File containing paths to CDX files in S3')
    parser.add_argument('--output_dir', required=True, help='Directory to save output files')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of parallel workers')
    args = parser.parse_args()

    # Read all CDX file paths
    with open(args.cdx_file_paths, 'r') as f:
        cdx_file_paths = [line.strip() for line in f if ".gz" in line.strip()]

    # Split cdx_file_paths into batches for each worker
    num_workers = min(args.num_workers, len(cdx_file_paths))
    batches = [cdx_file_paths[i::num_workers] for i in range(num_workers)]

    pool_args = [(args.bucket, batch, str(i), args.output_dir) for (i, batch) in enumerate(batches) if batch]

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_cdx_batch, pool_args)

    # Flatten results and save to parquet
    all_entries = [entry for batch in results for entry in batch]
    parquet_path = os.path.join(args.output_dir, "pdf_metadata.parquet")
    df = pd.DataFrame(all_entries)
    df.to_parquet(parquet_path, index=False)
    # Compute unique 'filename' values and save to a separate parquet file
    unique_filenames = df['filename'].dropna().unique()
    filenames_df = pd.DataFrame({'filename': unique_filenames})
    filenames_parquet_path = os.path.join(args.output_dir, "pdf_warc_files.parquet")
    filenames_df.to_parquet(filenames_parquet_path, index=False)


main()

