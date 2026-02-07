import argparse
import os
from warcio.archiveiterator import ArchiveIterator
import subprocess
import pandas as pd
import requests
import shutil
import multiprocessing
import time
import io 
from govscape.data_loader import build_data_loader

def main():
    parser = argparse.ArgumentParser(description='Retrieve PDFs from S3 & store them.')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--cdx_parquet', required=True, help='File containing paths to CDX files in S3')
    parser.add_argument('--output_dir', required=True, help='Directory to save output files')
    parser.add_argument('--backend', choices=['s3', 'local'], default='s3', help='Data backend to use')
    parser.add_argument('--local_base_dir', type=str, default='data', help='Base directory for local backend')
    args = parser.parse_args()

    df = pd.read_parquet(args.cdx_parquet)
    output_bucket_name = 'bcgl-public-bucket'
    output_directory_base = args.output_dir
    num_processes = multiprocessing.cpu_count() * 4
    batch_size = len(df) // num_processes + 1
    print(df)
    batches = [(df[i:i + batch_size], idx, output_bucket_name, output_directory_base, num_processes, args.backend, args.local_base_dir) for idx, i in enumerate(range(0, len(df), batch_size))]
    df = None
    with multiprocessing.get_context('fork').Pool(processes=num_processes) as pool:
        pool.starmap(retrieve_and_store_pdfs, batches)

def retrieve_and_store_pdfs(file_batch, idx, output_bucket_name, output_directory, num_processes, backend, local_base_dir):
    data_loader = build_data_loader(backend, output_bucket_name, local_base_dir)
    valid_pdfs = 0
    invalid_pdfs = 0
    start_time = time.time()
    for i in range(1, len(file_batch)):
        filename = file_batch.iloc[i]['filename']
        url = file_batch.iloc[i]['url']
        digest = file_batch.iloc[i]['digest']
        length = int(file_batch.iloc[i]['length'])
        offset = int(file_batch.iloc[i]['offset'])
        s3_url = f'https://eotarchive.s3.amazonaws.com/{filename}'
        myagent = 'govscape/0.1 (PDF Retrieval Script; kdeeds@cs.washington.edu)'
        byte_range = f'bytes={offset}-{offset + length - 1}'
        object_exists = False
        if ((valid_pdfs + invalid_pdfs) % 100 == 50):
            print(f'Heartbeat: {idx}')
            if idx == 0:
                print(f'Processed {valid_pdfs} PDFs in {time.time() - start_time:.4f} seconds')
                pdf_per_second = (valid_pdfs + invalid_pdfs) / (time.time() - start_time)
                print(f'Time Remaining: {(len(file_batch)- i) / pdf_per_second :.4f} seconds')
                print(f'Time per PDF: {1 / pdf_per_second:.4f} seconds')

        output_digest = digest.replace("sha1:", "")
        try:    
            object_exists = data_loader.exists(os.path.join(output_directory, output_digest + '.pdf'))
        except Exception:
            object_exists = False  # Object does not exist, continue to download

        if object_exists:
           invalid_pdfs += 1
           continue
        
        # Send the HTTP GET request to the S3 URL with the specified byte range
        try:
            response = requests.get(
                s3_url,
                headers={'user-agent': myagent, 'Range': byte_range},
                stream=True
            )
            for record in ArchiveIterator(response.raw):
                if record.rec_type == 'response':
                    is_valid_pdf = (record.rec_headers.get('Content-Type') == 'application/pdf') or (".pdf" in record.rec_headers.get('WARC-Target-URI', ''))
                    if not is_valid_pdf:
                        invalid_pdfs += 1
                        continue
                    data_loader.upload_bytes(
                        record.content_stream().read(),
                        os.path.join(output_directory, output_digest + '.pdf')
                    )
                    valid_pdfs += 1
                        
        except Exception as e:
            print(f"Error processing {filename}: {e} on Server: {idx}")
            invalid_pdfs += 1
            continue

if __name__ == "__main__":
    main()