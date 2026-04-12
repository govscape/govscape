import gzip
import json
import logging
import os
import re
from multiprocessing import Pool, cpu_count

import pandas as pd
from govscape.data_loader import build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def extract_date_from_crawl_string(crawl_string):
    """
    Extract the date string (YYYYMMDD format) from a crawl data string.

    Args:
        crawl_string (str): String like
            "crawl-data/EOT-2020/segments/IA-000/warc/"
            "EOT20-20201009165744-crawl812_EOT20-20201009165744-00001.warc.gz"

    Returns:
        str: Date string in YYYYMMDD format (e.g., "20201009") or None if not found
    """
    # Pattern to match 8 digits representing a date (YYYYMMDD)
    pattern = r"(\d{8})"

    match = re.search(pattern, crawl_string)
    if match:
        return match.group(1)
    return None


class CDXProcessor:
    def __init__(
        self, data_loader, bucket_name, cdx_file_paths, processor_id, output_dir
    ):
        self.bucket_name = bucket_name
        self.output_dir = output_dir
        self.cdx_file_paths = cdx_file_paths
        self.cdx_file_idx = 0
        self.processor_id = processor_id
        self.data_loader = data_loader
        self.file_handle = self.get_cdx_file_handle()

    def get_cdx_file_handle(self):
        cdx_path = self.cdx_file_paths[self.cdx_file_idx]
        self.local_cdx_path = os.path.join(
            self.output_dir, f"cdx_data_{self.processor_id}_{self.cdx_file_idx}.gz"
        )
        self.data_loader.download_file(cdx_path, self.local_cdx_path)
        self.file_handle = gzip.open(self.local_cdx_path, "rb")  # noqa: SIM115
        return self.file_handle

    def get_next_pdf_entry(self):
        pdf_entry = None
        while not pdf_entry:
            cdx_line = self.file_handle.readline()
            if not cdx_line:
                if self.cdx_file_idx < len(self.cdx_file_paths) - 1:
                    self.cdx_file_idx += 1
                    self.close_file_handle()
                    self.get_cdx_file_handle()
                else:
                    return None
            try:
                cdx_line_string = cdx_line.decode().partition(" ")[2].partition(" ")[2]
                data = json.loads(cdx_line_string)
            except Exception:
                continue  # Skip lines that are not valid JSON
            if (
                (data.get("mime") == "application/pdf") or (".pdf" in data.get("url"))
            ) and data.get("status") == "200":
                pdf_entry = {
                    "url": data.get("url"),
                    "filename": data.get("filename"),
                    "crawl_date": extract_date_from_crawl_string(data.get("filename")),
                    "digest": data.get("digest").replace("sha1:", ""),
                    "offset": data.get("offset"),
                    "length": data.get("length"),
                }
        return pdf_entry

    def close_file_handle(self):
        self.file_handle.close()
        os.remove(self.local_cdx_path)


def process_cdx_batch(args):
    backend, bucket, local_base_dir, cdx_file_paths, processor_id, output_dir = args
    data_loader = build_data_loader(backend, bucket, local_base_dir)
    processor = CDXProcessor(
        data_loader, bucket, cdx_file_paths, processor_id, output_dir
    )
    entries = []
    while True:
        pdf_entry = processor.get_next_pdf_entry()
        if not pdf_entry:
            break
        entries.append(pdf_entry)
    processor.close_file_handle()
    return entries


def main():
    parser = base_argument_parser(description="Process CDX files from S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--cdx_file_paths",
        required=True,
        help="File containing paths to CDX files in S3",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save output files"
    )
    parser.add_argument("--output_prefix", required=True, help="Prefix for output")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2 * cpu_count(),
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    # Read all CDX file paths
    with open(args.cdx_file_paths) as f:
        cdx_file_paths = [line.strip() for line in f if ".gz" in line.strip()]

    # Split cdx_file_paths into batches for each worker
    num_workers = min(args.num_workers, len(cdx_file_paths))
    batches = [cdx_file_paths[i::num_workers] for i in range(num_workers)]

    pool_args = [
        (args.backend, args.bucket, args.local_base_dir, batch, str(i), args.output_dir)
        for (i, batch) in enumerate(batches)
        if batch
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_cdx_batch, pool_args)

    # Flatten results and save to parquet
    all_entries = [entry for batch in results for entry in batch]
    parquet_path = os.path.join(args.output_dir, "pdf_metadata.parquet")
    df = pd.DataFrame(all_entries)
    df.to_parquet(parquet_path, index=False)

    data_loader = build_data_loader(args.backend, args.bucket, args.local_base_dir)
    data_loader.upload_file(parquet_path, os.path.join(args.output_prefix, "metadata"))


main()
