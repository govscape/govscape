import os
import argparse
import boto3
import warcio
import json
import re
import gzip
import pandas as pd
from multiprocessing import Pool, Manager, cpu_count


def main():
    entries = []
    cdx_f = gzip.open('/home/ubuntu/govscape/data/cdx_dir/EndOfTerm2020WebCrawls.cdx.gz')
    for line in cdx_f:
        cdx_line = line.decode().strip().split(' ')
        if cdx_line[4] == "200" and (".pdf" in cdx_line[2]) or (cdx_line[3] == 'application/pdf'):
            entries.append({
                'crawl_date': cdx_line[1],
                'url': cdx_line[2],
                'digest': cdx_line[5],
                'length': cdx_line[8],
                'offset': cdx_line[9],
                'warc_filename': cdx_line[10].replace('/', '_'),
            })
    entries = pd.DataFrame(entries)
    entries.to_parquet('/home/ubuntu/govscape/data/cdx_dir/pdf_metadata.parquet', index=False)
    cdx_f.close()

main()