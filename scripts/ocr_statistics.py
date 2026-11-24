"""
Script for analyzing OCR text from government documents to extract summary statistics
from the AI2 OLMOCR dataset. This script loads JSONL files containing OCR text.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import pdfplumber


def tokenize(text: str) -> list[str]:
    # Split on non-word characters
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _count_file(path: Path) -> tuple[int, list[int], list[int]]:
    """Process one JSONL file; returning a list of the ocr token counts for each page,
    and a list of the token counts for each page from the original"""
    ocr_page_tokens = []
    original_page_tokens = []

    doc_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc_count += 1

            doc = json.loads(line)
            txt = doc.get("text", "")

            page_nums = doc.get("attributes").get("pdf_page_numbers")

            source_file = doc.get("metadata").get("Source-File").split("/")[-1]
            if not (source_file.endswith(".pdf")):
                raise ValueError(f"Unexpected source file format: {source_file}")

            s3_key = f"s3://eot-pdf-archive/pdfs/{source_file}"

            # use s5cmd to download the PDF file from S3, then use pdfplumber to extract
            #  the text from the PDF file, and then count the tokens in each page
            temp_pdf_path = Path(f"/tmp/{source_file}")
            try:
                if not temp_pdf_path.exists():
                    subprocess.run(
                        ["s5cmd", "cp", s3_key, str(temp_pdf_path)], check=True
                    )

                with pdfplumber.open(temp_pdf_path) as pdf:
                    original_page_tokens.extend(
                        len(tokenize(page.extract_text() or "")) for page in pdf.pages
                    )
            except Exception as e:
                print(f"Skipping source file {source_file}: {e}")
            finally:
                if temp_pdf_path.exists():
                    temp_pdf_path.unlink()

            for page_num in page_nums:
                start = page_num[0]
                end = page_num[1]
                page_text = txt[start:end]
                ocr_page_tokens.append(len(tokenize(page_text)))
    return doc_count, ocr_page_tokens, original_page_tokens


def batched(iterable: iter[Path], n: int) -> iter[list[Path]]:
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _count_batch(paths: list[Path]) -> tuple[int, Counter, int]:
    batch_docs = 0
    batch_files = 0

    ocr_batch_counter: Counter[int] = Counter()

    for path in paths:
        n_docs, ocr_counts, original_counts = _count_file(path)
        for count in ocr_counts:
            ocr_batch_counter[count] += 1

        for count in original_counts:
            ocr_batch_counter[count] += 1

        batch_docs += n_docs
        batch_files += 1

    return batch_docs, ocr_batch_counter, original_counts, batch_files


def count_tokens_from_jsonl(data_dir: Path, batch_size: int = 250) -> dict[int, int]:
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {data_dir}")

    jsonl_files = sorted(jsonl_files)

    jsonl_files = jsonl_files[
        :1000
    ]  # limit to 1000 files for testing; remove this line to process all files

    batches = list(batched(jsonl_files, batch_size))

    ocr_total_tokens: Counter[int] = Counter()
    original_total_tokens: Counter[int] = Counter()
    doc_count = 0
    files_done = 0
    start_time = time.perf_counter()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_count_batch, batch) for batch in batches]

        for future in as_completed(futures):
            n_docs, ocr_counter, original_counter, n_files = future.result()

            doc_count += n_docs
            files_done += n_files
            ocr_total_tokens.update(ocr_counter)
            original_total_tokens.update(original_counter)

            if files_done % 100 == 0:
                elapsed = time.perf_counter() - start_time
                file_rate = files_done / elapsed
                doc_rate = doc_count / elapsed
                print(
                    f"{files_done}/{len(jsonl_files)} files "
                    f"in {elapsed:.1f}s "
                    f"({file_rate:.1f} files/s, {doc_rate:.0f} docs/s)"
                )

    elapsed = time.perf_counter() - start_time
    print(
        f"Processed {doc_count} documents from {files_done} files "
        f"in {elapsed:.1f}s "
        f"({files_done / elapsed:.1f} files/s, {doc_count / elapsed:.0f} docs/s)"
    )

    return dict(ocr_total_tokens), dict(original_total_tokens)


ocr, original = count_tokens_from_jsonl(Path("data/ai2-olmocr/jsonl"))
print("Done counting tokens")

# save tokens to csvs
with open("ocr_token_counts.csv", "w", encoding="utf-8") as f:
    f.write("token_count,num_pages\n")
    for token_count, num_pages in sorted(ocr.items()):
        f.write(f"{token_count},{num_pages}\n")

with open("original_token_counts.csv", "w", encoding="utf-8") as f:
    f.write("token_count,num_pages\n")
    for token_count, num_pages in sorted(original.items()):
        f.write(f"{token_count},{num_pages}\n")
