"""Page images are stored many to an object ("packs") and read by byte range.

Uploading each page's JPEG as its own object would take ~1.1 billion requests
for the archive. Instead each batch's page images are concatenated into one
pack, img/pack_{sha1}.jpegs, and each PDF's metadata.json records where its
pages are (the pack is named after the SHA-1 of its bytes):

    "page_images": {"pack": "pack_<sha1>.jpegs", "offsets": [0, 51234, ...]}

with one more offset than pages, so page i is bytes offsets[i] to
offsets[i + 1] - 1 of the pack. A PDF's pages are contiguous, in page order.
"""

import hashlib
import json
import os

from .config import DataModel
from .processing.pdf_extraction_stage import extracted_pdfs

PACK_SUFFIX = ".jpegs"


def pack_page_images(data_model: DataModel) -> str | None:
    """Concatenate the page images of data_model's extracted PDFs into one pack in
    its image directory, record each PDF's offsets in its metadata.json, and
    return the pack's name, or None if there are no PDFs."""
    pdfs = extracted_pdfs(data_model)
    if not pdfs:
        return None
    offsets = {}
    # Named after its bytes, so a pack's name always identifies the bytes that
    # offsets recorded with that name point into.
    sha1 = hashlib.sha1()
    unnamed_pack = data_model.img_pack_path("pack.partial")
    with open(unnamed_pack, "wb") as pack:
        for digest, num_pages in pdfs:
            offsets[digest] = [pack.tell()]
            for page_num in range(num_pages):
                with open(data_model.img_page_path(digest, page_num), "rb") as f:
                    image = f.read()
                sha1.update(image)
                pack.write(image)
                offsets[digest].append(pack.tell())
    pack_name = f"pack_{sha1.hexdigest()}{PACK_SUFFIX}"
    os.rename(unnamed_pack, data_model.img_pack_path(pack_name))
    for digest, pdf_offsets in offsets.items():
        _record_page_images(data_model, digest, pack_name, pdf_offsets)
    return pack_name


def _record_page_images(
    data_model: DataModel, digest: str, pack_name: str, offsets: list[int]
) -> None:
    metadata_path = data_model.metadata_file_path(digest)
    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["page_images"] = {"pack": pack_name, "offsets": offsets}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
