"""Page images are stored many to an object ("packs") and read by byte range.

Uploading each page's JPEG as its own object would take ~1.1 billion requests
for the archive. Instead each batch's page images are concatenated into one
pack, img/pack_{sha1}.jpegs, and each PDF's metadata.json records where its
pages are:

    "page_images": {"pack": "pack_<sha1>.jpegs", "offsets": [0, 51234, ...]}

with one more offset than pages, so page i is bytes offsets[i] to
offsets[i + 1] - 1 of the pack. A PDF's pages are contiguous, in page order.
"""

import hashlib
import json
import logging
import os

from .config import DataModel

PACK_SUFFIX = ".jpegs"


def pack_page_images(data_model: DataModel) -> str | None:
    """Concatenate data_model's page images into one pack in its image directory,
    record each PDF's offsets in its metadata.json, and return the pack's name,
    or None if there are no page images."""
    pdfs = []  # (digest, page image paths in page order)
    if not os.path.isdir(data_model.metadata_directory):
        return None
    for digest in sorted(os.listdir(data_model.metadata_directory)):
        metadata_path = data_model.metadata_file_path(digest)
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path) as f:
            num_pages = json.load(f)["num_pages"]
        pages = [data_model.img_page_path(digest, i) for i in range(num_pages)]
        missing = [p for p in pages if not os.path.exists(p)]
        if missing:
            # Extraction leaves no partial pages, so this is a PDF it couldn't
            # parse; it has no images to show.
            logging.warning(f"{digest}: {len(missing)} page images missing; skipped")
            continue
        pdfs.append((digest, pages))
    if not pdfs:
        return None

    # Named after its contents, so re-running a batch overwrites the same pack.
    names = "\n".join(os.path.basename(p) for _, pages in pdfs for p in pages)
    pack_name = f"pack_{hashlib.sha1(names.encode()).hexdigest()}{PACK_SUFFIX}"
    with open(data_model.img_pack_path(pack_name), "wb") as pack:
        for digest, pages in pdfs:
            offsets = [pack.tell()]
            for page in pages:
                with open(page, "rb") as image:
                    pack.write(image.read())
                offsets.append(pack.tell())
            _record_page_images(data_model, digest, pack_name, offsets)
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
