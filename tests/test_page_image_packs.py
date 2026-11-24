import json
import os

from govscape.config import DataModel
from govscape.page_image_packs import pack_page_images


def _write_pdf(data_model, digest, page_bytes):
    os.makedirs(data_model.metadata_pdf_directory(digest))
    with open(data_model.metadata_file_path(digest), "w") as f:
        json.dump({"digest": digest, "num_pages": len(page_bytes)}, f)
    os.makedirs(data_model.img_pdf_directory(digest))
    for page_num, data in enumerate(page_bytes):
        with open(data_model.img_page_path(digest, page_num), "wb") as f:
            f.write(data)


def _page_images(data_model, digest):
    with open(data_model.metadata_file_path(digest)) as f:
        return json.load(f).get("page_images")


def test_pages_are_packed_in_order_and_located_by_offsets(tmp_path):
    data_model = DataModel(str(tmp_path))
    # 11 pages, so page 10 would sort before page 2 as a string.
    pages_b = [f"b{i}-".encode() * (i + 1) for i in range(11)]
    _write_pdf(data_model, "BBBB", pages_b)
    _write_pdf(data_model, "AAAA", [b"a0", b"a1-longer"])

    pack_name = pack_page_images(data_model)

    with open(data_model.img_pack_path(pack_name), "rb") as f:
        pack = f.read()
    for digest, pages in [("AAAA", [b"a0", b"a1-longer"]), ("BBBB", pages_b)]:
        packed = _page_images(data_model, digest)
        assert packed["pack"] == pack_name
        offsets = packed["offsets"]
        assert len(offsets) == len(pages) + 1
        for i, page in enumerate(pages):
            assert pack[offsets[i] : offsets[i + 1]] == page
    # PDFs are packed in digest order, each one's pages contiguous.
    assert _page_images(data_model, "AAAA")["offsets"][0] == 0
    assert (
        _page_images(data_model, "BBBB")["offsets"][0]
        == _page_images(data_model, "AAAA")["offsets"][-1]
    )


def test_pack_name_depends_on_contents(tmp_path):
    first, second = DataModel(str(tmp_path / "a")), DataModel(str(tmp_path / "b"))
    _write_pdf(first, "AAAA", [b"x"])
    _write_pdf(second, "AAAA", [b"x"])
    assert pack_page_images(first) == pack_page_images(second)
    _write_pdf(second, "CCCC", [b"y"])
    assert pack_page_images(first) != pack_page_images(second)


def test_pdfs_without_all_their_page_images_are_left_out(tmp_path):
    data_model = DataModel(str(tmp_path))
    _write_pdf(data_model, "AAAA", [b"a0"])
    _write_pdf(data_model, "BBBB", [b"b0", b"b1"])
    os.remove(data_model.img_page_path("BBBB", 1))

    pack_page_images(data_model)
    assert _page_images(data_model, "AAAA") is not None
    assert _page_images(data_model, "BBBB") is None


def test_nothing_to_pack(tmp_path):
    assert pack_page_images(DataModel(str(tmp_path))) is None
