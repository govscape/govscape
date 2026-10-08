import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import duckdb
import pandas as pd
import requests

from scripts.data_prep import retrieve_source_coop_data as source_data


class FakeResponse(requests.Response):
    def iter_content(
        self, chunk_size: int | None = None, decode_unicode: bool = False
    ) -> Iterator[bytes]:
        assert chunk_size == 1024 * 1024
        yield b"pdf-content"

    def close(self) -> None:
        return None


def test_download_pdf_uses_archive_digest_path(tmp_path: Path) -> None:
    digest = "ABC123"
    session = requests.Session()
    response = FakeResponse()
    response.status_code = 200
    with patch.object(session, "get", return_value=response) as get:
        pdf_path = source_data._download_pdf(session, digest, tmp_path)

    get.assert_called_once_with(
        f"{source_data.PDF_BASE_URL}/{digest}.pdf",
        stream=True,
        timeout=(10, 120),
    )
    assert pdf_path == tmp_path / f"{digest}.pdf"
    assert pdf_path.read_bytes() == b"pdf-content"
    assert sorted(path.name for path in tmp_path.iterdir()) == [pdf_path.name]


DIGEST_1, DIGEST_2, DIGEST_3 = (f"{'A' * 31}{i}" for i in (1, 2, 3))


def _write_source_cdx(path: Path) -> None:
    pd.DataFrame(
        {
            "digest": [DIGEST_2, DIGEST_1, DIGEST_1, DIGEST_3, "short-digest"],
            "url": ["two.pdf", "one.pdf", "other.pdf", "three.pdf", "four.pdf"],
        }
    ).to_parquet(path, index=False)


def test_load_cdx_rows_selects_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    _write_source_cdx(source_path)
    monkeypatch.setattr(source_data, "PARQUET_URL", str(source_path))

    with duckdb.connect() as connection:
        rows = source_data._load_cdx_rows(connection, [DIGEST_1, DIGEST_2])

    assert sorted(zip(rows["digest"], rows["url"], strict=True)) == [
        (DIGEST_1, "one.pdf"),
        (DIGEST_1, "other.pdf"),
        (DIGEST_2, "two.pdf"),
    ]

    with (
        duckdb.connect() as connection,
        pytest.raises(
            ValueError, match="No CDX records found for PDF digests: missing-digest"
        ),
    ):
        source_data._load_cdx_rows(connection, ["missing-digest"])


def _run_main(
    monkeypatch: pytest.MonkeyPatch, num_pdfs: int, pdf_dir: Path, cdx_dir: Path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrieve_source_coop_data.py",
            "--num_pdfs",
            str(num_pdfs),
            "--pdf_dir",
            str(pdf_dir),
            "--cdx_dir",
            str(cdx_dir),
        ],
    )
    response = FakeResponse()
    response.status_code = 200
    with patch.object(requests.Session, "get", return_value=response) as get:
        source_data.main()
    return get


def test_main_downloads_requested_distinct_pdfs_and_matching_cdx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    _write_source_cdx(source_path)
    monkeypatch.setattr(source_data, "PARQUET_URL", str(source_path))
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / f"{DIGEST_3}.pdf").write_bytes(b"already present")
    cdx_dir = tmp_path / "cdx"

    get = _run_main(monkeypatch, 2, pdf_dir, cdx_dir)

    # The already-present PDF is not downloaded again, but is covered by the
    # CDX sample and the manifest along with the new downloads.
    assert get.call_count == 2
    assert sorted(path.name for path in pdf_dir.iterdir()) == [
        f"{DIGEST_1}.pdf",
        f"{DIGEST_2}.pdf",
        f"{DIGEST_3}.pdf",
    ]
    with duckdb.connect() as connection:
        result = connection.execute(
            "SELECT digest, url FROM read_parquet(?) ORDER BY digest, url",
            [str(cdx_dir / source_data.CDX_SAMPLE_FILENAME)],
        ).fetchall()
    assert result == [
        (DIGEST_1, "one.pdf"),
        (DIGEST_1, "other.pdf"),
        (DIGEST_2, "two.pdf"),
        (DIGEST_3, "three.pdf"),
    ]
    manifest = pd.read_csv(cdx_dir / "digests_manifest.csv")
    assert list(manifest.itertuples(index=False, name=None)) == [
        (f"{DIGEST_1}.pdf", DIGEST_1, "one.pdf"),
        (f"{DIGEST_2}.pdf", DIGEST_2, "two.pdf"),
        (f"{DIGEST_3}.pdf", DIGEST_3, "three.pdf"),
    ]
