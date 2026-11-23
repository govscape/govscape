from pathlib import Path

from govscape.config import digest_shards


def test_digest_shards_partition_the_pdf_digests():
    shards = digest_shards(2)
    assert len(shards) == len(set(shards)) == 32**2

    pdf_dir = Path(__file__).resolve().parent / "test_data" / "pdfs"
    for pdf in pdf_dir.glob("*.pdf"):
        assert sum(pdf.stem.startswith(shard) for shard in shards) == 1
