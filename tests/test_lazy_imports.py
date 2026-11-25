import subprocess
import sys


def test_pdf_extraction_and_tesseract_workers_do_not_load_torch():
    # Every spawned extraction and OCR worker imports the package to unpickle its
    # task; loading torch there costs ~1 GB per worker and ran ingest workers out
    # of memory.
    code = (
        "import sys, govscape, govscape.processing.pdf_extraction_stage;"
        "from govscape.processing.ocr_processing_stage import _ocr_engine_class;"
        "_ocr_engine_class('ocrmypdf');"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
