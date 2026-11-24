import subprocess
import sys


def test_pdf_extraction_workers_do_not_load_torch():
    # Every spawned extraction worker imports the package to unpickle its task;
    # loading torch there costs ~1 GB per worker and ran ingest workers out of memory.
    code = (
        "import sys, govscape, govscape.processing.pdf_extraction_stage;"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_package_exports_still_resolve():
    import govscape
    from govscape.processing import PageImageEmbeddingStage, PDFExtractionStage

    assert govscape.PDFProcessingPipeline.__name__ == "PDFProcessingPipeline"
    assert PDFExtractionStage.__name__ == "PDFExtractionStage"
    assert PageImageEmbeddingStage.__name__ == "PageImageEmbeddingStage"
    for name in govscape.__all__:
        assert getattr(govscape, name) is not None
