# OCR Processing Module

This module provides OCR (Optical Character Recognition) functionality to extract text from PDF page images. It supports multiple OCR engines and integrates seamlessly with the processing pipeline.

## Architecture

### Structure
```
processing/
├── ocr/
│   ├── __init__.py              # Package exports
│   ├── base_ocr.py              # Abstract base class for OCR engines
│   ├── easyocr_impl.py          # EasyOCR implementation
│   ├── paddleocr_impl.py        # PaddleOCR implementation
│   ├── olmocr_impl.py           # OLMOcr implementation
│   └── ocrmypdf_impl.py         # OcrMyPDF/Tesseract implementation
├── ocr_processing_stage.py      # Pipeline integration stage
└── README.md                    # This file
```

### Core Components

#### BaseOCR (Abstract Class)
All OCR implementations inherit from `BaseOCR` and must implement:
- `extract_text(image: np.ndarray) -> str`: Extract text from a page image
- `validate() -> None`: Validate installation and initialize the engine

#### OCR Implementations

1. **EasyOCR** (`easyocr_impl.py`)
   - Supports 80+ languages
   - GPU acceleration available
   - Default language: English
   - Good balance of speed and accuracy

2. **PaddleOCR** (`paddleocr_impl.py`)
   - Fast and accurate multilingual OCR
   - Supports document structure analysis
   - Efficient on both CPU and GPU
   - Default language: English

3. **OLMOcr** (`olmocr_impl.py`)
   - Open-source language model-based OCR
   - Customizable model selection
   - Good for specialized documents

4. **OcrMyPDF** (`ocrmypdf_impl.py`)
   - Built on Tesseract OCR engine
   - Integrates with ocrmypdf library
   - Simpler setup, widely supported languages

#### OCRProcessingStage
Integrates OCR into the processing pipeline:
- Reads images from `{image_directory}/{digest}/{digest}_{pg_no}.jpeg`
- Applies OCR extraction
- Saves text to `{txt_directory}/{digest}/{digest}_{pg_no}.txt`
- Follows the protocol defined in `DATA_MODEL.md`

## Usage

### Basic Usage with Processing Pipeline

```python
from govscape.config import DataModel
from govscape.processing import OCRProcessingStage

# Initialize data model
data_model = DataModel("/path/to/data")

# Create OCR processing stage with EasyOCR
ocr_stage = OCRProcessingStage(
    data_model=data_model,
    ocr_type="easyocr",
    languages=["en"],
    gpu=False
)

# Validate and run
ocr_stage.validate()
ocr_stage.run()
```

### Direct OCR Engine Usage

```python
from govscape.processing.ocr import EasyOCRImpl
import cv2

# Create OCR engine
ocr = EasyOCRImpl(languages=["en"], gpu=False)
ocr.validate()

# Load and process an image
image = cv2.imread("page.jpeg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract text
text = ocr.extract_text(image_rgb)
print(text)
```

## Installation

Install the required dependencies for your chosen OCR engine:

### EasyOCR
```bash
pip install easyocr
```

### PaddleOCR
```bash
pip install paddleocr paddlepaddle
```

### OLMOcr
```bash
pip install olmocr
```

### OcrMyPDF (with Tesseract)
```bash
pip install ocrmypdf pytesseract
# Also install Tesseract binary:
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Configuration

### OCRProcessingStage Parameters

- `data_model` (DataModel): Required. Defines the directory structure
- `ocr_type` (str): Type of OCR engine ('easyocr', 'paddleocr', 'olmocr', 'ocrmypdf')
- `**ocr_kwargs`: Engine-specific parameters

### EasyOCR Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="easyocr",
    languages=["en", "fr"],  # Support multiple languages
    gpu=True                  # Use GPU if available
)
```

### PaddleOCR Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="paddleocr",
    language="en",           # Single language
    use_gpu=True             # Use GPU if available
)
```

### OLMOcr Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="olmocr",
    model_name="default"     # Model variant
)
```

### OcrMyPDF Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="ocrmypdf",
    language="eng",          # Tesseract language code
    output_type="txt"        # Output format
)
```

## Output Format

Text extraction results are saved following the `DATA_MODEL.md` protocol:
```
{txt_directory}/{digest}/{digest}_{pg_no}.txt
```

Example:
```
txt/
├── abc123def456abc123def456abc123def45/
│   ├── abc123def456abc123def456abc123def45_0.txt
│   ├── abc123def456abc123def456abc123def45_1.txt
│   └── abc123def456abc123def456abc123def45_2.txt
└── xyz789abc123xyz789abc123xyz789abc123/
    ├── xyz789abc123xyz789abc123xyz789abc123_0.txt
    └── xyz789abc123xyz789abc123xyz789abc123_1.txt
```

## Error Handling

The module includes comprehensive error handling:
- **Validation Errors**: Raised if OCR engine dependencies are missing
- **Processing Errors**: Logged and skipped per image (doesn't stop the pipeline)
- **Image Read Errors**: Logged and counted separately

Check logs for details on failed extractions:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

### Speed Comparison
- **EasyOCR**: Moderate speed, good accuracy (GPU recommended for batch processing)
- **PaddleOCR**: Fastest, good accuracy, efficient memory usage
- **OLMOcr**: Varies by model, good for specialized content
- **OcrMyPDF**: Moderate speed, widely supported

### GPU Acceleration
For faster processing with large document volumes:
```python
# EasyOCR with GPU
OCRProcessingStage(data_model, ocr_type="easyocr", gpu=True)

# PaddleOCR with GPU
OCRProcessingStage(data_model, ocr_type="paddleocr", use_gpu=True)
```

### Memory Usage
- EasyOCR: ~2-4GB per GPU instance
- PaddleOCR: ~1-2GB per GPU instance
- OcrMyPDF: ~500MB (CPU only)

## Extending the Module

To add a new OCR engine:

1. Create a new file in `ocr/` directory
2. Implement the `BaseOCR` abstract class
3. Add the import and mapping in `ocr/__init__.py`
4. Add the mapping in `_build_ocr_engine()` function in `ocr_processing_stage.py`

Example:
```python
# ocr/tesseract_impl.py
from .base_ocr import BaseOCR

class TesseractImpl(BaseOCR):
    def __init__(self, language: str = "eng"):
        self.language = language
        self.ocr = None

    def validate(self) -> None:
        # Initialize Tesseract
        pass

    def extract_text(self, image: np.ndarray) -> str:
        # Extract text using Tesseract
        pass
```

## Testing

Test the OCR implementations:
```bash
# Run OCR tests
poetry run pytest tests/test_ocr.py -v

# Run specific OCR implementation test
poetry run pytest tests/test_ocr.py::test_easyocr -v
```

## Troubleshooting

### ImportError: ocr library not found
Make sure to install the library for your chosen OCR engine (see Installation section).

### CUDA/GPU not available
If GPU support fails, set `gpu=False` or `use_gpu=False` to use CPU-only mode.

### Poor OCR accuracy
- Try a different OCR engine
- Preprocess images (resize, denoise) before extraction
- Adjust language settings to match document content
- Use PaddleOCR or EasyOCR for better accuracy on complex layouts

### Out of memory errors
- Reduce batch size
- Use CPU-only mode
- Process fewer pages at a time

## References

- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [OcrMyPDF Documentation](https://ocrmypdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
