import os
import easyocr
from pdf2image import convert_from_path

input_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\testFiles"
output_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\MarkDownsEasyOcr"
poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"
os.makedirs(output_dir, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)

for file in os.listdir(input_dir):
    if file.lower().endswith(".pdf"):
        print(f"Processing {file} with EasyOCR...")
        images = convert_from_path(os.path.join(input_dir, file), poppler_path=poppler_path)
        with open(os.path.join(output_dir, file.replace(".pdf", ".md")), "w", encoding="utf-8") as f:
            for i, img in enumerate(images):
                text = reader.readtext(img, detail=0)
                f.write(f"## Page {i+1}\n" + " ".join(text) + "\n\n")