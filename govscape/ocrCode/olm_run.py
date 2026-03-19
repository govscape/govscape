import os
import subprocess

input_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\testFiles"
output_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\MarkDownsOlmOcr"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith(".pdf"):
        print(f"Processing {file} with OlmOCR...")
        # Running via CLI to let the OS manage memory cleanup between files
        subprocess.run([
            "python", "-m", "olmocr.pipeline", 
            output_dir, 
            "--pdfs", os.path.join(input_dir, file)
        ])