import os
import subprocess

input_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\testFiles"
output_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\MarkDownsOcrMyPdf"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith(".pdf"):
        print(f"Processing {file} with OCRMyPDF...")
        output_md = os.path.join(output_dir, file.replace(".pdf", ".md"))
        temp_pdf = os.path.join(output_dir, "temp_" + file)
        
        # --sidecar extracts the text to a file
        subprocess.run([
            "ocrmypdf", "--sidecar", output_md,
            os.path.join(input_dir, file), temp_pdf,
            "--optimize", "1", "--skip-text"
        ])
        if os.path.exists(temp_pdf): os.remove(temp_pdf)