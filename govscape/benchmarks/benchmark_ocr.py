import os
from pathlib import Path
# Import the concrete engine classes we created
from engines import EasyOCREngine, PaddleOCREngine, OCRmyPDFEngine, OlmOCREngine

def benchmark_all_engines(input_filename):
    """
    Iterates through a list of engine instances and collects benchmark data.
    """
    # 1. Initialize the engines in a list
    engines = [
        EasyOCREngine(),
        PaddleOCREngine(),
        OCRmyPDFEngine(),
        OlmOCREngine()
    ]
    
    report_data = {}
    print(f"🚀 Starting Refactored Benchmark for: {input_filename}\n" + "-"*30)

    # 2. Polymorphic loop: Every engine has the .run_benchmark() method
    for engine in engines:
        print(f"🔍 Running {engine.name}...")
        # The base class handles the timing and try-except logic
        report_data[engine.name] = engine.run_benchmark(input_filename)

    return report_data

def generate_markdown_report(report_data, source_file):
    """
    Compiles the timing and text into a formatted .md file.
    """
    output_file = f"Comparison_Report_{Path(source_file).stem}.md"
    
    with open(output_file, 'w', encoding='utf-8') as m:
        m.write(f"# OCR Benchmarking Report: `{source_file}`\n\n")
        
        m.write("## Performance Summary\n\n")
        m.write("| Engine | Status | Time (s) |\n| :--- | :--- | :--- |\n")
        for name, data in report_data.items():
            m.write(f"| {name} | {data['status']} | {data['time']:.3f}s |\n")
        
        m.write("\n---\n\n## Detailed Outputs\n\n")
        for name, data in report_data.items():
            m.write(f"### {name} Output\n")
            m.write(f"**Processing Time:** {data['time']:.4f} seconds\n\n")
            m.write("```markdown\n")
            m.write(data['content'] if data['content'] else "[No text detected]")
            m.write("\n```\n\n---\n")

    print(f"\n✅ Done! Report generated: {output_file}")

if __name__ == "__main__":
    target_file = "sample_document.pdf" 
    
    if os.path.exists(target_file):
        results = benchmark_all_engines(target_file)
        generate_markdown_report(results, target_file)
    else:
        print(f"❌ Error: Could not find '{target_file}' in this directory.")