import PyPDF2
import sys

pdf_path = '带有注意力机制的优化 Unet 用于多尺度语义分割.pdf'

try:
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        print(f"Total pages: {len(reader.pages)}")
        
        # Extract text from all pages
        full_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text += f"\n=== Page {i+1} ===\n{text}\n"
        
        # Search for journal-related keywords
        keywords = ['Journal', 'journal', 'IEEE', 'ACM', 'Springer', 'Elsevier', 
                   'MDPI', 'Remote Sensing', 'ISPRS', 'Transactions', 'Volume', 
                   'Vol.', 'DOI', 'doi', 'Conference', 'Proceedings', 'Workshop']
        
        print("\n=== Searching for journal information ===")
        lines = full_text.split('\n')
        for line in lines:
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    print(f"Found '{keyword}' in: {line[:200]}")
                    break
        
        # Print first few pages for manual inspection
        print("\n=== First 3 pages content ===")
        for i in range(min(3, len(reader.pages))):
            print(f"\n--- Page {i+1} ---")
            print(reader.pages[i].extract_text()[:2000])
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()




