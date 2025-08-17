import os
import json
from typing import List, Dict
from pathlib import Path
from iit_rank_extractor import IITRankExtractor
import pandas as pd

class BatchProcessor:
    def __init__(self, extractor: IITRankExtractor):
        self.extractor = extractor
        
    def process_multiple_pdfs(self, pdf_directory: str, output_directory: str) -> List[Dict]:
        """Process multiple PDF files in a directory"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        all_extracted_data = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            # Try to extract year from filename
            year = self.extract_year_from_filename(pdf_file.name)
            
            # Extract data
            data = self.extractor.extract_ranks_from_pdf(str(pdf_file), year)
            
            if data:
                # Save individual file
                output_file = os.path.join(output_directory, f"{pdf_file.stem}.json")
                self.extractor.save_extracted_data(data, output_file)
                all_extracted_data.append(data)
                print(f"✓ Successfully processed {pdf_file.name}")
            else:
                print(f"✗ Failed to process {pdf_file.name}")
        
        return all_extracted_data
    
    def extract_year_from_filename(self, filename: str) -> str:
        """Extract year from PDF filename"""
        import re
        year_match = re.search(r'20\d{2}', filename)
        return year_match.group(0) if year_match else None
    
    def combine_all_data(self, all_data: List[Dict], output_path: str):
        """Combine all extracted data into a single DataFrame"""
        all_rows = []
        
        for data in all_data:
            df = self.extractor.convert_to_dataframe(data)
            if not df.empty:
                all_rows.append(df)
        
        if all_rows:
            combined_df = pd.concat(all_rows, ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Combined data saved to {output_path}")
            return combined_df
        else:
            print("No data to combine")
            return pd.DataFrame()
