import requests
import json
import re
import logging
import os
import tempfile
from datetime import datetime
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import pymupdf4llm
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IITRankExtractor:
    def __init__(self, 
                 gemini_api_key: str = None,
                 openai_api_key: str = None, 
                 anthropic_api_key: str = None,
                 llm_provider: str = "gemini",
                 llm_model: str = "gemini-2.5-flash"):
        """
        Initialize the IIT Rank Extractor
        
        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            llm_provider: Which LLM to use ('gemini', 'openai', 'anthropic')
            llm_model: Specific model to use
        """
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF4LLM"""
        try:
            # Try PyMuPDF4LLM first for better text extraction
            md_text = pymupdf4llm.to_markdown(pdf_path)
            if md_text.strip():
                logger.info("Successfully extracted text using PyMuPDF4LLM")
                return md_text
        except Exception as e:
            logger.error(f"Error with PyMuPDF4LLM: {e}")
        
        # Fallback to basic PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            logger.info("Successfully extracted text using PyMuPDF fallback")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images"""
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                images.append(image)
            doc.close()
            logger.info(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        return images

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        try:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            import base64
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error encoding image to base64: {e}")
            return ""

    def create_extraction_prompt(self, extracted_text: str, year: str = None) -> str:
        """Create the prompt for extracting IIT rank data"""
        prompt = f"""
You are an expert at extracting structured data from IIT admission documents. 
Extract the category-wise opening and closing ranks for all courses from the provided IIT admission PDF.

IMPORTANT: Return the data in the following JSON format:

{{
    "year": "{year or 'YYYY'}",
    "document_type": "IIT_CUTOFF_RANKS",
    "institutes": [
        {{
            "institute_name": "IIT Delhi",
            "institute_code": "IIT001",
            "courses": [
                {{
                    "course_name": "Computer Science and Engineering",
                    "course_code": "CS",
                    "duration": "4 years",
                    "degree": "B.Tech",
                    "category_ranks": {{
                        "General": {{
                            "opening_rank": 1,
                            "closing_rank": 150
                        }},
                        "OBC": {{
                            "opening_rank": 45,
                            "closing_rank": 380
                        }},
                        "SC": {{
                            "opening_rank": 15,
                            "closing_rank": 180
                        }},
                        "ST": {{
                            "opening_rank": 8,
                            "closing_rank": 95
                        }},
                        "EWS": {{
                            "opening_rank": 25,
                            "closing_rank": 200
                        }}
                    }}
                }}
            ]
        }}
    ]
}}

EXTRACTION RULES:
1. Extract ALL institutes mentioned in the document
2. Extract ALL courses/branches for each institute
3. Include category-wise opening and closing ranks (General, OBC, SC, ST, EWS, etc.)
4. If a rank is not available, use null
5. Maintain the exact JSON structure shown above
6. Use the actual institute names and course names from the document
7. Include course codes if available

EXTRACTED TEXT:
{extracted_text}

Return only the JSON data, no additional text or explanation.
"""
        return prompt

    def clean_json_response(self, response_text: str) -> Dict:
        """Clean and parse JSON response from LLM"""
        try:
            # Remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            # Find JSON content between braces
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = cleaned_text[start_idx:end_idx]
                
                # Try to fix common JSON issues
                json_str = json_str.replace('\n', ' ')  # Remove newlines
                json_str = json_str.replace('\t', ' ')  # Remove tabs
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
                json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                
                return json.loads(json_str)
            else:
                logger.error("No JSON found in response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            # Try to extract partial data if possible
            try:
                # Look for institute data patterns
                if '"institute_name"' in response_text and '"programs"' in response_text:
                    # Try to construct a minimal valid JSON
                    return {"institutes": []}
            except:
                pass
            return {}
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return {}

    def extract_with_gemini(self, prompt: str, images: List[Image.Image]) -> Dict:
        """Extract data using Google Gemini"""
        if not self.gemini_api_key:
            logger.error("Gemini API key not provided")
            return {}

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
            
            # Limit images to avoid API limits (Gemini has limits on number of images)
            max_images = 5  # Further reduce to 5 images to avoid token limits
            limited_images = images[:max_images]
            logger.info(f"Using {len(limited_images)} out of {len(images)} images")
            
            # Prepare image parts
            image_parts = []
            for image in limited_images:
                base64_image = self.encode_image_to_base64(image)
                if base64_image:
                    image_parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    })

            # Debug: Print payload structure
            logger.info(f"Number of images: {len(images)}")
            logger.info(f"Number of image_parts: {len(image_parts)}")
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}] + image_parts
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 8192,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            # Debug: Print first few characters of payload
            logger.info(f"Payload keys: {list(payload.keys())}")
            logger.info(f"Contents structure: {type(payload['contents'][0]['parts'])}")

            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload)
            
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Response keys: {list(result.keys())}")
                
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    logger.info(f"Candidate keys: {list(candidate.keys())}")
                    logger.info(f"Finish reason: {candidate.get('finishReason', 'Not specified')}")
                    
                    if "content" in candidate:
                        logger.info(f"Content keys: {list(candidate['content'].keys())}")
                        if "parts" in candidate["content"]:
                            content = candidate["content"]["parts"][0]["text"]
                            return self.clean_json_response(content)
                        else:
                            logger.error(f"No 'parts' in content. Content structure: {candidate['content']}")
                            logger.error(f"This might be due to safety filters or content policy violations")
                            return {}
                    else:
                        logger.error(f"No 'content' in candidate. Candidate structure: {candidate}")
                        return {}
                else:
                    logger.error(f"No candidates in Gemini response. Full response: {result}")
                    return {}
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error with Gemini extraction: {e}")
            return {}
    
    def extract_with_regex_parser(self, extracted_text: str) -> Dict:
        """Extract data using regex parsing of the table structure"""
        import re
        
        try:
            # Detect table format based on content analysis
            if self._is_standard_format(extracted_text):
                logger.info("Detected standard format (2022-2024)")
                return self._extract_standard_format(extracted_text)
            else:
                logger.info("Detected alternative format (2020)")
                return self._extract_alternative_format(extracted_text)
            
        except Exception as e:
            logger.error(f"Error with regex extraction: {e}")
            return {}
    
    def _is_standard_format(self, text: str) -> bool:
        """Detect if text uses standard format (with Sl. No. column)"""
        import re
        # Standard format has Sl. No. as first column
        standard_patterns = [
            r'\|S\.No\|Institute\|Academic Program Name\|Seat Type\|Gender\|OR\|CR\|',
            r'\|.*\|Institute\|.*\|Seat Type\|Gender\|.*\|.*\|',
            r'\|\d+\|.*IIT.*\|.*\|.*\|.*\|\d+\|\d+\|'  # Pattern for data rows with 7 columns
        ]
        
        for pattern in standard_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Additional check: ensure we have 7-column structure
                lines = text.split('\n')
                seven_col_count = 0
                for line in lines:
                    if line.count('|') == 8:  # 7 columns = 8 pipes
                        seven_col_count += 1
                        if seven_col_count >= 5:  # If we find at least 5 rows with 7 columns
                            return True
        return False
    
    def _extract_standard_format(self, extracted_text: str) -> Dict:
        """Extract data from standard format (2022-2024)"""
        import re
        
        # Enhanced pattern to handle strikethrough (~~) and <br> tags
        # Pattern: |Sl.No|Institute|Program|Seat Type|Gender|OR|CR|
        # Handles: |123|, |~~123~~|, |123<br>|
        pattern = r'\|(?:~~)?(\d+)(?:~~)?(?:<br>)?\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
        matches = re.findall(pattern, extracted_text, re.DOTALL)
        
        if not matches:
            logger.warning("No matches found with standard format pattern")
            return {}
        
        logger.info(f"Found {len(matches)} rows in standard format")
        return self._process_standard_format(matches)
    
    def _extract_alternative_format(self, extracted_text: str) -> Dict:
        """Extract data from alternative format (2020)"""
        import re
        
        # Pattern: |Institute|Program|Quota|Seat Type|Gender|OR|CR|
        pattern = r'\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
        matches = re.findall(pattern, extracted_text, re.DOTALL)
        
        if not matches:
            logger.warning("No matches found with alternative format pattern")
            return {}
        
        logger.info(f"Found {len(matches)} rows in alternative format")
        return self._process_alternative_format(matches)
    
    def _process_standard_format(self, matches):
        """Process matches from standard format (2022-2024)"""
        import re
        institutes_data = {}
        
        for match in matches:
            sl_no, institute, program, seat_type, gender, opening_rank, closing_rank = match
            
            # Clean up the extracted data
            institute = institute.strip()
            program = re.sub(r'~~([^~]+)~~', r'\1', program).strip()  # Remove strikethrough
            program = re.sub(r'<br>', ' ', program)  # Replace <br> with space
            seat_type = seat_type.strip()
            gender = re.sub(r'~~([^~]+)~~', r'\1', gender).strip()  # Remove strikethrough
            gender = re.sub(r'<br>', ' ', gender)  # Replace <br> with space
            
            # Extract numeric values for ranks
            try:
                opening_rank = int(re.sub(r'[^0-9]', '', opening_rank))
                closing_rank = int(re.sub(r'[^0-9]', '', closing_rank))
            except ValueError:
                continue  # Skip rows with invalid rank data
            
            # Group by institute
            if institute not in institutes_data:
                institutes_data[institute] = []
            
            institutes_data[institute].append({
                "academic_program_name": program,
                "seat_type": seat_type,
                "gender": gender,
                "opening_rank": opening_rank,
                "closing_rank": closing_rank
            })
        
        return self._format_result(institutes_data)
    
    def _process_alternative_format(self, matches):
        """Process matches from alternative format (2020)"""
        import re
        institutes_data = {}
        
        for match in matches:
            institute, program, quota, seat_type, gender, opening_rank, closing_rank = match
            
            # Skip header rows and malformed sections
            if ('Opening Closing' in institute or 'Col2' in institute or 'Col3' in institute or
                'Institute Academic Program' in institute or institute.strip() == '---' or
                'Rank Rank' in institute or len(institute.strip()) < 3):
                continue
            
            # Skip rows where any field contains header-like text
            if any('Opening Closing' in field or 'Col2' in field or 'Col3' in field or
                   'Institute Academic Program' in field for field in match):
                continue
            
            # Clean up the extracted data
            institute = re.sub(r'<br>', ' ', institute).strip()
            program = re.sub(r'<br>', ' ', program).strip()
            quota = re.sub(r'<br>', ' ', quota).strip()
            seat_type = re.sub(r'<br>', ' ', seat_type).strip()
            gender = re.sub(r'<br>', ' ', gender).strip()
            
            # Skip if institute name is empty, looks like a number, or doesn't contain "Institute"
            if (not institute or institute.isdigit() or 
                'Institute' not in institute or len(institute) < 10):
                continue
            
            # Skip if program name is just a number or too short
            if not program or program.isdigit() or len(program) < 5:
                continue
            
            # Extract numeric values for ranks
            try:
                opening_rank_clean = re.sub(r'[^0-9]', '', opening_rank)
                closing_rank_clean = re.sub(r'[^0-9]', '', closing_rank)
                if opening_rank_clean and closing_rank_clean:
                    opening_rank = int(opening_rank_clean)
                    closing_rank = int(closing_rank_clean)
                    # Sanity check for reasonable rank values
                    if opening_rank < 1 or closing_rank < 1 or opening_rank > 100000 or closing_rank > 100000:
                        continue
                else:
                    continue
            except ValueError:
                continue  # Skip rows with invalid rank data
            
            # Group by institute
            if institute not in institutes_data:
                institutes_data[institute] = []
            
            institutes_data[institute].append({
                "academic_program_name": program,
                "quota": quota,
                "seat_type": seat_type,
                "gender": gender,
                "opening_rank": opening_rank,
                "closing_rank": closing_rank
            })
        
        return self._format_result(institutes_data)
    
    def _format_result(self, institutes_data):
        """Format the institutes data into the expected result structure"""
        result = {
            "institutes": [
                {
                    "institute_name": institute_name,
                    "programs": programs
                }
                for institute_name, programs in institutes_data.items()
            ]
        }
        
        total_programs = sum(len(programs) for programs in institutes_data.values())
        logger.info(f"Successfully extracted {len(institutes_data)} institutes with {total_programs} programs using regex parser")
        
        return result

    def extract_ranks_from_pdf(self, pdf_path: str, year: str = None) -> Dict:
        """
        Main method to extract IIT ranks from PDF
        
        Args:
            pdf_path: Path to the PDF file
            year: Academic year (optional)
            
        Returns:
            Dictionary containing extracted rank data
        """
        try:
            # Extract text from PDF
            extracted_text = self.extract_text_from_pdf(pdf_path)
            if not extracted_text.strip():
                logger.error("No text extracted from PDF")
                return {}

            # Save extracted text to file for analysis (with year prefix)
            sample_filename = f'extracted_text_sample_{year if year else "unknown"}.txt'
            with open(sample_filename, "w", encoding="utf-8") as f:
                f.write(extracted_text[:5000])  # Save first 5000 characters
            logger.info(f"Saved extracted text sample to {sample_filename}")

            # Convert PDF to images for visual analysis
            images = self.pdf_to_images(pdf_path)
            if not images:
                logger.error("No images extracted from PDF")
                return {}

            # Create extraction prompt
            prompt = self.create_extraction_prompt(extracted_text, year)

            # Extract using specified LLM provider
            if self.llm_provider == "gemini":
                # Try regex-based extraction first
                return self.extract_with_regex_parser(extracted_text)
            else:
                logger.error(f"LLM provider '{self.llm_provider}' not implemented yet")
                return {}

        except Exception as e:
            logger.error(f"Error in extract_ranks_from_pdf: {e}")
            return {}

    def save_extracted_data(self, data: Dict, output_path: str):
        """Save extracted data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def convert_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """Convert extracted data to pandas DataFrame for analysis"""
        try:
            rows = []
            for institute in data.get('institutes', []):
                institute_name = institute.get('institute_name', '')
                
                # Handle both old and new data structures
                if 'programs' in institute:
                    # New structure
                    for program in institute.get('programs', []):
                        row = {
                            'year': data.get('year', ''),
                            'institute_name': institute_name,
                            'academic_program_name': program.get('academic_program_name', ''),
                            'seat_type': program.get('seat_type', ''),
                            'gender': program.get('gender', ''),
                            'opening_rank': program.get('opening_rank'),
                            'closing_rank': program.get('closing_rank')
                        }
                        rows.append(row)
                elif 'courses' in institute:
                    # Old structure
                    for course in institute.get('courses', []):
                        for category, ranks in course.get('category_ranks', {}).items():
                            row = {
                                'year': data.get('year', ''),
                                'institute_name': institute_name,
                                'academic_program_name': course.get('course_name', ''),
                                'seat_type': category,
                                'gender': 'Gender-Neutral',
                                'opening_rank': ranks.get('opening_rank'),
                                'closing_rank': ranks.get('closing_rank')
                            }
                            rows.append(row)
            
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {e}")
            return pd.DataFrame()


# Example usage and configuration
def main():
    """Main function to extract IIT rank data from recent years (2022-2024) with reliable extraction"""
    
    import sys
    
    # Get API key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please add your Gemini API key to the .env file.")
        return
    
    # Initialize the extractor
    extractor = IITRankExtractor(
        gemini_api_key=gemini_api_key,
        llm_provider="gemini"
    )
    
    # PDF directory path
    pdf_dir = "data/pdfs"
    
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return
    
    # Check if year is provided as command line argument
    if len(sys.argv) > 1:
        # Process only the specified year
        target_years = [sys.argv[1]]
        print(f"\n=== EXTRACTING RANKS FOR {target_years[0]} ===")
    else:
        # Default behavior: process all recent years
        target_years = ['2022', '2023', '2024']
        print(f"Processing recent years with reliable extraction: {target_years}")
    
    # Ensure output directory exists
    output_dir = "data/extracted"
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    for year in target_years:
        pdf_file = f"{year}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        if not os.path.exists(pdf_path):
            print(f"PDF file for {year} not found: {pdf_path}")
            continue
            
        print(f"\nProcessing {pdf_file} for year {year}...")
        
        # Extract data from PDF
        extracted_data = extractor.extract_ranks_from_pdf(pdf_path, year)
        
        if extracted_data:
            # Update year in extracted data
            extracted_data['year'] = year
            
            # Save individual JSON file
            json_output_path = os.path.join(output_dir, f"iit_ranks_{year}.json")
            extractor.save_extracted_data(extracted_data, json_output_path)
            
            # Convert to DataFrame
            df = extractor.convert_to_dataframe(extracted_data)
            if not df.empty:
                # Save individual CSV file
                csv_output_path = os.path.join(output_dir, f"iit_ranks_{year}.csv")
                df.to_csv(csv_output_path, index=False)
                print(f"Extracted {len(df)} rank records for {year}")
                
                # Add to combined data
                all_data.append(df)
            else:
                print(f"No data converted to DataFrame for {year}")
        else:
            print(f"No data extracted from {pdf_file}")
    
    # Combine all data and save (only if processing multiple years)
    if all_data and len(target_years) > 1:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv_path = os.path.join(output_dir, "iit_ranks_recent_years.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        
        print(f"\n=== SUMMARY ===")
        print(f"Total records extracted: {len(combined_df)}")
        print(f"Years processed: {sorted(combined_df['year'].unique())}")
        print(f"Institutes found: {len(combined_df['institute_name'].unique())}")
        print(f"\nSample data:")
        print(combined_df.head())
        print(f"\nCombined data saved to {combined_csv_path}")
    elif len(target_years) == 1 and all_data:
        # Just print summary for single year
        df = all_data[0]
        print(f"\n=== EXTRACTION COMPLETE ===\n")
        print(f"Total records extracted for {target_years[0]}: {len(df)}")
        print(f"Institutes found: {len(df['institute_name'].unique())}")
    else:
        print("No data extracted from any PDF files")


if __name__ == "__main__":
    main()