import pymupdf4llm
import re
import pandas as pd

def analyze_missing_rows(year):
    """Analyze missing rows to understand why they're not being captured"""
    pdf_path = f"data/pdfs/{year}.pdf"
    
    print(f"\n=== DEBUGGING {year} ===")
    
    # Extract text from PDF
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # Find all serial numbers and their corresponding lines
    lines = md_text.split('\n')
    serial_lines = []
    
    for i, line in enumerate(lines):
        if re.match(r'^\|(\d+)\|', line):
            serial_match = re.match(r'^\|(\d+)\|', line)
            if serial_match:
                serial_num = int(serial_match.group(1))
                serial_lines.append((serial_num, line, i+1))
    
    # Sort by serial number
    serial_lines.sort(key=lambda x: x[0])
    
    print(f"Total lines with serial numbers: {len(serial_lines)}")
    
    if serial_lines:
        min_serial = serial_lines[0][0]
        max_serial = serial_lines[-1][0]
        expected_range = list(range(min_serial, max_serial + 1))
        found_serials = [x[0] for x in serial_lines]
        missing_serials = [x for x in expected_range if x not in found_serials]
        
        print(f"Serial range: {min_serial} to {max_serial}")
        print(f"Missing serials: {len(missing_serials)}")
        
        if missing_serials:
            print(f"First 10 missing: {missing_serials[:10]}")
            
            # Try to find these missing serials in the text
            print("\n--- Searching for missing serials in text ---")
            for missing in missing_serials[:5]:  # Check first 5 missing
                print(f"\nLooking for serial {missing}:")
                
                # Search for the missing serial number in various patterns
                patterns = [
                    rf'\|{missing}\|',  # Standard pattern
                    rf'\|\s*{missing}\s*\|',  # With spaces
                    rf'{missing}\s*\|',  # Without leading pipe
                    rf'\|{missing}[^\|]*\|',  # With extra content
                    str(missing)  # Just the number
                ]
                
                found_contexts = []
                for pattern in patterns:
                    matches = re.finditer(pattern, md_text)
                    for match in matches:
                        start = max(0, match.start() - 100)
                        end = min(len(md_text), match.end() + 100)
                        context = md_text[start:end].replace('\n', '\\n')
                        found_contexts.append(f"Pattern '{pattern}': ...{context}...")
                
                if found_contexts:
                    for context in found_contexts[:2]:  # Show first 2 contexts
                        print(f"  {context}")
                else:
                    print(f"  Serial {missing} not found in text")
    
    # Test current regex pattern
    print("\n--- Testing current regex pattern ---")
    current_pattern = r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
    current_matches = re.findall(current_pattern, md_text, re.DOTALL)
    print(f"Current pattern matches: {len(current_matches)}")
    
    # Test more flexible patterns
    print("\n--- Testing alternative patterns ---")
    
    patterns_to_test = [
        ("Flexible spaces", r'\|\s*(\d+)\s*\|([^|]*?)\|([^|]*?)\|([^|]*?)\|([^|]*?)\|([^|]*?)\|([^|]*?)\|'),
        ("Multiline content", r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'),
        ("Any content", r'\|(\d+)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|'),
        ("Non-greedy", r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|')
    ]
    
    for name, pattern in patterns_to_test:
        try:
            matches = re.findall(pattern, md_text, re.DOTALL | re.MULTILINE)
            print(f"{name}: {len(matches)} matches")
            
            if matches and len(matches) > len(current_matches):
                print(f"  -> This pattern captures {len(matches) - len(current_matches)} more rows!")
                # Show a sample of the additional matches
                additional_serials = [int(m[0]) for m in matches if int(m[0]) not in [int(cm[0]) for cm in current_matches]]
                if additional_serials:
                    print(f"  -> Additional serials found: {sorted(additional_serials)[:10]}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    return len(serial_lines), len(missing_serials)

def main():
    print("=== DEBUGGING MISSING ROWS ===")
    
    for year in [2022, 2024]:
        total_found, missing_count = analyze_missing_rows(year)
        print(f"\n{year} Summary: Found {total_found} rows, Missing {missing_count} rows")
        print("=" * 60)

if __name__ == "__main__":
    main()