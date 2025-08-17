import pymupdf4llm
import re

def analyze_specific_missing_rows(year, missing_serials):
    """Analyze specific missing serial numbers to understand their patterns"""
    pdf_path = f"data/pdfs/{year}.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    print(f"\n=== ANALYZING SPECIFIC PATTERNS FOR {year} ===")
    
    for serial in missing_serials[:5]:  # Analyze first 5 missing
        print(f"\n--- Serial {serial} ---")
        
        # Find all occurrences of this serial number
        pattern = rf'\|{serial}\|[^\n]*'
        matches = re.findall(pattern, md_text)
        
        if matches:
            for i, match in enumerate(matches):
                print(f"Match {i+1}: {match}")
                
                # Count pipes to see if it's a complete row
                pipe_count = match.count('|')
                print(f"  Pipe count: {pipe_count} (expected: 8 for 7 columns)")
                
                # Check for formatting issues
                if '<br>' in match:
                    print(f"  Contains <br> tags")
                if '~~' in match:
                    print(f"  Contains strikethrough (~~)")
                if '\n' in match:
                    print(f"  Contains newlines")
        else:
            print(f"No direct matches found for |{serial}|")
            
            # Look for the serial in a broader context
            broader_pattern = rf'{serial}[^\d]*\|'
            broader_matches = re.findall(broader_pattern, md_text)
            if broader_matches:
                print(f"Found in broader context: {broader_matches[:3]}")

def test_improved_regex(year):
    """Test improved regex patterns that handle formatting issues"""
    pdf_path = f"data/pdfs/{year}.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    print(f"\n=== TESTING IMPROVED REGEX FOR {year} ===")
    
    # Current pattern
    current_pattern = r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
    current_matches = re.findall(current_pattern, md_text, re.DOTALL)
    print(f"Current pattern: {len(current_matches)} matches")
    
    # Improved patterns that handle formatting issues
    improved_patterns = [
        # Handle <br> tags and strikethrough
        ("Handle <br> and ~~", r'\|(\d+)\|([^|]*(?:<br>[^|]*)*?)\|([^|]*(?:<br>[^|]*)*?)\|([^|]*(?:<br>[^|]*)*?)\|([^|]*(?:<br>[^|]*)*?)\|([^|]*?)\|([^|]*?)\|'),
        
        # More flexible whitespace and newlines
        ("Flexible whitespace", r'\|\s*(\d+)\s*\|([^|]*?)\|([^|]*?)\|([^|]*?)\|([^|]*?)\|([^|]*?)\|([^|]*?)\|'),
        
        # Handle multiline cells with newlines
        ("Multiline cells", r'\|(\d+)\|([^|]*(?:\n[^|]*)*?)\|([^|]*(?:\n[^|]*)*?)\|([^|]*(?:\n[^|]*)*?)\|([^|]*(?:\n[^|]*)*?)\|([^|]*?)\|([^|]*?)\|'),
        
        # Combined: handle all formatting issues
        ("Combined improved", r'\|\s*(\d+)\s*\|([^|]*(?:<br>\s*[^|]*|\n\s*[^|]*)*?)\|([^|]*(?:<br>\s*[^|]*|\n\s*[^|]*)*?)\|([^|]*(?:<br>\s*[^|]*|\n\s*[^|]*)*?)\|([^|]*(?:<br>\s*[^|]*|\n\s*[^|]*)*?)\|([^|]*?)\|([^|]*?)\|')
    ]
    
    best_pattern = None
    best_count = len(current_matches)
    
    for name, pattern in improved_patterns:
        try:
            matches = re.findall(pattern, md_text, re.DOTALL | re.MULTILINE)
            print(f"{name}: {len(matches)} matches")
            
            if len(matches) > best_count:
                best_count = len(matches)
                best_pattern = (name, pattern)
                
                # Show some of the additional matches
                current_serials = set(int(m[0]) for m in current_matches)
                new_serials = [int(m[0]) for m in matches if int(m[0]) not in current_serials]
                if new_serials:
                    print(f"  -> Additional serials: {sorted(new_serials)[:10]}")
                    
                    # Show a sample of the improved match
                    for match in matches:
                        if int(match[0]) in new_serials[:3]:
                            print(f"  -> Sample: |{match[0]}|{match[1][:30]}...|{match[2][:20]}...|")
                            
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    if best_pattern:
        print(f"\nBest pattern: {best_pattern[0]} with {best_count} matches")
        return best_pattern[1]
    else:
        print(f"\nNo improvement found. Current pattern is best.")
        return current_pattern

def main():
    import sys
    
    # Define known missing data for specific years
    missing_data = {
        2022: [2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775],  # From previous output
        2023: [],  # Add missing data if known
        2024: [196, 197, 198, 199, 200, 201, 203, 204, 205, 206]  # From previous output
    }
    
    # Check if year is provided as command line argument
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        print(f"\n=== ANALYZING PATTERNS FOR {year} ===")
        
        # Analyze missing rows if we have data for this year
        if year in missing_data and missing_data[year]:
            analyze_specific_missing_rows(year, missing_data[year])
        
        # Test improved regex patterns
        best_pattern = test_improved_regex(year)
        print(f"\nRecommended pattern for {year}: {best_pattern}")
    else:
        # Default behavior when no year is specified
        print("\n=== ANALYZING PATTERNS FOR ALL YEARS ===")
        for year in [2022, 2023, 2024]:
            if year in missing_data and missing_data[year]:
                analyze_specific_missing_rows(year, missing_data[year])
            best_pattern = test_improved_regex(year)
            print(f"\nRecommended pattern for {year}: {best_pattern}")
            print("=" * 80)

if __name__ == "__main__":
    main()