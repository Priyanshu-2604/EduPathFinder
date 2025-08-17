import pymupdf4llm
import re

def examine_table_around_missing(year, missing_serial):
    """Examine the table structure around a missing serial number"""
    pdf_path = f"data/pdfs/{year}.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    print(f"\n=== EXAMINING TABLE STRUCTURE AROUND SERIAL {missing_serial} IN {year} ===")
    
    # Find the context around the missing serial
    lines = md_text.split('\n')
    
    # Find lines that contain the missing serial
    serial_line_indices = []
    for i, line in enumerate(lines):
        if str(missing_serial) in line and '|' in line:
            serial_line_indices.append(i)
    
    if not serial_line_indices:
        print(f"Serial {missing_serial} not found in any table-like lines")
        return
    
    # Examine context around each occurrence
    for idx in serial_line_indices:
        print(f"\n--- Context around line {idx+1} ---")
        
        # Show 5 lines before and after
        start = max(0, idx - 5)
        end = min(len(lines), idx + 6)
        
        for i in range(start, end):
            marker = " >>> " if i == idx else "     "
            pipe_count = lines[i].count('|')
            print(f"{marker}Line {i+1:4d} (pipes:{pipe_count:2d}): {lines[i][:100]}")
            if len(lines[i]) > 100:
                print(f"                    ... (truncated, total length: {len(lines[i])})")

def find_table_breaks(year):
    """Find where the table structure breaks"""
    pdf_path = f"data/pdfs/{year}.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    print(f"\n=== FINDING TABLE BREAKS IN {year} ===")
    
    lines = md_text.split('\n')
    
    # Find all lines that look like table rows (start with |number|)
    table_rows = []
    for i, line in enumerate(lines):
        if re.match(r'^\|\d+\|', line):
            pipe_count = line.count('|')
            serial_match = re.match(r'^\|(\d+)\|', line)
            serial = int(serial_match.group(1)) if serial_match else None
            table_rows.append((i, serial, pipe_count, line))
    
    print(f"Found {len(table_rows)} table rows")
    
    # Look for inconsistencies
    print("\n--- Table structure analysis ---")
    
    # Group by pipe count
    pipe_counts = {}
    for row in table_rows:
        pipe_count = row[2]
        if pipe_count not in pipe_counts:
            pipe_counts[pipe_count] = []
        pipe_counts[pipe_count].append(row)
    
    for pipe_count, rows in sorted(pipe_counts.items()):
        print(f"Rows with {pipe_count} pipes: {len(rows)}")
        if pipe_count != 8:  # 8 pipes = 7 columns
            print(f"  -> These are problematic! Expected 8 pipes.")
            # Show first few examples
            for i, (line_idx, serial, _, line) in enumerate(rows[:3]):
                print(f"     Example {i+1}: Line {line_idx+1}, Serial {serial}: {line[:80]}...")
    
    # Look for serial number gaps
    serials = sorted([row[1] for row in table_rows if row[1] is not None])
    gaps = []
    for i in range(len(serials) - 1):
        if serials[i+1] - serials[i] > 1:
            gap_start = serials[i] + 1
            gap_end = serials[i+1] - 1
            gaps.extend(range(gap_start, gap_end + 1))
    
    print(f"\nSerial number gaps: {len(gaps)}")
    if gaps:
        print(f"First 10 gaps: {gaps[:10]}")
        
        # For each gap, try to find where it might be
        for gap_serial in gaps[:5]:
            print(f"\n--- Looking for missing serial {gap_serial} ---")
            
            # Search for this serial in non-table lines
            for i, line in enumerate(lines):
                if str(gap_serial) in line and not re.match(r'^\|\d+\|', line):
                    pipe_count = line.count('|')
                    print(f"  Found in line {i+1} (pipes:{pipe_count}): {line[:80]}...")
                    
                    # Show context
                    if pipe_count > 0:  # If it has pipes, show context
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        print(f"    Context:")
                        for j in range(start, end):
                            marker = "    >>> " if j == i else "        "
                            print(f"{marker}{lines[j][:60]}...")
                    break

def main():
    import sys
    
    # Check if year is provided as command line argument
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        print(f"\n=== EXAMINING TABLE STRUCTURE FOR {year} ===")
        
        # Find table breaks for the specified year
        find_table_breaks(year)
        
        # For specific years, examine known problematic serials
        if year == 2022:
            examine_table_around_missing(year, 2770)
        elif year == 2024:
            examine_table_around_missing(year, 199)
    else:
        # Default behavior when no year is specified
        print("\n=== EXAMINING TABLE STRUCTURE FOR ALL YEARS ===")
        # Examine specific missing serials
        examine_table_around_missing(2022, 2770)
        examine_table_around_missing(2024, 199)
        
        # Find table breaks
        find_table_breaks(2022)
        find_table_breaks(2023)
        find_table_breaks(2024)

if __name__ == "__main__":
    main()