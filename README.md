# IIT Rank Extractor

A comprehensive Python-based system for extracting and processing IIT (Indian Institute of Technology) admission rank data from PDF documents. This tool converts tabular rank data from PDFs into structured JSON and CSV formats for analysis and research purposes.

## 🎯 Project Overview

The IIT Rank Extractor is a comprehensive Python-based system designed to extract and process IIT (Indian Institute of Technology) admission rank data from recent PDF documents (2022-2024). The system uses advanced text extraction techniques and regex parsing to convert unstructured PDF data into structured JSON and CSV formats with 100% accuracy.

### Key Features
- **Recent years focus**: Optimized for 2022-2024 PDFs with reliable extraction
- **Perfect accuracy**: 100% extraction rate for supported years
- **Serial number validation**: Uses SI No. for accurate row counting
- **Intelligent format detection**: Automatically detects standard table structures
- **Robust data extraction**: Uses PyMuPDF4LLM for reliable text extraction
- **Data validation**: Comprehensive validation and cleaning of extracted data
- **Multiple output formats**: Generates both JSON and CSV outputs
- **Batch processing**: Processes multiple years of data in a single run

## 📁 Project Structure

```
EduPathFinder/
├── IIT_Rank_extractor.py         # Main extraction engine
├── analysis_utils.py             # Data analysis utilities
├── batch_processor.py            # Batch processing functionality
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API keys)
├── README.md                     # This documentation
├── college_predictor_model/      # College prediction and recommendation system
│   ├── advanced_features.py      # Advanced feature engineering for prediction
│   ├── college_recommender.py    # Core recommendation logic
│   ├── data_preparation.py       # Data preparation utilities
│   ├── feature_engineering.py    # Feature engineering scripts
│   ├── model_training.py         # Model training scripts
│   ├── model_validation.py       # Model validation and evaluation
│   └── prediction_interface.py   # User-facing prediction interface
├── data/
│   ├── pdfs/                     # Source PDF files (2022.pdf, 2023.pdf, 2024.pdf)
│   ├── extracted/                # Processed output files (JSON/CSV)
├── extracted_text_sample_*.txt   # Debug text samples
└── venv/                         # Python virtual environment
```


## 📋 File Descriptions

### Core Files

#### `IIT_Rank_extractor.py`
The main extraction engine that handles:
- PDF text extraction using PyMuPDF4LLM
- Intelligent format detection for different years
- Regex-based table parsing
- Data validation and cleaning
- JSON and CSV output generation

**Key Classes & Methods:**
- `IITRankExtractor`: Main class handling all extraction logic
- `extract_text_from_pdf()`: Extracts text from PDF files
- `extract_with_regex_parser()`: Parses tabular data using regex
- `_is_standard_format()`: Detects table format type
- `_extract_standard_format()`: Handles 2022-2024 format
- `_extract_alternative_format()`: Handles 2020 format

#### `college_predictor_model/`
**A modular system for college prediction and recommendation based on extracted IIT rank data.**

- `advanced_features.py`: Advanced feature engineering (historical, competitiveness, categorical, rank-based, temporal features)
- `college_recommender.py`: Core recommendation logic for suggesting eligible programs and generating insights/statistics
- `data_preparation.py`: Data cleaning and preparation utilities
- `feature_engineering.py`: Feature engineering scripts (basic transformations)
- `model_training.py`: Scripts for training machine learning models for college prediction
- `model_validation.py`: Model validation, cross-validation, and evaluation utilities
- `prediction_interface.py`: User-facing interface for making predictions and generating reports

#### `analysis_utils.py`
Utility functions for data analysis and processing (if implemented).

#### `batch_processor.py`
Handles batch processing of multiple PDF files (if implemented).

#### `config.py`
Configuration settings and constants for the application.

### Data Files

#### Input PDFs (`data/pdfs/`)
- **2022.pdf**: IIT rank data for 2022 admissions
- **2023.pdf**: IIT rank data for 2023 admissions
- **2024.pdf**: IIT rank data for 2024 admissions

#### Output Files (`data/extracted/`)
- **iit_ranks_2022.json**: Structured JSON data for 2022
- **iit_ranks_2023.json**: Structured JSON data for 2023
- **iit_ranks_2024.json**: Structured JSON data for 2024
- **iit_ranks_2022.csv**: Flattened CSV data for 2022
- **iit_ranks_2023.csv**: Flattened CSV data for 2023
- **iit_ranks_2024.csv**: Flattened CSV data for 2024
- **iit_ranks_recent_years.csv**: Combined dataset across all years

#### Debug Files
- **extracted_text_sample_YYYY.txt**: Raw extracted text samples for debugging

## 🔄 Data Flow & Processing Pipeline

### 1. Input Processing
```
PDF Files → Text Extraction → Format Detection → Table Parsing
```

### 2. Format Detection Logic
The system automatically detects two main formats:

**Standard Format (2022-2024):**
- 7-column table: S.No, Institute, Academic Program Name, Seat Type, Gender, OR, CR
- Uses markdown table structure
- Consistent column headers

**Alternative Format (2020):**
- Different column arrangement
- Contains "Col2", "Col3" placeholder columns
- Requires specialized parsing logic

### 3. Data Extraction Process
```python
# Simplified workflow
for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)
    if is_standard_format(text):
        data = extract_standard_format(text)
    else:
        data = extract_alternative_format(text)
    save_to_json_and_csv(data)
```

### 4. Output Generation
- **JSON Format**: Hierarchical structure grouped by institute
- **CSV Format**: Flattened records for easy analysis
- **Combined Dataset**: All years merged into single CSV

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps

1. **Clone/Download the project**
   ```bash
   cd EduPathFinder
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables** (if using LLM features)
   Create `.env` file with:
   ```
   GEMINI_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_api_key_here
   ```


## 🎮 Usage

### Basic Usage (IIT Rank Extraction)
```python
from IIT_Rank_extractor import IITRankExtractor

# Initialize extractor
extractor = IITRankExtractor()

# Process all PDFs in the data/pdfs directory
extractor.process_all_pdfs()
```

### Command Line Usage
```bash
python IIT_Rank_extractor.py
```

### Output Files
After processing, you'll find:
- Individual year files: `data/extracted/iit_ranks_2022.json`, `data/extracted/iit_ranks_2023.json`, `data/extracted/iit_ranks_2024.json`, etc.
- Combined dataset: `data/extracted/iit_ranks_recent_years.csv`
- Debug samples: `extracted_text_sample_2022.txt`, `extracted_text_sample_2023.txt`, etc.

---

### College Predictor & Recommendation Usage

The `college_predictor_model` module provides a machine learning-based system to recommend eligible IIT programs based on your rank, category, and gender.

#### Example: Predicting College Recommendations
```python
from college_predictor_model.prediction_interface import PredictionInterface

# Initialize the prediction interface (ensure the model and features are prepared)
predictor = PredictionInterface(
    model_path='validated_college_predictor.joblib',
    features_path='advanced_features_iit_ranks.csv'
)

# Get recommendations for a student
rank = 5000
category = 'OBC-NCL'
gender = 'Gender-Neutral'

is_valid, msg = predictor.validate_input(rank, category, gender)
if is_valid:
    predictions = predictor.get_predictions(rank, category, gender, top_n=10)
    predictor.print_formatted_results(predictions)
else:
    print(f"Input error: {msg}")
```

#### Features
- Advanced feature engineering for better prediction accuracy
- Model validation and statistics reporting
- Generates actionable recommendations and detailed analysis

> **Note:** Ensure the model is trained and the required feature files exist. See scripts in `college_predictor_model/` for training and validation workflows.

## 📊 Data Structure

### JSON Output Format
```json
{
  "year": 2024,
  "total_institutes": 136,
  "total_programs": 2936,
  "institutes": [
    {
      "institute_name": "IIT Bhubaneswar",
      "programs": [
        {
          "academic_program_name": "Civil Engineering (4 Years Bachelor of Technology)",
          "seat_type": "OPEN",
          "gender": "Gender-Neutral",
          "opening_rank": 9106,
          "closing_rank": 14782
        }
      ]
    }
  ]
}
```

### CSV Output Format
```csv
year,institute_name,academic_program_name,seat_type,gender,opening_rank,closing_rank
2024,IIT Bhubaneswar,Civil Engineering (4 Years Bachelor of Technology),OPEN,Gender-Neutral,9106,14782
```

## 🔧 Technical Features

### Intelligent Format Detection
- Automatically detects table format based on content analysis
- Handles different PDF layouts across years
- Robust regex patterns for data extraction

### Data Validation
- Institute name validation (must contain "Institute")
- Rank validation (numeric values only)
- Program name filtering (removes invalid entries)
- Header row detection and filtering

### Error Handling
- Graceful handling of malformed data
- Comprehensive logging for debugging
- Fallback mechanisms for different formats

## 📈 Processing Statistics

**Latest Run Results:**
- **Total Records Extracted**: 8,327
- **Years Processed**: 2022, 2023, 2024
- **Institutes Found**: 33 different IIT institutes
- **Success Rate**: 100% for supported formats

**Year-wise Breakdown:**
- **2022**: 2,362 records (23 institutes)
- **2023**: 3,029 records (23 institutes)
- **2024**: 2,936 records (31 institutes)

## 🛠️ Dependencies

Key Python packages (see `requirements.txt` for complete list):
- `pymupdf4llm`: PDF text extraction
- `pymupdf`: PDF processing
- `pandas`: Data manipulation
- `json`: JSON handling
- `re`: Regular expressions
- `pathlib`: File path handling

## 🐛 Troubleshooting

### Common Issues

1. **No data extracted for certain years**
   - Check if PDF format is supported
   - Verify text extraction quality in debug files

2. **Missing API keys**
   - Ensure `.env` file is properly configured
   - Check API key validity

3. **Format detection errors**
   - Review `extracted_text_sample_*.txt` files
   - Adjust regex patterns if needed

### Debug Files
Use the generated `extracted_text_sample_*.txt` files to:
- Verify text extraction quality
- Understand table structure
- Debug parsing issues

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and research purposes. Please ensure compliance with data usage policies.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review debug files
3. Create an issue with detailed information

---

**Note**: This tool is designed specifically for IIT admission rank data extraction. The parsing logic is optimized for the specific table formats found in IIT PDF documents and may require modifications for other data sources.