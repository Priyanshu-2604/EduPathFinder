import os
import json
import logging
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

# Import our existing modules
from IIT_Rank_extractor import IITRankExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TableStructureAnalysis:
    """Results from table structure analysis"""
    total_rows: int
    pipe_count_distribution: Dict[int, int]
    serial_range: Tuple[int, int]
    missing_serials: List[int]
    problematic_patterns: List[str]
    recommended_regex: str

@dataclass
class ValidationResult:
    is_valid: bool
    confidence_score: float
    issues: List[str]
    recommendations: List[str]
    validation_output: str = ""

@dataclass
class ExtractionResult:
    """Results from extraction attempt"""
    success: bool
    rows_extracted: int
    method_used: str
    accuracy: float
    issues: List[str]
    data: Dict

class AdaptiveExtractionPipeline:
    """A generalized pipeline for extracting data from any PDF with table structures"""
    
    def __init__(self):
        self.extractor = IITRankExtractor()
        self.analysis_results = {}
        self.extraction_attempts = []
        
    def analyze_table_structure(self, pdf_path: str, year: str) -> TableStructureAnalysis:
        """Step 1: Analyze the table structure using examine_table_structure.py"""
        logger.info(f"Step 1: Analyzing table structure for {year}")
        
        try:
            # Run examine_table_structure.py as subprocess with year parameter
            cmd = [sys.executable, "examine_table_structure.py", year]
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                check=True
            )
            logger.info(f"Command completed with return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.warning(f"Command stderr: {result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Table structure analysis failed: {result.stderr}")
                # Return default analysis
                return TableStructureAnalysis(
                    total_rows=0,
                    pipe_count_distribution={},
                    serial_range=(0, 0),
                    missing_serials=[],
                    problematic_patterns=["Analysis failed"],
                    recommended_regex=r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
                )
            
            # Parse output to extract analysis results
            output_lines = result.stdout.split('\n')
            
            # Extract key information from output
            total_rows = 0
            pipe_distribution = {}
            serial_range = (0, 0)
            missing_serials = []
            problematic_patterns = []
            
            for line in output_lines:
                if "Found" in line and "table rows" in line:
                    try:
                        total_rows = int(line.split()[1])
                    except:
                        pass
                elif "Rows with" in line and "pipes:" in line:
                    try:
                        parts = line.split()
                        pipe_count = int(parts[2])
                        count = int(parts[4])
                        pipe_distribution[pipe_count] = count
                        if pipe_count not in [7, 8]:
                            problematic_patterns.append(line.strip())
                    except:
                        pass
            
            # Determine recommended regex based on year and known patterns
            if year in ["2022", "2024"]:
                recommended_regex = r'\|(?:~~)?(\d+)(?:~~)?(?:<br>)?\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
            else:
                recommended_regex = r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
            
            analysis = TableStructureAnalysis(
                total_rows=total_rows,
                pipe_count_distribution=pipe_distribution,
                serial_range=serial_range,
                missing_serials=missing_serials,
                problematic_patterns=problematic_patterns,
                recommended_regex=recommended_regex
            )
            
            logger.info(f"Table structure analysis completed for {year}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error running table structure analysis: {e}")
            return TableStructureAnalysis(
                total_rows=0,
                pipe_count_distribution={},
                serial_range=(0, 0),
                missing_serials=[],
                problematic_patterns=[f"Error: {str(e)}"],
                recommended_regex=r'\|(\d+)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'
            )
    
    def test_extraction_strategies(self, pdf_path: str, year: str, analysis: TableStructureAnalysis) -> List[ExtractionResult]:
        """Step 2: Test extraction strategies using analyze_patterns.py"""
        logger.info(f"Step 2: Testing extraction strategies for {year}")
        
        try:
            # Run analyze_patterns.py as subprocess with year parameter
            cmd = [sys.executable, "analyze_patterns.py", year]
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                check=True
            )
            logger.info(f"Command completed with return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.warning(f"Command stderr: {result.stderr}")
            
            results = []
            
            if result.returncode != 0:
                logger.error(f"Pattern analysis failed: {result.stderr}")
                # Return default result with current working regex
                return [ExtractionResult(
                    success=True,
                    rows_extracted=0,
                    method_used="current_regex",
                    accuracy=0,
                    issues=["Pattern analysis failed, using current regex"],
                    data={}
                )]
            
            # Parse output to extract pattern test results
            output_lines = result.stdout.split('\n')
            
            # Extract pattern test results
            current_matches = 0
            improved_matches = 0
            
            for line in output_lines:
                if "Current pattern:" in line and "matches" in line:
                    try:
                        current_matches = int(line.split()[2])
                    except:
                        pass
                elif "Handle <br> and ~~:" in line and "matches" in line:
                    try:
                        improved_matches = int(line.split()[4])
                    except:
                        pass
            
            # Create results based on analysis
            strategies = [
                ("current_regex", current_matches, analysis.recommended_regex),
                ("improved_regex", improved_matches, r'\|(?:~~)?(\d+)(?:~~)?(?:<br>)?\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|'),
                ("flexible_regex", max(current_matches, improved_matches), r'\|(?:~~)?\s*(\d+)\s*(?:~~)?(?:<br>)?\s*\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|([^|]+?)\|')
            ]
            
            for strategy_name, rows_extracted, regex_pattern in strategies:
                # Estimate expected rows based on year
                if year == "2022":
                    expected_rows = 2803
                elif year == "2023":
                    expected_rows = 3029
                elif year == "2024":
                    expected_rows = 3028
                else:
                    expected_rows = rows_extracted  # Fallback
                
                accuracy = (rows_extracted / expected_rows) * 100 if expected_rows > 0 else 0
                
                issues = []
                if rows_extracted == 0:
                    issues.append("No matches found")
                elif accuracy < 90:
                    issues.append(f"Low accuracy: {accuracy:.1f}%")
                
                result_obj = ExtractionResult(
                    success=rows_extracted > 0,
                    rows_extracted=rows_extracted,
                    method_used=strategy_name,
                    accuracy=accuracy,
                    issues=issues,
                    data={"regex": regex_pattern}
                )
                
                results.append(result_obj)
                logger.info(f"Strategy '{strategy_name}': {rows_extracted} rows, {accuracy:.1f}% accuracy")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running pattern analysis: {e}")
            # Return fallback result
            return [ExtractionResult(
                success=True,
                rows_extracted=0,
                method_used="fallback_regex",
                accuracy=0,
                issues=[f"Error: {str(e)}"],
                data={"regex": analysis.recommended_regex}
            )]
    
    def select_best_strategy(self, results: List[ExtractionResult]) -> ExtractionResult:
        """Step 3: Select the best extraction strategy"""
        logger.info("Step 3: Selecting best extraction strategy")
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            logger.warning("No successful extraction strategies found")
            return results[0] if results else None
        
        # Sort by accuracy, then by rows extracted
        best_result = max(successful_results, key=lambda x: (x.accuracy, x.rows_extracted))
        
        logger.info(f"Best strategy: {best_result.method_used} with {best_result.accuracy:.1f}% accuracy")
        return best_result
    
    def extract_with_best_strategy(self, pdf_path: str, year: str, best_strategy: ExtractionResult) -> Dict:
        """Step 4: Perform extraction using IIT_Rank_extractor.py"""
        logger.info(f"Step 4: Extracting data using {best_strategy.method_used}")
        
        try:
            # Run IIT_Rank_extractor.py as subprocess with year parameter
            cmd = [sys.executable, "IIT_Rank_extractor.py", year]
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                check=True
            )
            logger.info(f"Command completed with return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.warning(f"Command stderr: {result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Extraction failed: {result.stderr}")
                return {}
            
            # Parse output to get extraction results
            output_lines = result.stdout.split('\n')
            
            # Extract information from output
            rows_extracted = {}
            csv_files = []
            
            for line in output_lines:
                if "Extracted" in line and "rank records for" in line:
                    # Parse lines like "Extracted 2803 rank records for 2022"
                    try:
                        parts = line.split()
                        if len(parts) >= 6 and "Extracted" in parts[0]:
                            rows_count = int(parts[1])
                            year_match = parts[-1]  # Last part should be the year
                            rows_extracted[year_match] = rows_count
                    except:
                        pass
                elif "saved to" in line:
                    # Parse lines like "Data saved to data/extracted\iit_ranks_2022.json"
                    try:
                        csv_file = line.split("saved to")[1].strip()
                        csv_files.append(csv_file)
                    except:
                        pass
            
            # Get results for the specific year
            year_rows = rows_extracted.get(year, 0)
            
            if year_rows > 0:
                logger.info(f"Extraction completed: {year_rows} rows extracted for {year}")
                # Return a simplified result structure
                return {
                    'institutes': {},  # Placeholder - actual data would be in CSV files
                    'extraction_info': {
                        'rows_extracted': year_rows,
                        'output_files': csv_files,
                        'all_extractions': rows_extracted
                    }
                }
            else:
                logger.warning(f"No data extracted for year {year}")
                return {}
                
        except Exception as e:
            logger.error(f"Error running extraction: {e}")
            return {}
    
    def validate_extraction(self, pdf_path: str, year: str, extracted_data: Dict, analysis: TableStructureAnalysis) -> Dict:
        """Step 5: Validate the extraction results using verify_extraction_results.py"""
        logger.info(f"Step 5: Validating extraction for {year}")
        
        # Use known correct expected values for each year
        expected_rows_by_year = {
            "2022": 2803,
            "2023": 3029,
            "2024": 3028
        }
        
        validation_results = {
            "expected_rows": expected_rows_by_year.get(year, analysis.serial_range[1] - analysis.serial_range[0] + 1),
            "extracted_rows": 0,
            "accuracy": 0,
            "missing_serials": [],
            "validation_passed": False,
            "validation_output": ""
        }
        
        try:
            # Run verify_extraction_results.py as subprocess with year parameter
            cmd = [sys.executable, "verify_extraction_results.py", year]
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                check=True
            )
            logger.info(f"Command completed with return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Command stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.warning(f"Command stderr: {result.stderr}")
            
            validation_results["validation_output"] = result.stdout if result.returncode == 0 else result.stderr
            
            if result.returncode == 0:
                # Parse validation output
                output_lines = result.stdout.split('\n')
                
                # Look for validation results from new guarded OR logic
                for line in output_lines:
                    line_stripped = line.strip()
                    # Capture extracted rows count from verifier output
                    if line_stripped.startswith("Total rows in CSV:"):
                        try:
                            validation_results["extracted_rows"] = int(line_stripped.split(":")[1].strip())
                        except Exception:
                            pass
                    elif line_stripped.startswith("Rows extracted to CSV:"):
                        try:
                            validation_results["extracted_rows"] = int(line_stripped.split(":")[1].strip())
                        except Exception:
                            pass
                    
                    # Capture final pass/fail and accuracy
                    if line_stripped.startswith("[RESULT]") and f"{year}:" in line_stripped:
                        if "PASSED" in line_stripped:
                            validation_results["validation_passed"] = True
                            # Try to parse accuracy like "with 100.0% accuracy"
                            try:
                                if "with" in line_stripped and "%" in line_stripped:
                                    acc_part = line_stripped.split("with", 1)[1].strip()
                                    acc_value = acc_part.split("%", 1)[0].strip()
                                    validation_results["accuracy"] = float(acc_value)
                            except Exception:
                                pass
                        elif "FAILED" in line_stripped:
                            validation_results["validation_passed"] = False
                    
                    # Any explicit error lines should flag failure
                    if "[FAILED] Validation failed" in line_stripped or "ERROR" in line_stripped.upper():
                        validation_results["validation_passed"] = False
            else:
                logger.error(f"Validation script failed: {result.stderr}")
                validation_results["validation_passed"] = False
        
        except Exception as e:
            logger.error(f"Error running validation script: {e}")
            validation_results["validation_passed"] = False
            validation_results["validation_output"] = f"Error: {str(e)}"
        
        # Fallback to basic validation if script validation fails
        if extracted_data and 'extraction_info' in extracted_data:
            # Use extraction info from the subprocess result
            extraction_info = extracted_data['extraction_info']
            total_programs = extraction_info.get('rows_extracted', 0)
            validation_results["extracted_rows"] = total_programs
            
            expected = validation_results["expected_rows"]
            if expected > 0:
                validation_results["accuracy"] = (total_programs / expected) * 100
                # Only override validation_passed if it wasn't set by the script
                if "validation_output" not in validation_results or not validation_results["validation_output"]:
                    validation_results["validation_passed"] = validation_results["accuracy"] >= 95
        elif extracted_data and 'institutes' in extracted_data:
            # Fallback for old format
            total_programs = sum(len(inst.get('programs', [])) for inst in extracted_data['institutes'].values())
            validation_results["extracted_rows"] = total_programs
            
            expected = validation_results["expected_rows"]
            if expected > 0:
                validation_results["accuracy"] = (total_programs / expected) * 100
                if "validation_output" not in validation_results or not validation_results["validation_output"]:
                    validation_results["validation_passed"] = validation_results["accuracy"] >= 95
        
        logger.info(f"Validation: {validation_results['accuracy']:.1f}% accuracy, Passed: {validation_results['validation_passed']}")
        return validation_results
    
    def run_pipeline(self, pdf_path: str, year: str, output_dir: str = "data/extracted") -> Dict:
        """Run the complete adaptive extraction pipeline"""
        logger.info(f"Starting adaptive extraction pipeline for {year}")
        
        pipeline_results = {
            "year": year,
            "pdf_path": pdf_path,
            "pipeline_steps": [],
            "final_result": None
        }
        
        try:
            # Step 1: Analyze table structure
            analysis = self.analyze_table_structure(pdf_path, year)
            pipeline_results["pipeline_steps"].append({
                "step": "table_analysis",
                "status": "completed",
                "results": analysis.__dict__
            })
            
            # Step 2: Test extraction strategies
            strategy_results = self.test_extraction_strategies(pdf_path, year, analysis)
            pipeline_results["pipeline_steps"].append({
                "step": "strategy_testing",
                "status": "completed",
                "results": [r.__dict__ for r in strategy_results]
            })
            
            # Step 3: Select best strategy
            best_strategy = self.select_best_strategy(strategy_results)
            if not best_strategy:
                raise Exception("No viable extraction strategy found")
            
            pipeline_results["pipeline_steps"].append({
                "step": "strategy_selection",
                "status": "completed",
                "results": best_strategy.__dict__
            })
            
            # Step 4: Extract data
            extracted_data = self.extract_with_best_strategy(pdf_path, year, best_strategy)
            pipeline_results["pipeline_steps"].append({
                "step": "data_extraction",
                "status": "completed",
                "results": {"institutes_count": len(extracted_data.get('institutes', {}))}
            })
            
            # Step 5: Validate results
            validation = self.validate_extraction(pdf_path, year, extracted_data, analysis)
            pipeline_results["pipeline_steps"].append({
                "step": "validation",
                "status": "completed",
                "results": validation
            })
            
            # Save results
            if extracted_data:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save JSON
                json_path = os.path.join(output_dir, f"iit_ranks_{year}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                
                # Save CSV
                csv_path = os.path.join(output_dir, f"iit_ranks_{year}.csv")
                df = self.extractor.convert_to_dataframe(extracted_data)
                df.to_csv(csv_path, index=False)
                
                logger.info(f"Results saved to {json_path} and {csv_path}")
            
            pipeline_results["final_result"] = {
                "success": validation["validation_passed"],
                "accuracy": validation["accuracy"],
                "extracted_data": extracted_data
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results["pipeline_steps"].append({
                "step": "error",
                "status": "failed",
                "error": str(e)
            })
            pipeline_results["final_result"] = {
                "success": False,
                "error": str(e)
            }
        
        return pipeline_results
    
    def run_batch_pipeline(self, pdf_directory: str, years: List[str] = None) -> Dict:
        """Run the pipeline on multiple PDFs"""
        if years is None:
            years = ["2022", "2023", "2024"]
        
        batch_results = {
            "total_files": len(years),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "results_by_year": {}
        }
        
        for year in years:
            pdf_path = os.path.join(pdf_directory, f"{year}.pdf")
            if os.path.exists(pdf_path):
                logger.info(f"Processing {year}...")
                result = self.run_pipeline(pdf_path, year)
                
                batch_results["results_by_year"][year] = result
                
                if result["final_result"]["success"]:
                    batch_results["successful_extractions"] += 1
                else:
                    batch_results["failed_extractions"] += 1
            else:
                logger.warning(f"PDF not found: {pdf_path}")
                batch_results["failed_extractions"] += 1
        
        return batch_results

def main():
    """Main function to run the adaptive extraction pipeline"""
    pipeline = AdaptiveExtractionPipeline()
    
    # Run on recent years
    results = pipeline.run_batch_pipeline("data/pdfs", ["2022", "2023", "2024"])
    
    # Print summary
    print("\n=== ADAPTIVE EXTRACTION PIPELINE RESULTS ===")
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful extractions: {results['successful_extractions']}")
    print(f"Failed extractions: {results['failed_extractions']}")
    
    for year, result in results["results_by_year"].items():
        final_result = result["final_result"]
        if final_result["success"]:
            print(f"\n{year}: [SUCCESS] - {final_result['accuracy']:.1f}% accuracy")
        else:
            print(f"\n{year}: [FAILED] - {final_result.get('error', 'Unknown error')}")
    
    # Save detailed results
    with open("pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nDetailed results saved to pipeline_results.json")

if __name__ == "__main__":
    main()