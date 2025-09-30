#!/usr/bin/env python3
"""
Complete Medical Registry Data Extraction Script
This script processes the entire doctor list and extracts medical registry data

Usage:
    python run_extraction.py <doctor_list_path> <output_dir> [--max-doctors N] [--start-from N] [--headless]
"""

import pandas as pd
import json
import time
import logging
import os
import argparse
from datetime import datetime
from typing import List, Dict
from selenium_scraper import SeleniumMedRegScraper, parse_doctor_names

class MedRegDataExtractor:
    """
    Complete data extraction pipeline for medical registry data
    """
    
    def __init__(self, doctor_list_path: str, output_dir: str = "output", headless: bool = True):
        """
        Initialize the data extractor
        
        Args:
            doctor_list_path: Path to the Excel file with doctor list
            output_dir: Directory to save output files
            headless: Whether to run browser in headless mode
        """
        self.doctor_list_path = doctor_list_path
        self.output_dir = output_dir
        self.headless = headless
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging to output directory
        self._setup_logging()
        
        # Initialize scraper
        self.scraper = SeleniumMedRegScraper(headless=headless, delay_between_requests=3.0)
        
        self.logger.info(f"Initialized MedRegDataExtractor with output dir: {output_dir}")
    
    def _setup_logging(self):
        """Setup logging to save logs in the output directory"""
        log_filename = f'medreg_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        log_path = os.path.join(self.output_dir, log_filename)
        
        # Create logger
        self.logger = logging.getLogger(f'{__name__}_{id(self)}')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler - logs to output directory
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_path}")
    
    def extract_all_doctors(self, max_doctors: int = None, start_from: int = 0) -> List[Dict]:
        """
        Extract data for all doctors in the list
        
        Args:
            max_doctors: Maximum number of doctors to process (None for all)
            start_from: Index to start from (for resuming interrupted runs)
            
        Returns:
            List of dictionaries with doctor data
        """
        try:
            # Parse doctor names
            doctors = parse_doctor_names(self.doctor_list_path)
            
            if not doctors:
                self.logger.error("No doctors found in the list")
                return []
            
            # Apply limits
            if start_from > 0:
                doctors = doctors[start_from:]
                self.logger.info(f"Starting from doctor #{start_from}")
            
            if max_doctors:
                doctors = doctors[:max_doctors]
                self.logger.info(f"Processing {len(doctors)} doctors (max: {max_doctors})")
            else:
                self.logger.info(f"Processing all {len(doctors)} doctors")
            
            # Start browser
            self.scraper.start_driver()
            
            # Extract data
            results = []
            for i, (first_name, last_name) in enumerate(doctors):
                try:
                    self.logger.info(f"Processing {i+1}/{len(doctors)}: {first_name} {last_name}")
                    
                    # Extract data
                    doctor_data = self.scraper.search_doctor(first_name, last_name)
                    
                    # Add processing metadata
                    doctor_data['processing_index'] = start_from + i
                    doctor_data['processing_timestamp'] = datetime.now().isoformat()
                    
                    results.append(doctor_data)
                    
                    # Save interim results every 10 doctors
                    if (i + 1) % 10 == 0:
                        self._save_interim_results(results, i + 1)
                    
                    # Progress logging
                    if doctor_data['search_successful']:
                        self.logger.info(f"✓ Successfully extracted data for {first_name} {last_name}")
                    else:
                        self.logger.warning(f"✗ Failed to extract data for {first_name} {last_name}: {doctor_data.get('error_message', 'Unknown error')}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {first_name} {last_name}: {e}")
                    results.append({
                        'first_name': first_name,
                        'last_name': last_name,
                        'full_name': f"{first_name} {last_name}",
                        'licence_date': None,
                        'year_of_birth': None,
                        'specialist_qualifications': [],
                        'additional_qualifications': [],
                        'search_successful': False,
                        'multiple_matches': False,
                        'error_message': str(e),
                        'processing_index': start_from + i,
                        'processing_timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in extract_all_doctors: {e}")
            raise
        finally:
            # Always close the browser
            self.scraper.close_driver()
    
    def _save_interim_results(self, results: List[Dict], count: int):
        """Save interim results during processing"""
        try:
            interim_file = os.path.join(self.output_dir, f'interim_results_{count}_doctors.json')
            with open(interim_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved interim results to {interim_file}")
        except Exception as e:
            self.logger.error(f"Failed to save interim results: {e}")
    
    def save_results(self, results: List[Dict], format: str = 'both') -> Dict[str, str]:
        """
        Save extraction results to files
        
        Args:
            results: List of doctor data dictionaries
            format: 'csv', 'excel', 'json', or 'both' (saves in multiple formats)
            
        Returns:
            Dictionary with file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_paths = {}
        
        try:
            # Save as JSON
            if format in ['json', 'both']:
                json_file = os.path.join(self.output_dir, f'medreg_data_{timestamp}.json')
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                file_paths['json'] = json_file
                self.logger.info(f"Saved JSON data to {json_file}")
            
            # Convert to DataFrame for CSV/Excel
            if format in ['csv', 'excel', 'both']:
                df = self._results_to_dataframe(results)
                
                # Save as CSV
                if format in ['csv', 'both']:
                    csv_file = os.path.join(self.output_dir, f'medreg_data_{timestamp}.csv')
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    file_paths['csv'] = csv_file
                    self.logger.info(f"Saved CSV data to {csv_file}")
                
                # Save as Excel
                if format in ['excel', 'both']:
                    excel_file = os.path.join(self.output_dir, f'medreg_data_{timestamp}.xlsx')
                    df.to_excel(excel_file, index=False)
                    file_paths['excel'] = excel_file
                    self.logger.info(f"Saved Excel data to {excel_file}")
            
            return file_paths
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        # Flatten the data for tabular format
        flattened_results = []
        
        for result in results:
            flattened = {
                'first_name': result.get('first_name', ''),
                'last_name': result.get('last_name', ''),
                'full_name': result.get('full_name', ''),
                'licence_date': result.get('licence_date', ''),
                'year_of_birth': result.get('year_of_birth', ''),
                'specialist_qualifications': '; '.join(result.get('specialist_qualifications', [])),
                'additional_qualifications': '; '.join(result.get('additional_qualifications', [])),
                'search_successful': result.get('search_successful', False),
                'multiple_matches': result.get('multiple_matches', False),
                'error_message': result.get('error_message', ''),
                'processing_index': result.get('processing_index', ''),
                'processing_timestamp': result.get('processing_timestamp', ''),
                'selected_result_text': result.get('selected_result_text', '')[:500] if result.get('selected_result_text') else ''  # Truncate for CSV
            }
            flattened_results.append(flattened)
        
        return pd.DataFrame(flattened_results)
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """Generate a summary report of the extraction"""
        total_doctors = len(results)
        successful_extractions = sum(1 for r in results if r.get('search_successful', False))
        failed_extractions = total_doctors - successful_extractions
        multiple_matches = sum(1 for r in results if r.get('multiple_matches', False))
        
        # Count doctors with data
        doctors_with_licence_date = sum(1 for r in results if r.get('licence_date'))
        doctors_with_year_of_birth = sum(1 for r in results if r.get('year_of_birth'))
        doctors_with_specialist_quals = sum(1 for r in results if r.get('specialist_qualifications'))
        doctors_with_additional_quals = sum(1 for r in results if r.get('additional_qualifications'))
        
        summary = {
            'total_doctors_processed': total_doctors,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': f"{(successful_extractions / total_doctors * 100):.1f}%" if total_doctors > 0 else "0%",
            'multiple_matches_found': multiple_matches,
            'doctors_with_licence_date': doctors_with_licence_date,
            'doctors_with_year_of_birth': doctors_with_year_of_birth,
            'doctors_with_specialist_qualifications': doctors_with_specialist_quals,
            'doctors_with_additional_qualifications': doctors_with_additional_quals,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def run_extraction(self, max_doctors: int = None, start_from: int = 0) -> str:
        """
        Run the complete extraction pipeline
        
        Args:
            max_doctors: Maximum number of doctors to process
            start_from: Index to start from
            
        Returns:
            Summary message
        """
        try:
            self.logger.info("=== Starting Medical Registry Data Extraction ===")
            
            # Extract data
            results = self.extract_all_doctors(max_doctors=max_doctors, start_from=start_from)
            
            if not results:
                return "No data extracted"
            
            # Save results
            file_paths = self.save_results(results, format='both')
            
            # Generate and save summary
            summary = self.generate_summary_report(results)
            summary_file = os.path.join(self.output_dir, f'extraction_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Log summary
            self.logger.info("=== Extraction Summary ===")
            for key, value in summary.items():
                self.logger.info(f"{key}: {value}")
            
            self.logger.info("=== Files Created ===")
            for format_type, path in file_paths.items():
                self.logger.info(f"{format_type.upper()}: {path}")
            self.logger.info(f"SUMMARY: {summary_file}")
            
            return f"Extraction completed. Processed {summary['total_doctors_processed']} doctors with {summary['success_rate']} success rate."
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Extract medical registry data for a list of doctors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_extraction.py doctors.xlsx output_folder
    python run_extraction.py doctors.xlsx output_folder --max-doctors 10
    python run_extraction.py doctors.xlsx output_folder --start-from 50 --headless
        """
    )
    
    parser.add_argument('doctor_list_path', 
                       help='Path to the Excel file containing the doctor list')
    parser.add_argument('output_dir', 
                       help='Directory to save output files and logs')
    parser.add_argument('--max-doctors', type=int, default=None,
                       help='Maximum number of doctors to process (default: all)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Index to start from (default: 0)')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode (default: False for debugging)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.doctor_list_path):
        print(f"Error: Doctor list file '{args.doctor_list_path}' not found.")
        return 1
    
    # Create extractor
    extractor = MedRegDataExtractor(
        doctor_list_path=args.doctor_list_path,
        output_dir=args.output_dir,
        headless=args.headless
    )
    
    # Run extraction
    try:
        result_message = extractor.run_extraction(
            max_doctors=args.max_doctors, 
            start_from=args.start_from
        )
        print(f"\n{result_message}")
        return 0
    except Exception as e:
        print(f"Extraction failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())