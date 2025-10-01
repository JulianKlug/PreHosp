#!/usr/bin/env python3
"""
Retry Failed Medical Registry Extractions
This script automatically retries failed extractions from previous runs
"""

import pandas as pd
import os
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from selenium_scraper import SeleniumMedRegScraper, normalize_text_for_matching

def parse_complex_name(full_name: str) -> List[Tuple[str, str]]:
    """
    Parse complex names and return multiple possible name combinations
    
    Args:
        full_name: Full name to parse like "Ana Patricia Borda de Agua Reis"
        
    Returns:
        List of (first_name, last_name) tuples to try
    """
    if not full_name:
        return []
    
    parts = full_name.strip().split()
    if len(parts) < 2:
        return [(full_name, "")]
    
    combinations = []
    
    # For names like "Ana Patricia Borda de Agua Reis"
    # Try different splits: first 1, 2, or 3 words as first name
    for split_point in range(1, min(4, len(parts))):
        first_name = " ".join(parts[:split_point])
        last_name = " ".join(parts[split_point:])
        combinations.append((first_name, last_name))
    
    return combinations

def find_latest_extraction_files(folder: str) -> Dict[str, str]:
    """
    Find the latest extraction files in a folder
    
    Args:
        folder: Path to folder containing extraction files
        
    Returns:
        Dictionary with file paths
    """
    # Look for common extraction file patterns
    patterns = [
        os.path.join(folder, "extracted_data_*.csv"),
        os.path.join(folder, "medreg_data_*.csv"),
        os.path.join(folder, "merged_results_*.csv")
    ]
    
    latest_files = {}
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            # Get the most recent file
            latest_file = max(files, key=os.path.getmtime)
            latest_files[pattern] = latest_file
    
    return latest_files

def load_extraction_results(file_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Load extraction results from the most recent file
    
    Args:
        file_paths: Dictionary of file paths
        
    Returns:
        DataFrame with extraction results
    """
    if not file_paths:
        raise ValueError("No extraction files found")
    
    # Use the most recent file
    latest_file = max(file_paths.values(), key=os.path.getmtime)
    print(f"Loading extraction results from: {latest_file}")
    
    return pd.read_csv(latest_file)

def identify_failed_extractions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Identify failed extractions from the results DataFrame
    
    Args:
        df: DataFrame with extraction results
        
    Returns:
        List of (first_name, last_name) tuples for failed extractions
    """
    # Find rows where search was unsuccessful
    failed_mask = ~df['search_successful']
    failed_df = df[failed_mask]
    
    failed_doctors = []
    for _, row in failed_df.iterrows():
        first_name = str(row['first_name']).strip()
        last_name = str(row['last_name']).strip()
        
        # Skip if names are empty or NaN
        if first_name and last_name and first_name != 'nan' and last_name != 'nan':
            failed_doctors.append((first_name, last_name))
    
    print(f"Found {len(failed_doctors)} failed extractions:")
    for first, last in failed_doctors:
        print(f"  - {first} {last}")
    
    return failed_doctors

def retry_failed_extractions(failed_doctors: List[Tuple[str, str]], 
                           output_folder: str,
                           headless: bool = True,
                           delay: float = 1.0) -> List[Dict]:
    """
    Retry extraction for failed doctors
    
    Args:
        failed_doctors: List of (first_name, last_name) tuples
        output_folder: Folder to save retry results
        headless: Whether to run browser in headless mode
        delay: Delay between requests
        
    Returns:
        List of extraction result dictionaries
    """
    if not failed_doctors:
        return []
    
    print(f"Starting retry extraction for {len(failed_doctors)} doctors...")
    
    # Initialize scraper
    scraper = SeleniumMedRegScraper(headless=headless, delay_between_requests=delay)
    
    try:
        scraper.start_driver()
        
        retry_results = []
        for i, (original_first, original_last) in enumerate(failed_doctors):
            full_name = f"{original_first} {original_last}"
            print(f"Retrying {i+1}/{len(failed_doctors)}: {full_name}")
            
            # Generate possible name combinations for complex names
            name_combinations = parse_complex_name(full_name)
            
            # If we only got one combination, add the original as well
            if len(name_combinations) == 1:
                name_combinations.append((original_first, original_last))
            
            success = False
            best_result = None
            
            # Try each name combination
            for j, (first_name, last_name) in enumerate(name_combinations):
                try:
                    print(f"  Trying combination {j+1}/{len(name_combinations)}: '{first_name}' + '{last_name}'")
                    result = scraper.search_doctor(first_name, last_name)
                    result['retry_attempt'] = True
                    result['retry_timestamp'] = datetime.now().isoformat()
                    result['original_first_name'] = original_first
                    result['original_last_name'] = original_last
                    result['combination_tried'] = f"{first_name} | {last_name}"
                    
                    if result['search_successful']:
                        print(f"  ✓ Success with combination: '{first_name}' + '{last_name}'")
                        best_result = result
                        success = True
                        break  # Stop trying other combinations
                    else:
                        print(f"  ✗ Failed with combination: '{first_name}' + '{last_name}'")
                        if not best_result:  # Keep first result as fallback
                            best_result = result
                            
                except Exception as e:
                    print(f"  ✗ Error with combination '{first_name}' + '{last_name}': {e}")
                    error_result = {
                        'first_name': first_name,
                        'last_name': last_name,
                        'full_name': full_name,
                        'search_successful': False,
                        'error_message': str(e),
                        'retry_attempt': True,
                        'retry_timestamp': datetime.now().isoformat(),
                        'original_first_name': original_first,
                        'original_last_name': original_last,
                        'combination_tried': f"{first_name} | {last_name}"
                    }
                    if not best_result:
                        best_result = error_result
            
            # Add the best result we found
            if best_result:
                retry_results.append(best_result)
                if success:
                    print(f"  ✓ Final result: Success for {full_name}")
                else:
                    print(f"  ✗ Final result: Failed for {full_name}")
            else:
                # Fallback if no result at all
                retry_results.append({
                    'first_name': original_first,
                    'last_name': original_last,
                    'full_name': full_name,
                    'search_successful': False,
                    'error_message': 'No combinations worked',
                    'retry_attempt': True,
                    'retry_timestamp': datetime.now().isoformat(),
                    'original_first_name': original_first,
                    'original_last_name': original_last,
                    'combination_tried': 'all failed'
                })
        
        return retry_results
        
    finally:
        scraper.close_driver()

def save_retry_results(retry_results: List[Dict], output_folder: str):
    """
    Save retry results to files
    
    Args:
        retry_results: List of retry result dictionaries
        output_folder: Folder to save results
    """
    if not retry_results:
        print("No retry results to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to DataFrame
    df = pd.DataFrame(retry_results)
    
    # Save as CSV
    csv_file = os.path.join(output_folder, f"retry_results_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    # Save as Excel
    excel_file = os.path.join(output_folder, f"retry_results_{timestamp}.xlsx")
    df.to_excel(excel_file, index=False)
    
    print(f"Retry results saved to: {output_folder}")
    print(f"  CSV: {csv_file}")
    print(f"  Excel: {excel_file}")

def merge_results_with_original(original_df: pd.DataFrame, 
                              retry_results: List[Dict],
                              output_folder: str):
    """
    Merge successful retry results with original data
    """
    if not retry_results:
        print("No retry results to merge")
        return
    
    # Convert retry results to DataFrame
    retry_df = pd.DataFrame([{
        'first_name': r.get('first_name', ''),
        'last_name': r.get('last_name', ''),
        'full_name': r.get('full_name', ''),
        'licence_date': r.get('licence_date', ''),
        'year_of_birth': r.get('year_of_birth', ''),
        'specialist_qualifications': '; '.join(r.get('specialist_qualifications', [])),
        'additional_qualifications': '; '.join(r.get('additional_qualifications', [])),
        'search_successful': r.get('search_successful', False),
        'multiple_matches': r.get('multiple_matches', False),
        'error_message': r.get('error_message', ''),
        'retry_attempt': True
    } for r in retry_results if r.get('search_successful', False)])  # Only successful retries
    
    if retry_df.empty:
        print("No successful retry results to merge")
        return
    
    print(f"Merging {len(retry_df)} successful retry results with original data")
    
    # Create a copy of original data
    merged_df = original_df.copy()
    
    # Update original data with successful retries
    for _, retry_row in retry_df.iterrows():
        # Try multiple matching strategies
        original_first = retry_row.get('original_first_name', retry_row['first_name'])
        original_last = retry_row.get('original_last_name', retry_row['last_name'])
        
        # Strategy 1: Match using original names if available
        mask1 = ((merged_df['first_name'] == original_first) & 
                 (merged_df['last_name'] == original_last))
        
        # Strategy 2: Match using current retry names
        mask2 = ((merged_df['first_name'] == retry_row['first_name']) & 
                 (merged_df['last_name'] == retry_row['last_name']))
        
        # Strategy 3: Match using full name reconstruction
        original_full = f"{original_first} {original_last}"
        mask3 = merged_df['full_name'] == original_full
        
        # Use whichever mask finds a match
        mask = mask1 | mask2 | mask3
        
        if mask.any():
            print(f"  Updating: {original_full} -> successful extraction")
            # Update the row with retry data (excluding the original name fields)
            update_fields = ['licence_date', 'year_of_birth', 'specialist_qualifications', 
                           'additional_qualifications', 'search_successful', 'multiple_matches', 
                           'error_message', 'retry_attempt', 'selected_result_text']
            
            for field in update_fields:
                if field in retry_row.index and field in merged_df.columns:
                    merged_df.loc[mask, field] = retry_row[field]
        else:
            print(f"  Warning: Could not find original record for {original_full}")
    
    # Save merged results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    merged_csv = os.path.join(output_folder, f"merged_results_{timestamp}.csv")
    merged_df.to_csv(merged_csv, index=False)
    
    merged_excel = os.path.join(output_folder, f"merged_results_{timestamp}.xlsx")
    merged_df.to_excel(merged_excel, index=False)
    
    print("Merged results saved to:")
    print(f"  CSV: {merged_csv}")
    print(f"  Excel: {merged_excel}")
    
    # Print updated summary
    total = len(merged_df)
    successful = len(merged_df[merged_df['search_successful']])
    print(f"Updated success rate: {successful}/{total} ({successful/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Retry failed medical registry extractions')
    parser.add_argument('extraction_folder', help='Path to folder containing previous extraction results')
    parser.add_argument('--output-folder', help='Output folder for retry results (default: same as extraction_folder)')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    parser.add_argument('--no-headless', action='store_true', help='Run browser in visible mode')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds')
    parser.add_argument('--merge', action='store_true', help='Merge successful retries with original data')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be retried without actually doing it')
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically confirm retry without user prompt')
    parser.add_argument('--max-retries', type=int, help='Maximum number of failed extractions to retry (default: all)')
    
    args = parser.parse_args()
    
    # Handle headless flag
    headless = args.headless and not args.no_headless
    
    # Set output folder
    output_folder = args.output_folder or args.extraction_folder
    
    try:
        print(f"Looking for extraction files in: {args.extraction_folder}")
        
        # Find latest extraction files
        file_paths = find_latest_extraction_files(args.extraction_folder)
        
        if not file_paths:
            print("No extraction files found!")
            return 1
        
        # Load extraction results
        df = load_extraction_results(file_paths)
        print(f"Loaded {len(df)} total extraction records")
        
        # Identify failed extractions
        failed_doctors = identify_failed_extractions(df)
        
        # Limit the number of retries if specified
        if args.max_retries and len(failed_doctors) > args.max_retries:
            print(f"Limiting retry to first {args.max_retries} of {len(failed_doctors)} failed extractions")
            failed_doctors = failed_doctors[:args.max_retries]
        
        if not failed_doctors:
            print("No failed extractions found! All extractions were successful.")
            return 0
        
        if args.dry_run:
            print(f"DRY RUN: Would retry {len(failed_doctors)} failed extractions")
            print("Use --merge flag to merge successful retries with original data")
            return 0
        
        # Ask for confirmation unless --yes flag is provided
        if not args.yes:
            response = input(f"Retry extraction for {len(failed_doctors)} doctors? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Retry cancelled")
                return 0
        else:
            print(f"Auto-confirming retry for {len(failed_doctors)} doctors (--yes flag provided)")
        
        # Retry failed extractions
        retry_results = retry_failed_extractions(
            failed_doctors, 
            output_folder,
            headless=headless,
            delay=args.delay
        )
        
        # Save retry results
        save_retry_results(retry_results, output_folder)
        
        # Merge results if requested
        if args.merge:
            merge_results_with_original(df, retry_results, output_folder)
        
        # Print summary
        successful_retries = sum(1 for r in retry_results if r.get('search_successful', False))
        print(f"\nRetry completed: {successful_retries}/{len(retry_results)} successful")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
