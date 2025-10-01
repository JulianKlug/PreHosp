"""
Join Automated and Manual Medical Registry Extractions

This script joins automated extraction results (94.3% success rate) with manual completion 
data for failed cases to achieve near-complete data coverage for medical research.

Usage:
    python join_extractions.py automated_results.xlsx manual_completions.xlsx
    python join_extractions.py automated_results.xlsx manual_completions.xlsx --output-dir /path/to/output
    python join_extractions.py automated_results.xlsx manual_completions.xlsx --verbose

The script performs intelligent matching using multiple strategies:
- Exact name matching
- Normalized matching (handles umlauts and accented characters)
- Partial matching for complex names
- Full name matching when available

Output includes:
- Complete merged dataset (Excel + CSV)
- Summary statistics
- Remaining failures (if any)
- Manual completions tracking
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict
import logging
import argparse
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def join_automated_and_manual_extraction(automated_file_path: str, manual_file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Join automated extraction results with manual completion data for failed cases.
    
    The automated extraction achieved 94.3% success rate (214/227 doctors).
    This function fills in the remaining 13 failed cases with manual data.
    
    Args:
        automated_file_path: Path to the automated extraction results Excel file
        manual_file_path: Path to the manual completion Excel file
        
    Returns:
        Tuple of (final_dataframe, summary_statistics)
    """
    try:
        logger.info("Starting join of automated and manual extractions...")
        
        # Load automated extraction results
        logger.info(f"Loading automated extractions from: {automated_file_path}")
        auto_df = pd.read_excel(automated_file_path)
        logger.info(f"Loaded {len(auto_df)} automated extraction records")
        
        # Load manual completion data
        logger.info(f"Loading manual completions from: {manual_file_path}")
        manual_df = pd.read_excel(manual_file_path)
        logger.info(f"Loaded {len(manual_df)} manual completion records")
        
        # Validate required columns in automated data
        required_auto_cols = ['first_name', 'last_name', 'full_name', 'search_successful', 
                             'licence_date', 'year_of_birth', 'specialist_qualifications', 
                             'additional_qualifications']
        missing_auto_cols = [col for col in required_auto_cols if col not in auto_df.columns]
        if missing_auto_cols:
            raise ValueError(f"Missing required columns in automated data: {missing_auto_cols}")
        
        # Validate required columns in manual data
        required_manual_cols = ['first_name', 'last_name', 'licence_date', 'year_of_birth', 
                               'specialist_qualifications', 'additional_qualifications']
        missing_manual_cols = [col for col in required_manual_cols if col not in manual_df.columns]
        if missing_manual_cols:
            raise ValueError(f"Missing required columns in manual data: {missing_manual_cols}")
        
        # Create a copy of automated data to work with
        final_df = auto_df.copy()
        
        # Identify failed automated extractions
        failed_mask = ~final_df['search_successful']
        failed_df = final_df[failed_mask].copy()
        logger.info(f"Found {len(failed_df)} failed automated extractions")
        
        # Track update statistics
        stats = {
            'total_records': len(final_df),
            'automated_successful': len(final_df[final_df['search_successful']]),
            'automated_failed': len(failed_df),
            'manual_records_available': len(manual_df),
            'successful_merges': 0,
            'unmatched_failures': 0,
            'duplicate_manual_entries': 0,
            'final_success_rate': 0.0
        }
        
        # Process each failed automated extraction
        updated_count = 0
        unmatched_failures = []
        
        for idx, failed_row in failed_df.iterrows():
            first_name = str(failed_row['first_name']).strip()
            last_name = str(failed_row['last_name']).strip()
            full_name = str(failed_row['full_name']).strip()
            
            logger.debug(f"Looking for manual data for: {first_name} {last_name}")
            
            # Try multiple matching strategies
            manual_matches = find_manual_match(manual_df, first_name, last_name, full_name)
            
            if len(manual_matches) == 1:
                # Single match found - update the record
                manual_row = manual_matches.iloc[0]
                
                # Update the failed record with manual data
                final_df.loc[idx, 'licence_date'] = manual_row['licence_date']
                final_df.loc[idx, 'year_of_birth'] = manual_row['year_of_birth']
                final_df.loc[idx, 'specialist_qualifications'] = manual_row['specialist_qualifications']
                final_df.loc[idx, 'additional_qualifications'] = manual_row['additional_qualifications']
                final_df.loc[idx, 'search_successful'] = True
                final_df.loc[idx, 'error_message'] = 'Completed manually'
                final_df.loc[idx, 'manual_completion'] = True
                
                updated_count += 1
                logger.info(f"✓ Updated {first_name} {last_name} with manual data")
                
            elif len(manual_matches) > 1:
                stats['duplicate_manual_entries'] += 1
                logger.warning(f"⚠ Multiple manual matches for {first_name} {last_name}, skipping")
                unmatched_failures.append(f"{first_name} {last_name} (multiple matches)")
                
            else:
                logger.warning(f"✗ No manual data found for {first_name} {last_name}")
                unmatched_failures.append(f"{first_name} {last_name} (no manual data)")
        
        # Update statistics
        stats['successful_merges'] = updated_count
        stats['unmatched_failures'] = len(unmatched_failures)
        stats['final_successful'] = len(final_df[final_df['search_successful']])
        stats['final_success_rate'] = (stats['final_successful'] / stats['total_records']) * 100
        
        # Add metadata columns
        final_df['extraction_completion_date'] = datetime.now().strftime('%Y-%m-%d')
        final_df['manual_completion'] = final_df.get('manual_completion', False)
        
        # Generate summary report
        logger.info("=== JOIN EXTRACTION SUMMARY ===")
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Automated successful: {stats['automated_successful']}")
        logger.info(f"Automated failed: {stats['automated_failed']}")
        logger.info(f"Manual records available: {stats['manual_records_available']}")
        logger.info(f"Successful merges: {stats['successful_merges']}")
        logger.info(f"Unmatched failures: {stats['unmatched_failures']}")
        logger.info(f"Final success rate: {stats['final_success_rate']:.1f}%")
        
        if unmatched_failures:
            logger.info(f"Unmatched failures: {unmatched_failures}")
        
        # Save results
        output_path = save_joined_results(final_df, stats, automated_file_path)
        logger.info(f"Joined results saved to: {output_path}")
        
        return final_df, stats
        
    except Exception as e:
        logger.error(f"Error joining extractions: {e}")
        raise


def find_manual_match(manual_df: pd.DataFrame, first_name: str, last_name: str, full_name: str) -> pd.DataFrame:
    """
    Find matching manual completion record using multiple strategies.
    
    Args:
        manual_df: DataFrame with manual completion data
        first_name: First name to match
        last_name: Last name to match  
        full_name: Full name to match
        
    Returns:
        DataFrame with matching rows (0, 1, or multiple matches)
    """
    # Strategy 1: Exact first + last name match
    mask1 = ((manual_df['first_name'].astype(str).str.strip().str.lower() == first_name.lower()) & 
             (manual_df['last_name'].astype(str).str.strip().str.lower() == last_name.lower()))
    
    matches = manual_df[mask1]
    if len(matches) > 0:
        return matches
    
    # Strategy 2: Full name match
    if 'full_name' in manual_df.columns:
        mask2 = manual_df['full_name'].astype(str).str.strip().str.lower() == full_name.lower()
        matches = manual_df[mask2]
        if len(matches) > 0:
            return matches
    
    # Strategy 3: Partial matching (last name + first name contains)
    mask3 = ((manual_df['last_name'].astype(str).str.strip().str.lower() == last_name.lower()) & 
             (manual_df['first_name'].astype(str).str.strip().str.lower().str.contains(first_name.lower())))
    
    matches = manual_df[mask3]
    if len(matches) > 0:
        return matches
    
    # Strategy 4: Handle names with special characters (normalize)
    first_normalized = normalize_name_for_matching(first_name)
    last_normalized = normalize_name_for_matching(last_name)
    
    manual_first_normalized = manual_df['first_name'].astype(str).apply(normalize_name_for_matching)
    manual_last_normalized = manual_df['last_name'].astype(str).apply(normalize_name_for_matching)
    
    mask4 = ((manual_first_normalized == first_normalized) & 
             (manual_last_normalized == last_normalized))
    
    matches = manual_df[mask4]
    return matches


def normalize_name_for_matching(name: str) -> str:
    """
    Normalize names for better matching by handling special characters and formatting.
    
    Args:
        name: Name to normalize
        
    Returns:
        Normalized name
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    name = name.lower().strip()
    
    # Handle umlauts and special characters
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    
    for original, replacement in replacements.items():
        name = name.replace(original, replacement)
    
    # Remove extra spaces and special characters
    name = ' '.join(name.split())
    
    return name


def save_joined_results(final_df: pd.DataFrame, stats: Dict, automated_file_path: str) -> str:
    """
    Save the joined results to Excel and CSV files.
    
    Args:
        final_df: Final joined DataFrame
        stats: Summary statistics
        automated_file_path: Original automated file path for naming
        
    Returns:
        Path to the saved Excel file
    """
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = Path(automated_file_path)
    output_dir = input_path.parent
    
    # Excel file
    excel_output = output_dir / f"final_complete_extractions_{timestamp}.xlsx"
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        # Main data
        final_df.to_excel(writer, sheet_name='Complete_Extractions', index=False)
        
        # Statistics summary
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value']
        stats_df.to_excel(writer, sheet_name='Summary_Statistics')
        
        # Failed extractions (if any remain)
        failed_final = final_df[~final_df['search_successful'].fillna(False)]
        if len(failed_final) > 0:
            failed_final.to_excel(writer, sheet_name='Remaining_Failures', index=False)
        
        # Manual completions
        manual_mask = final_df.get('manual_completion', pd.Series([False] * len(final_df), index=final_df.index)).fillna(False)
        manual_completed = final_df[manual_mask]
        if len(manual_completed) > 0:
            manual_completed.to_excel(writer, sheet_name='Manual_Completions', index=False)
    
    # CSV file for easy analysis
    csv_output = output_dir / f"final_complete_extractions_{timestamp}.csv"
    final_df.to_csv(csv_output, index=False, encoding='utf-8')
    
    logger.info("Results saved to:")
    logger.info(f"  Excel: {excel_output}")
    logger.info(f"  CSV: {csv_output}")
    
    return str(excel_output)


def parse_arguments():
    """
    Parse command-line arguments for the join extraction script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Join automated extraction results with manual completion data for failed cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Join automated and manual extractions
  python join_extractions.py automated_results.xlsx manual_completions.xlsx
  
  # Specify custom output directory
  python join_extractions.py automated_results.xlsx manual_completions.xlsx --output-dir /path/to/output
  
  # Enable verbose logging
  python join_extractions.py automated_results.xlsx manual_completions.xlsx --verbose
        """
    )
    
    parser.add_argument(
        'automated_file',
        type=str,
        help='Path to the automated extraction results Excel file'
    )
    
    parser.add_argument(
        'manual_file',
        type=str,
        help='Path to the manual completion Excel file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for joined results (default: same as automated file directory)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """
    Main function with command-line interface for joining extractions.
    """
    args = parse_arguments()
    
    # Configure logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate input files exist
    automated_path = Path(args.automated_file)
    manual_path = Path(args.manual_file)
    
    if not automated_path.exists():
        logger.error(f"Automated extraction file not found: {automated_path}")
        return 1
    
    if not manual_path.exists():
        logger.error(f"Manual completion file not found: {manual_path}")
        return 1
    
    try:
        logger.info("Starting join extraction process...")
        logger.info(f"Automated file: {automated_path}")
        logger.info(f"Manual file: {manual_path}")
        
        if args.output_dir:
            # Use custom output directory
            final_df, stats = join_automated_and_manual_extraction(
                str(automated_path), 
                str(manual_path)
            )
            # Save to custom directory
            save_joined_results_to_dir(final_df, stats, args.output_dir)
        else:
            # Use default output directory (same as automated file)
            final_df, stats = join_automated_and_manual_extraction(
                str(automated_path), 
                str(manual_path)
            )
        
        logger.info("\n=== FINAL EXTRACTION RESULTS ===")
        logger.info(f"Total doctors processed: {stats['total_records']}")
        logger.info(f"Final successful extractions: {stats['final_successful']}")
        logger.info(f"Final success rate: {stats['final_success_rate']:.1f}%")
        logger.info(f"Manual completions added: {stats['successful_merges']}")
        logger.info(f"Still missing data: {stats['unmatched_failures']}")
        
        if stats['unmatched_failures'] > 0:
            logger.warning(f"Note: {stats['unmatched_failures']} records still lack complete data")
            return 2  # Partial success
        
        logger.info("✅ Join operation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Join operation failed: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1


def save_joined_results_to_dir(final_df: pd.DataFrame, stats: Dict, output_dir: str) -> str:
    """
    Save the joined results to a specified directory.
    
    Args:
        final_df: Final joined DataFrame
        stats: Summary statistics
        output_dir: Output directory path
        
    Returns:
        Path to the saved Excel file
    """
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Excel file
    excel_output = output_path / f"final_complete_extractions_{timestamp}.xlsx"
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        # Main data
        final_df.to_excel(writer, sheet_name='Complete_Extractions', index=False)
        
        # Statistics summary
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value']
        stats_df.to_excel(writer, sheet_name='Summary_Statistics')
        
        # Failed extractions (if any remain)
        failed_final = final_df[~final_df['search_successful'].fillna(False)]
        if len(failed_final) > 0:
            failed_final.to_excel(writer, sheet_name='Remaining_Failures', index=False)
        
        # Manual completions
        manual_mask = final_df.get('manual_completion', pd.Series([False] * len(final_df), index=final_df.index)).fillna(False)
        manual_completed = final_df[manual_mask]
        if len(manual_completed) > 0:
            manual_completed.to_excel(writer, sheet_name='Manual_Completions', index=False)
    
    # CSV file for easy analysis
    csv_output = output_path / f"final_complete_extractions_{timestamp}.csv"
    final_df.to_csv(csv_output, index=False, encoding='utf-8')
    
    logger.info("Results saved to:")
    logger.info(f"  Excel: {excel_output}")
    logger.info(f"  CSV: {csv_output}")
    
    return str(excel_output)


if __name__ == "__main__":
    exit(main())