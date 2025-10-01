# Swiss Medical Registry Data Extraction

A robust Selenium-based scraper for extracting physician data from the Swiss Medical Registry website (https://www.medregom.admin.ch/medreg/search).

## Overview

This tool automates the extraction of physician licensing and qualification data from the Swiss Medical Registry for research purposes. It handles complex JavaScript interactions, form state management, and provides comprehensive data extraction with intelligent doctor matching and emergency medicine qualification detection.

## Features

- **Automated Data Extraction**: Extracts comprehensive physician data including licensing dates, specialist qualifications, and additional certifications
- **Intelligent Doctor Matching**: Advanced scoring algorithm to select the correct physician from multiple search results

## Data Extracted

For each physician in the input list, the tool extracts:

- **License Date**: Date of issue of physician license
- **Year of Birth**: Birth year of the physician
- **Specialist Qualifications**: All postgraduate specialist qualifications (e.g., Anaesthesiology, Intensive Care Medicine)
- **Additional Qualifications**: Additional certifications and training (e.g., prehospital emergency medicine)
- **Search Metadata**: Multiple matches flag, selected result text, search success status

## Installation

### Prerequisites

- Python 3.8 or higher
- Chrome browser installed
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository and navigate to the directory
cd /path/to/PreHosp/analgesia/extract_medreg_data

# Install required dependencies
pip install selenium pandas openpyxl webdriver-manager

# Verify installation
python selenium_scraper.py --help
```

## Usage

### Basic Usage

```bash
# Extract data for all doctors in the list
python run_extraction.py "/path/to/doctor_list.xlsx" "output_directory"

# Extract data for first 10 doctors only
python run_extraction.py "/path/to/doctor_list.xlsx" "output_directory" --max-doctors 10

# Run in visible browser mode (for debugging)
python run_extraction.py "/path/to/doctor_list.xlsx" "output_directory" --no-headless

# Resume extraction starting from doctor 50
python run_extraction.py "/path/to/doctor_list.xlsx" "output_directory" --start-from 50
```

### Command Line Options

```
python run_extraction.py <doctor_list_path> <output_dir> [options]

Arguments:
  doctor_list_path    Path to Excel file containing doctor names
  output_dir         Directory to save extracted data

Options:
  --max-doctors N     Extract data for first N doctors only
  --start-from N      Start extraction from doctor N (for resuming)
  --headless         Run browser in headless mode (default: True)
  --no-headless      Run browser in visible mode (for debugging)
  --delay SECONDS    Delay between requests in seconds (default: 1.0)
```

### Input File Format

The input Excel file should contain a column named `'Mitglieder mit Einsatzfunktion'` with doctor names in the format:
```
"First Last (Role)"
```

Example:
```
Joao Bomao (Notarzt)
Giorgina Giorgina (Notärztin)
Alessandro Gemmelli (Notarzt)
```

### Output Files

The tool generates several output files in the specified directory:

- **`extracted_data_YYYYMMDD_HHMMSS.csv`**: Main data in CSV format
- **`extracted_data_YYYYMMDD_HHMMSS.xlsx`**: Main data in Excel format  
- **`extracted_data_YYYYMMDD_HHMMSS.json`**: Detailed data in JSON format
- **`extraction_summary_YYYYMMDD_HHMMSS.txt`**: Extraction summary and statistics
- **`medreg_extraction_YYYYMMDD_HHMMSS.log`**: Detailed extraction logs

### Sample Output

```csv
first_name,last_name,licence_date,year_of_birth,specialist_qualifications,additional_qualifications,emergency_medicine_qualified
Joao,Bomao,23.11.2012,1995,"['Intensive care medicine', 'Anaesthesiology']","['prehospital emergency medicine (emergency physician) SSERM']",True
Giorgina,Giorgina,15.02.2015,1978,"['Anaesthesiology']","['prehospital emergency medicine (emergency physician) SSERM']",True
```

## Technical Details

### Architecture

- **`selenium_scraper.py`**: Core Selenium automation engine with intelligent doctor selection
- **`run_extraction.py`**: Production CLI interface for batch processing
- **Browser Automation**: Chrome WebDriver with optimized performance settings
- **Session Management**: Aggressive session clearing (cookies, localStorage, sessionStorage) between searches

### Performance Optimizations

- **Dynamic Waits**: WebDriverWait with expected conditions instead of fixed delays
- **Reduced Timeouts**: 15-second maximum wait times for page elements
- **Chrome Optimization**: Disabled image loading, GPU acceleration for faster page loads
- **Intelligent Delays**: 1.0-second delays between requests (configurable)

### Reliability Features

- **Profession Filtering**: Automatic selection of "Physician" filter to exclude non-physicians
- **Retry Logic**: 3-attempt retry for dropdown selections with verification
- **Form State Clearing**: Page refresh and storage clearing between searches
- **Multiple Match Handling**: Scoring algorithm prioritizes emergency medicine specialists
- **Error Handling**: Comprehensive error handling with detailed logging

## Testing

### Quick Tests

```bash
# Test individual doctor extraction
python -c "
from selenium_scraper import SeleniumMedRegScraper
scraper = SeleniumMedRegScraper(headless=True)
scraper.start_driver()
result = scraper.search_doctor('Julian', 'Klug')
print(result)
scraper.close_driver()
"

# Test batch processing with sample doctors
python selenium_scraper.py
```

## Troubleshooting

### Common Issues

1. **Chrome Driver Issues**: The tool automatically downloads and manages Chrome drivers
2. **Timeout Errors**: Increase delays with `--delay` option for slower connections
3. **No Results Found**: Some doctors may not exist in the registry or have name variations
4. **Session Issues**: The tool automatically clears browser state between searches

### Debug Mode

Run in visible browser mode to diagnose issues:
```bash
python run_extraction.py "/path/to/doctors.xlsx" "output" --no-headless
```

### Logs

Check the generated log files for detailed execution information:
```bash
tail -f output/medreg_extraction_*.log
```

## Performance

- **Processing Speed**: ~15-20 seconds per doctor (including page loads and data extraction)
- **Memory Usage**: Minimal - browser session is managed efficiently

## License & Ethics

This tool is designed for legitimate research purposes. Users must:
- Respect the Swiss Medical Registry's terms of use
- Use data responsibly and in compliance with privacy regulations
- Avoid excessive request rates that could impact the service
- Ensure appropriate institutional permissions for data collection

## Contributing

This tool is part of a research project. For issues or improvements, please check the project documentation or contact the research team.
