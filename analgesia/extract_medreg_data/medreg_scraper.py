#!/usr/bin/env python3
"""
Medical Registry Scraper for Swiss doctors
Scrapes data from https://www.medregom.admin.ch/medreg/search

This scraper extracts:
- Date of issue of physician licence
- All (Postgraduate) specialist qualifications 
- All Additional qualifications
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import re
from typing import Dict, List, Optional, Tuple
import json
from urllib.parse import urljoin
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medreg_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedRegScraper:
    """
    Scraper for the Swiss Medical Registry website
    """
    
    def __init__(self, delay_between_requests: float = 2.0):
        """
        Initialize the scraper
        
        Args:
            delay_between_requests: Delay in seconds between requests to be respectful
        """
        self.base_url = "https://www.medregom.admin.ch"
        self.search_url = f"{self.base_url}/medreg/search"
        self.delay = delay_between_requests
        
        # Initialize session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logger.info(f"Initialized MedRegScraper with {delay_between_requests}s delay")
    
    def search_doctor(self, first_name: str, last_name: str) -> Optional[Dict]:
        """
        Search for a doctor by name and extract their information
        
        Args:
            first_name: Doctor's first name
            last_name: Doctor's last name
            
        Returns:
            Dictionary with doctor information or None if not found
        """
        try:
            logger.info(f"Searching for: {first_name} {last_name}")
            
            # Get the search page first to obtain any necessary form tokens
            response = self.session.get(self.search_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Prepare search data
            search_data = {
                'name': last_name,
                'vorname': first_name,
                'gln': '',
                'beruf': '',  # profession - leave empty for all
                'fachrichtung': '',  # specialist qualification - leave empty for all
                'zusatzbezeichnung': '',  # additional qualifications - leave empty for all
                'geschlecht': '',  # gender - leave empty for all
                'sprachkenntnisse': '',  # language skills - leave empty for all
                'staatsangehoerigkeit': '',  # citizenship - leave empty for all
            }
            
            # Submit search
            search_response = self.session.post(self.search_url, data=search_data)
            search_response.raise_for_status()
            
            # Parse search results
            result_soup = BeautifulSoup(search_response.content, 'html.parser')
            
            # Look for doctor information in the results
            doctor_info = self._extract_doctor_info(result_soup, first_name, last_name)
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            return doctor_info
            
        except Exception as e:
            logger.error(f"Error searching for {first_name} {last_name}: {e}")
            return None
    
    def _extract_doctor_info(self, soup: BeautifulSoup, first_name: str, last_name: str) -> Optional[Dict]:
        """
        Extract doctor information from search results page
        
        Args:
            soup: BeautifulSoup object of the search results page
            first_name: Expected first name
            last_name: Expected last name
            
        Returns:
            Dictionary with extracted information
        """
        try:
            # Look for results table or cards
            # The exact structure needs to be determined from actual page inspection
            
            # Initialize result structure
            doctor_data = {
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'licence_date': None,
                'specialist_qualifications': [],
                'additional_qualifications': [],
                'search_successful': False,
                'multiple_matches': False,
                'error_message': None
            }
            
            # Look for common result containers
            results = soup.find_all(['div', 'tr', 'li'], class_=re.compile(r'(result|person|doctor|entry)', re.I))
            
            if not results:
                # Try alternative selectors
                results = soup.find_all('div', string=re.compile(rf'{first_name}.*{last_name}|{last_name}.*{first_name}', re.I))
            
            if not results:
                doctor_data['error_message'] = "No results found"
                logger.warning(f"No results found for {first_name} {last_name}")
                return doctor_data
            
            # Check if we have multiple matches
            if len(results) > 1:
                doctor_data['multiple_matches'] = True
                logger.info(f"Multiple matches found for {first_name} {last_name}")
            
            # Extract information from first/best match
            first_result = results[0]
            
            # Extract licence date
            licence_date = self._extract_licence_date(first_result)
            if licence_date:
                doctor_data['licence_date'] = licence_date
            
            # Extract specialist qualifications
            specialist_quals = self._extract_specialist_qualifications(first_result)
            doctor_data['specialist_qualifications'] = specialist_quals
            
            # Extract additional qualifications
            additional_quals = self._extract_additional_qualifications(first_result)
            doctor_data['additional_qualifications'] = additional_quals
            
            doctor_data['search_successful'] = True
            logger.info(f"Successfully extracted data for {first_name} {last_name}")
            
            return doctor_data
            
        except Exception as e:
            logger.error(f"Error extracting info for {first_name} {last_name}: {e}")
            return {
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'licence_date': None,
                'specialist_qualifications': [],
                'additional_qualifications': [],
                'search_successful': False,
                'multiple_matches': False,
                'error_message': str(e)
            }
    
    def _extract_licence_date(self, element) -> Optional[str]:
        """Extract licence date from result element"""
        try:
            # Look for date patterns (Swiss format: DD.MM.YYYY)
            date_pattern = r'\d{1,2}\.\d{1,2}\.\d{4}'
            text = element.get_text()
            
            # Look for licence-related keywords
            licence_keywords = ['approbation', 'berechtigung', 'lizenz', 'licence', 'bewilligung']
            
            # Search for date near licence keywords
            lines = text.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in licence_keywords):
                    dates = re.findall(date_pattern, line)
                    if dates:
                        return dates[0]
            
            # If no specific context found, look for any date
            dates = re.findall(date_pattern, text)
            if dates:
                return dates[0]  # Return first date found
                
        except Exception as e:
            logger.error(f"Error extracting licence date: {e}")
        
        return None
    
    def _extract_specialist_qualifications(self, element) -> List[str]:
        """Extract specialist qualifications from result element"""
        try:
            qualifications = []
            text = element.get_text()
            
            # Look for specialist qualification keywords
            specialist_keywords = ['fachrichtung', 'spezialist', 'specialist', 'facharzt']
            
            lines = text.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in specialist_keywords):
                    # Extract qualification name after keyword
                    for keyword in specialist_keywords:
                        if keyword in line.lower():
                            # Split and clean
                            parts = line.split(':')
                            if len(parts) > 1:
                                qual = parts[1].strip()
                                if qual and qual not in qualifications:
                                    qualifications.append(qual)
            
            return qualifications
            
        except Exception as e:
            logger.error(f"Error extracting specialist qualifications: {e}")
            return []
    
    def _extract_additional_qualifications(self, element) -> List[str]:
        """Extract additional qualifications from result element"""
        try:
            qualifications = []
            text = element.get_text()
            
            # Look for additional qualification keywords
            additional_keywords = ['zusatzbezeichnung', 'additional', 'zusatz', 'weiterbildung']
            
            lines = text.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in additional_keywords):
                    # Extract qualification name after keyword
                    for keyword in additional_keywords:
                        if keyword in line.lower():
                            # Split and clean
                            parts = line.split(':')
                            if len(parts) > 1:
                                qual = parts[1].strip()
                                if qual and qual not in qualifications:
                                    qualifications.append(qual)
            
            return qualifications
            
        except Exception as e:
            logger.error(f"Error extracting additional qualifications: {e}")
            return []

def parse_doctor_names(file_path: str) -> List[Tuple[str, str]]:
    """
    Parse doctor names from the Excel file
    
    Args:
        file_path: Path to the Excel file with doctor list
        
    Returns:
        List of (first_name, last_name) tuples
    """
    try:
        df = pd.read_excel(file_path)
        
        # Extract unique names
        unique_names = df['Mitglieder mit Einsatzfunktion'].unique()
        
        # Parse names - format is "First Last (Role)"
        pattern = r'^([^(]+)\s*\([^)]+\)$'
        doctors = []
        
        for name_entry in unique_names:
            match = re.match(pattern, name_entry)
            if match:
                full_name = match.group(1).strip()
                # Split into first and last name
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = ' '.join(name_parts[1:])
                    doctors.append((first_name, last_name))
        
        logger.info(f"Parsed {len(doctors)} unique doctors from {file_path}")
        return doctors
        
    except Exception as e:
        logger.error(f"Error parsing doctor names: {e}")
        return []

if __name__ == "__main__":
    # Test with a sample doctor
    scraper = MedRegScraper()
    
    # Test search
    result = scraper.search_doctor("Adrian", "Lüthi")
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("No result found")