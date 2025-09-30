#!/usr/bin/env python3
"""
Selenium-based Medical Registry Scraper for Swiss doctors
This scraper uses browser automation to interact with the JavaScript SPA
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import logging
import re
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medreg_selenium_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SeleniumMedRegScraper:
    """
    Selenium-based scraper for the Swiss Medical Registry website
    """
    
    def __init__(self, headless: bool = True, delay_between_requests: float = 3.0):
        """
        Initialize the scraper
        
        Args:
            headless: Whether to run browser in headless mode
            delay_between_requests: Delay in seconds between requests
        """
        self.base_url = "https://www.medregom.admin.ch"
        self.search_url = f"{self.base_url}/medreg/search"
        self.delay = delay_between_requests
        self.headless = headless
        self.driver = None
        
        logger.info(f"Initialized SeleniumMedRegScraper with {delay_between_requests}s delay, headless={headless}")
    
    def start_driver(self):
        """Initialize the Chrome driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            # Use webdriver-manager to automatically download and manage ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("Chrome driver started successfully")
        except Exception as e:
            logger.error(f"Failed to start Chrome driver: {e}")
            raise
    
    def close_driver(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
            logger.info("Chrome driver closed")
    
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
            
            # Navigate to search page
            self.driver.get(self.search_url)
            
            # Wait for page to load and find input fields
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[matInput]"))
            )            # Look for name input fields using the exact IDs found in debugging
            # The website has clear IDs: 'name' for last name and 'firstName' for first name
            
            # Wait for the form to be fully loaded
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "name"))
            )
            time.sleep(2)
            
            name_field = None  # for last name
            firstname_field = None  # for first name
            
            # Use exact IDs discovered in debugging
            try:
                name_field = self.driver.find_element(By.ID, "name")
                logger.info("Found last name field by ID 'name'")
            except Exception as e:
                logger.error(f"Could not find last name field by ID: {e}")
            
            try:
                firstname_field = self.driver.find_element(By.ID, "firstName")
                logger.info("Found first name field by ID 'firstName'")
            except Exception as e:
                logger.error(f"Could not find first name field by ID: {e}")
            
            # Fill the fields if found
            if name_field:
                try:
                    # Clear and fill last name
                    name_field.clear()
                    name_field.send_keys(last_name)
                    logger.info(f"Successfully entered last name: {last_name}")
                except Exception as e:
                    logger.error(f"Error entering last name: {e}")
            else:
                logger.warning("Could not find last name field")
            
            if firstname_field:
                try:
                    # Clear and fill first name
                    firstname_field.clear()
                    firstname_field.send_keys(first_name)
                    logger.info(f"Successfully entered first name: {first_name}")
                except Exception as e:
                    logger.error(f"Error entering first name: {e}")
            else:
                logger.warning("Could not find first name field")
            
            # Wait a moment after filling fields
            time.sleep(2)
            
            # Use the exact button ID found in debugging: 'button-search'
            search_submitted = False
            
            try:
                # Find the search button by its exact ID
                search_button = self.driver.find_element(By.ID, "button-search")
                logger.info("Found search button by ID 'button-search'")
                
                # Multiple strategies to click the button
                click_methods = [
                    ("JavaScript click", lambda: self.driver.execute_script("arguments[0].click();", search_button)),
                    ("Regular click", lambda: search_button.click()),
                    ("Actions click", lambda: ActionChains(self.driver).move_to_element(search_button).click().perform())
                ]
                
                for method_name, click_method in click_methods:
                    try:
                        logger.info(f"Trying {method_name}")
                        
                        # Ensure button is in view and clickable
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", search_button)
                        time.sleep(0.5)
                        
                        # Execute click
                        click_method()
                        logger.info(f"Executed {method_name}")
                        
                        # Wait and check if search was submitted
                        time.sleep(5)
                        
                        # Check for results or URL change
                        current_url = self.driver.current_url
                        page_source = self.driver.page_source.lower()
                        
                        # Look for signs that search was executed
                        if ("result" in page_source or 
                            "no result" in page_source or 
                            "keine ergebnis" in page_source or
                            "search result" in page_source or
                            current_url != self.search_url):
                            search_submitted = True
                            logger.info(f"Search submitted successfully with {method_name}")
                            break
                        else:
                            logger.warning(f"{method_name} did not trigger search - no results page detected")
                            
                    except Exception as e:
                        logger.warning(f"{method_name} failed: {e}")
                        continue
                
                if not search_submitted:
                    logger.warning("All click methods failed to submit search")
                    
            except Exception as e:
                logger.error(f"Could not find search button: {e}")
            
            # Fallback: Try Enter key submission if button click didn't work
            if not search_submitted:
                try:
                    logger.info("Trying Enter key submission as fallback")
                    active_field = firstname_field if firstname_field else name_field
                    if active_field:
                        active_field.send_keys("\n")
                        time.sleep(5)
                        
                        # Check if search was submitted
                        current_url = self.driver.current_url
                        page_source = self.driver.page_source.lower()
                        
                        if ("result" in page_source or 
                            "no result" in page_source or 
                            current_url != self.search_url):
                            search_submitted = True
                            logger.info("Search submitted successfully with Enter key")
                        
                except Exception as e:
                    logger.error(f"Enter key submission failed: {e}")
            
            # Final fallback: Try direct form submission
            if not search_submitted:
                try:
                    logger.info("Trying direct form submission via JavaScript")
                    # Submit the form containing the name field
                    self.driver.execute_script("""
                        var nameField = document.getElementById('name');
                        if (nameField && nameField.form) {
                            nameField.form.submit();
                        }
                    """)
                    time.sleep(5)
                    search_submitted = True
                    logger.info("Form submitted via JavaScript")
                except Exception as e:
                    logger.error(f"JavaScript form submission failed: {e}")
            
            if not search_submitted:
                logger.error("All search submission methods failed")
                # Take a screenshot for debugging
                try:
                    self.driver.save_screenshot("search_submission_failed.png")
                    logger.info("Screenshot saved as search_submission_failed.png")
                except Exception:
                    pass
            
            # Wait for results to load - increased time for list to appear
            time.sleep(8)  # Longer wait for results list
            
            # Extract results - now expecting a list of doctors to select from
            doctor_info = self._extract_doctor_info_selenium(first_name, last_name)
            
            # Add delay between requests
            time.sleep(self.delay)
            
            return doctor_info
            
        except Exception as e:
            logger.error(f"Error searching for {first_name} {last_name}: {e}")
            return {
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'licence_date': None,
                'year_of_birth': None,
                'specialist_qualifications': [],
                'additional_qualifications': [],
                'search_successful': False,
                'multiple_matches': False,
                'error_message': str(e)
            }
    
    def _extract_doctor_info_selenium(self, first_name: str, last_name: str) -> Dict:
        """
        Extract doctor information from search results page using Selenium
        Now handles a list of search results where we need to click on the correct doctor
        """
        try:
            # Initialize result structure
            doctor_data = {
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'licence_date': None,
                'year_of_birth': None,
                'specialist_qualifications': [],
                'additional_qualifications': [],
                'search_successful': False,
                'multiple_matches': False,
                'error_message': None,
                'selected_result_text': None
            }
            
            # Wait for results list to appear
            wait = WebDriverWait(self.driver, 20)
            
            # Look for result containers - try different selectors for result lists
            result_selectors = [
                "tr",  # Table rows (common for result lists)
                ".mat-list-item",
                ".result-item",
                ".person-result", 
                ".search-result",
                "mat-card",
                ".mat-card",
                ".list-item",
                "[role='listitem']",
                ".doctor-entry",
                ".person-entry"
            ]
            
            results = []
            
            for selector in result_selectors:
                try:
                    # Wait for any results to appear
                    elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
                    if elements:
                        # Filter elements that actually contain meaningful text (not just empty rows)
                        text_elements = []
                        for elem in elements:
                            elem_text = elem.text.strip()
                            # Look for elements that contain names (have at least 2 words)
                            if len(elem_text.split()) >= 2 and any(c.isalpha() for c in elem_text):
                                text_elements.append(elem)
                        
                        if text_elements:
                            results = text_elements
                            logger.info(f"Found {len(results)} meaningful results with selector: {selector}")
                            # Log first few results for debugging
                            for i, result in enumerate(results[:5]):
                                logger.info(f"Result {i}: '{result.text[:100]}'")
                            break
                except TimeoutException:
                    continue
            
            if not results:
                # Try to get page text to see what's there
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                logger.info(f"No structured results found. Page text preview: {page_text[:1000]}")
                
                if first_name.lower() in page_text.lower() and last_name.lower() in page_text.lower():
                    logger.info("Doctor names found in page text but no structured results")
                    doctor_data['search_successful'] = True
                    doctor_data['error_message'] = "Found doctor in page text but could not parse structured results"
                    doctor_data['selected_result_text'] = page_text[:1000]
                else:
                    doctor_data['error_message'] = "No results found"
                    logger.warning(f"No results found for {first_name} {last_name}")
                return doctor_data
            
            # Select the best matching result from the list
            selected_result = self._select_and_click_best_match(results, first_name, last_name)
            
            if not selected_result:
                doctor_data['error_message'] = "No matching result found in list"
                doctor_data['multiple_matches'] = len(results) > 1
                return doctor_data
            
            # Check for multiple matches
            if len(results) > 1:
                doctor_data['multiple_matches'] = True
                logger.info(f"Multiple matches found ({len(results)}) for {first_name} {last_name}, selected and clicked best match")
            
            # After clicking, wait for detailed page to load
            time.sleep(5)
            
            # Now extract detailed information from the doctor's detail page
            detail_page_text = self.driver.find_element(By.TAG_NAME, "body").text
            doctor_data['selected_result_text'] = detail_page_text
            
            # Extract licence date from detail page
            licence_date = self._extract_licence_date_selenium(detail_page_text)
            if licence_date:
                doctor_data['licence_date'] = licence_date
            
            # Extract year of birth from detail page
            year_of_birth = self._extract_year_of_birth_selenium(detail_page_text)
            if year_of_birth:
                doctor_data['year_of_birth'] = year_of_birth
            
            # Extract specialist qualifications from detail page
            specialist_quals = self._extract_specialist_qualifications_selenium(detail_page_text)
            doctor_data['specialist_qualifications'] = specialist_quals
            
            # Extract additional qualifications from detail page
            additional_quals = self._extract_additional_qualifications_selenium(detail_page_text)
            doctor_data['additional_qualifications'] = additional_quals
            
            doctor_data['search_successful'] = True
            logger.info(f"Successfully extracted detailed data for {first_name} {last_name}")
            
            return doctor_data
            
        except Exception as e:
            logger.error(f"Error extracting info for {first_name} {last_name}: {e}")
            return {
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'licence_date': None,
                'year_of_birth': None,
                'specialist_qualifications': [],
                'additional_qualifications': [],
                'search_successful': False,
                'multiple_matches': False,
                'error_message': str(e)
            }
    
    def _extract_licence_date_selenium(self, text: str) -> Optional[str]:
        """Extract licence date from result text"""
        try:
            # Look for Swiss date patterns (DD.MM.YYYY)
            date_pattern = r'\d{1,2}\.\d{1,2}\.\d{4}'
            
            # Look for licence-related keywords in German/French
            licence_keywords = [
                'approbation', 'berechtigung', 'lizenz', 'licence', 'bewilligung',
                'erteilung', 'ausstellung', 'erteilt', 'ausgestellt'
            ]
            
            lines = text.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in licence_keywords):
                    dates = re.findall(date_pattern, line)
                    if dates:
                        return dates[0]
            
            # If no specific context, look for any date
            dates = re.findall(date_pattern, text)
            if dates:
                return dates[0]
                
        except Exception as e:
            logger.error(f"Error extracting licence date: {e}")
        
        return None
    
    def _extract_year_of_birth_selenium(self, text: str) -> Optional[str]:
        """Extract year of birth from result text"""
        try:
            # Look for "Year of birth:" pattern followed by year
            year_pattern = r'Year of birth:\s*(\d{4})'
            match = re.search(year_pattern, text)
            if match:
                return match.group(1)
            
            # Alternative patterns for German/French
            year_patterns = [
                r'Geburtsjahr:\s*(\d{4})',
                r'Année de naissance:\s*(\d{4})',
                r'Birth year:\s*(\d{4})'
            ]
            
            for pattern in year_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
                    
        except Exception as e:
            logger.error(f"Error extracting year of birth: {e}")
        
        return None
    
    def _extract_specialist_qualifications_selenium(self, text: str) -> List[str]:
        """Extract specialist qualifications from result text"""
        try:
            qualifications = []
            
            # Look for the specialist qualification section
            specialist_section_pattern = r'\(Postgraduate\) specialist qualification.*?(?=Additional qualifications|Professional licence|Footer|$)'
            specialist_match = re.search(specialist_section_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if specialist_match:
                specialist_text = specialist_match.group(0)
                
                # Extract all titles from the specialist section
                title_pattern = r'Title:\s*([^\n]+)'
                titles = re.findall(title_pattern, specialist_text)
                
                for title in titles:
                    title = title.strip()
                    if title and title not in qualifications:
                        qualifications.append(title)
            
            # Also try to extract from simpler patterns in case the above fails
            if not qualifications:
                # Look for patterns like "Physician, Anaesthesiology, Intensive care medicine"
                physician_pattern = r'Physician[,\s]+([^\n]+)'
                physician_matches = re.findall(physician_pattern, text, re.IGNORECASE)
                
                for match in physician_matches:
                    # Split by comma and clean up
                    specialties = [spec.strip() for spec in match.split(',')]
                    for specialty in specialties:
                        # Skip generic terms
                        if specialty and specialty.lower() not in ['physician', 'doctor'] and specialty not in qualifications:
                            qualifications.append(specialty)
            
            return qualifications
            
        except Exception as e:
            logger.error(f"Error extracting specialist qualifications: {e}")
            return []
    
    def _extract_additional_qualifications_selenium(self, text: str) -> List[str]:
        """Extract additional qualifications from result text"""
        try:
            qualifications = []
            
            # Look for the additional qualifications section
            additional_section_pattern = r'Additional qualifications \(through private organisations or through FSVO\).*?(?=Professional licence|Footer|$)'
            additional_match = re.search(additional_section_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if additional_match:
                additional_text = additional_match.group(0)
                
                # Extract all designations from the additional section
                designation_pattern = r'Designation:\s*([^\n]+)'
                designations = re.findall(designation_pattern, additional_text)
                
                for designation in designations:
                    designation = designation.strip()
                    if designation and designation not in qualifications:
                        qualifications.append(designation)
            
            return qualifications
            
        except Exception as e:
            logger.error(f"Error extracting additional qualifications: {e}")
            return []
    
    def _select_best_matching_result(self, results, first_name: str, last_name: str):
        """
        Select the best matching result from search results
        
        Args:
            results: List of WebElement results
            first_name: Expected first name
            last_name: Expected last name
            
        Returns:
            Best matching WebElement or None
        """
        try:
            best_match = None
            best_score = 0
            
            logger.info(f"Selecting best match from {len(results)} results for {first_name} {last_name}")
            
            for i, result in enumerate(results):
                try:
                    result_text = result.text.lower()
                    score = 0
                    
                    # Score based on name matching
                    first_name_lower = first_name.lower()
                    last_name_lower = last_name.lower()
                    
                    # Exact first name match
                    if first_name_lower in result_text:
                        score += 10
                        
                    # Exact last name match
                    if last_name_lower in result_text:
                        score += 10
                        
                    # Both names present
                    if first_name_lower in result_text and last_name_lower in result_text:
                        score += 20
                        
                    # Bonus for full name match
                    full_name_variations = [
                        f"{first_name_lower} {last_name_lower}",
                        f"{last_name_lower} {first_name_lower}",
                        f"{last_name_lower}, {first_name_lower}"
                    ]
                    
                    for variation in full_name_variations:
                        if variation in result_text:
                            score += 30
                            break
                    
                    # Penalty for additional names (indicating different person)
                    name_words = result_text.split()
                    name_count = sum(1 for word in name_words if len(word) > 2 and word.isalpha())
                    if name_count > 3:  # More than first, last, and one middle name
                        score -= 5
                    
                    logger.info(f"Result {i}: score={score}, text preview: '{result_text[:100]}'")
                    
                    if score > best_score:
                        best_score = score
                        best_match = result
                        
                except Exception as e:
                    logger.error(f"Error scoring result {i}: {e}")
                    continue
            
            if best_match and best_score > 5:  # Minimum threshold
                logger.info(f"Selected result with score {best_score}")
                return best_match
            else:
                logger.warning(f"No result met minimum score threshold (best score: {best_score})")
                return None
                
        except Exception as e:
            logger.error(f"Error selecting best match: {e}")
            return results[0] if results else None
    
    def _select_and_click_best_match(self, results, first_name: str, last_name: str):
        """
        Select the best matching result from search results list and click on it
        
        Args:
            results: List of WebElement results from search
            first_name: Expected first name
            last_name: Expected last name
            
        Returns:
            The selected WebElement or None if no good match found
        """
        try:
            best_match = None
            best_score = 0
            
            logger.info(f"Selecting and clicking best match from {len(results)} results for {first_name} {last_name}")
            
            for i, result in enumerate(results):
                try:
                    result_text = result.text.lower()
                    score = 0
                    
                    # Score based on name matching
                    first_name_lower = first_name.lower()
                    last_name_lower = last_name.lower()
                    
                    # Exact first name match
                    if first_name_lower in result_text:
                        score += 10
                        
                    # Exact last name match
                    if last_name_lower in result_text:
                        score += 10
                        
                    # Both names present
                    if first_name_lower in result_text and last_name_lower in result_text:
                        score += 20
                        
                    # Bonus for full name match (various formats)
                    full_name_variations = [
                        f"{first_name_lower} {last_name_lower}",
                        f"{last_name_lower} {first_name_lower}",
                        f"{last_name_lower}, {first_name_lower}",
                        f"{first_name_lower}, {last_name_lower}"
                    ]
                    
                    for variation in full_name_variations:
                        if variation in result_text:
                            score += 30
                            break
                    
                    # Penalty for additional names (indicating potentially different person)
                    words = result_text.split()
                    name_like_words = [word for word in words if len(word) > 2 and word.isalpha()]
                    if len(name_like_words) > 4:  # More than expected names
                        score -= 3
                    
                    logger.info(f"Result {i}: score={score}, text: '{result_text[:150]}'")
                    
                    if score > best_score:
                        best_score = score
                        best_match = result
                        
                except Exception as e:
                    logger.error(f"Error scoring result {i}: {e}")
                    continue
            
            if best_match and best_score > 10:  # Minimum threshold for clicking
                logger.info(f"Selected result with score {best_score}, attempting to click")
                
                # Try multiple methods to click on the selected result
                click_methods = [
                    ("JavaScript click", lambda: self.driver.execute_script("arguments[0].click();", best_match)),
                    ("Regular click", lambda: best_match.click()),
                    ("Actions click", lambda: ActionChains(self.driver).move_to_element(best_match).click().perform())
                ]
                
                for method_name, click_method in click_methods:
                    try:
                        # Scroll to element
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", best_match)
                        time.sleep(0.5)
                        
                        # Try the click
                        click_method()
                        logger.info(f"Successfully clicked on best match using {method_name}")
                        
                        # Wait for detail page to load
                        time.sleep(3)
                        
                        # Check if we navigated to a detail page
                        current_url = self.driver.current_url
                        logger.info(f"After click, current URL: {current_url}")
                        
                        return best_match
                        
                    except Exception as e:
                        logger.warning(f"Click method {method_name} failed: {e}")
                        continue
                
                logger.error("All click methods failed for the selected result")
                return None
                
            else:
                logger.warning(f"No result met minimum score threshold (best score: {best_score})")
                return None
                
        except Exception as e:
            logger.error(f"Error in _select_and_click_best_match: {e}")
            return None

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

def test_scraper():
    """Test the scraper with sample doctors"""
    scraper = SeleniumMedRegScraper(headless=False, delay_between_requests=3.0)
    
    # Test with multiple doctors
    test_doctors = [
        ("Adrian", "Lüthi"),
        ("Alessa", "Grossbach"),
        ("Alessandro", "Genini")
    ]
    
    try:
        scraper.start_driver()
        
        for first_name, last_name in test_doctors:
            print(f"\n=== Testing {first_name} {last_name} ===")
            result = scraper.search_doctor(first_name, last_name)
            if result:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("No result found")
            
            # Small delay between tests
            time.sleep(2)
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        scraper.close_driver()

if __name__ == "__main__":
    test_scraper()