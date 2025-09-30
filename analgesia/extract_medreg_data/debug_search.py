#!/usr/bin/env python3
"""
Debug script to test search functionality step by step
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def debug_search():
    """Debug the search functionality step by step"""
    
    # Setup Chrome driver
    chrome_options = Options()
    # Don't use headless mode for debugging
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        print("=== DEBUGGING SEARCH FUNCTIONALITY ===")
        
        # Navigate to search page
        url = "https://www.medregom.admin.ch/medreg/search"
        print(f"Navigating to: {url}")
        driver.get(url)
        
        # Wait for page to load
        print("Waiting for page to load...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)
        
        # Check current page title and URL
        print(f"Page title: {driver.title}")
        print(f"Current URL: {driver.current_url}")
        
        # Find all input fields
        inputs = driver.find_elements(By.CSS_SELECTOR, "input")
        print(f"\nFound {len(inputs)} input fields:")
        
        for i, inp in enumerate(inputs):
            input_type = inp.get_attribute('type') or 'text'
            input_id = inp.get_attribute('id') or 'no-id'
            input_name = inp.get_attribute('name') or 'no-name'
            placeholder = inp.get_attribute('placeholder') or 'no-placeholder'
            print(f"  Input {i}: type='{input_type}', id='{input_id}', name='{input_name}', placeholder='{placeholder}'")
        
        # Look for labels to understand the form structure
        labels = driver.find_elements(By.TAG_NAME, "label")
        print(f"\nFound {len(labels)} labels:")
        for i, label in enumerate(labels):
            label_text = label.text.strip()
            for_attr = label.get_attribute('for') or 'no-for'
            if label_text:
                print(f"  Label {i}: '{label_text}' (for='{for_attr}')")
        
        # Try to identify name fields by looking at the visible text structure
        print("\n=== TRYING TO FILL FIELDS ===")
        
        # Method: Use the first two visible input fields (typically Name and First name)
        visible_inputs = [inp for inp in inputs if inp.is_displayed() and inp.is_enabled()]
        print(f"Found {len(visible_inputs)} visible and enabled inputs")
        
        if len(visible_inputs) >= 2:
            # Assume first input is Last Name, second is First Name
            last_name_field = visible_inputs[0]
            first_name_field = visible_inputs[1]
            
            print("Filling last name field...")
            last_name_field.clear()
            last_name_field.send_keys("Lüthi")
            
            print("Filling first name field...")
            first_name_field.clear()
            first_name_field.send_keys("Adrian")
            
            print("Fields filled successfully!")
            
            # Take a screenshot
            driver.save_screenshot("after_filling_fields.png")
            print("Screenshot saved as after_filling_fields.png")
            
            # Wait and look for submit button
            time.sleep(2)
            
            print("\n=== LOOKING FOR SUBMIT BUTTON ===")
            buttons = driver.find_elements(By.TAG_NAME, "button")
            print(f"Found {len(buttons)} buttons:")
            
            for i, button in enumerate(buttons):
                button_text = button.text.strip()
                button_type = button.get_attribute('type') or 'button'
                button_id = button.get_attribute('id') or 'no-id'
                is_displayed = button.is_displayed()
                is_enabled = button.is_enabled()
                print(f"  Button {i}: text='{button_text}', type='{button_type}', id='{button_id}', visible={is_displayed}, enabled={is_enabled}")
            
            # Try to submit with Enter key first
            print("\n=== TRYING ENTER KEY SUBMISSION ===")
            try:
                first_name_field.send_keys("\n")
                print("Pressed Enter key")
                time.sleep(5)
                
                # Check if page changed
                new_url = driver.current_url
                page_text = driver.find_element(By.TAG_NAME, "body").text[:500]
                print(f"After Enter - URL: {new_url}")
                print(f"Page content preview: {page_text}")
                
                if "Adrian" in page_text and "Lüthi" in page_text:
                    print("SUCCESS: Found search results!")
                else:
                    print("No results found, trying button click...")
                    
                    # Find and click submit button
                    submit_buttons = [btn for btn in buttons if 
                                    btn.get_attribute('type') == 'submit' or 
                                    'search' in btn.text.lower() or 
                                    'suchen' in btn.text.lower()]
                    
                    if submit_buttons:
                        submit_button = submit_buttons[0]
                        print(f"Found submit button: {submit_button.text}")
                        
                        # Try JavaScript click
                        driver.execute_script("arguments[0].click();", submit_button)
                        print("Clicked submit button with JavaScript")
                        time.sleep(5)
                        
                        # Check results again
                        final_url = driver.current_url
                        final_text = driver.find_element(By.TAG_NAME, "body").text[:500]
                        print(f"Final URL: {final_url}")
                        print(f"Final page content: {final_text}")
                        
            except Exception as e:
                print(f"Error during submission: {e}")
        
        else:
            print("Not enough visible input fields found")
        
        # Keep browser open for manual inspection
        input("Press Enter to close browser...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    debug_search()