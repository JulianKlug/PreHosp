#!/usr/bin/env python3
"""
Inspect the actual form structure of the medreg website
"""

import requests
from bs4 import BeautifulSoup
import json

def inspect_search_form():
    """Inspect the search form to understand the proper submission method"""
    url = "https://www.medregom.admin.ch/medreg/search"
    
    try:
        # Get the search page
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("=== FORM INSPECTION ===")
        
        # Find all forms
        forms = soup.find_all('form')
        print(f"Found {len(forms)} forms")
        
        for i, form in enumerate(forms):
            print(f"\n--- Form {i+1} ---")
            print(f"Action: {form.get('action', 'Not specified')}")
            print(f"Method: {form.get('method', 'GET')}")
            print(f"Enctype: {form.get('enctype', 'Not specified')}")
            
            # Find all input fields
            inputs = form.find_all(['input', 'select', 'textarea'])
            print(f"Input fields ({len(inputs)}):")
            
            for inp in inputs:
                name = inp.get('name', 'No name')
                inp_type = inp.get('type', 'text')
                value = inp.get('value', '')
                placeholder = inp.get('placeholder', '')
                
                print(f"  - {name}: type={inp_type}, value='{value}', placeholder='{placeholder}'")
                
                # For select elements, show options
                if inp.name == 'select':
                    options = inp.find_all('option')
                    print(f"    Options: {[opt.get('value', opt.text.strip()) for opt in options[:5]]}")
        
        # Look for any CSRF tokens or similar
        csrf_inputs = soup.find_all('input', attrs={'name': re.compile(r'(csrf|token|_token)', re.I)})
        if csrf_inputs:
            print(f"\nCSRF/Token fields found:")
            for inp in csrf_inputs:
                print(f"  - {inp.get('name')}: {inp.get('value', '')}")
        
        # Save raw HTML for inspection
        with open('search_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"\nSaved raw HTML to search_page.html")
        
        # Look for JavaScript or AJAX endpoints
        scripts = soup.find_all('script')
        print(f"\nFound {len(scripts)} script tags")
        for script in scripts:
            if script.string and ('ajax' in script.string.lower() or 'fetch' in script.string.lower()):
                print("Found potential AJAX/fetch in script")
                
    except Exception as e:
        print(f"Error inspecting form: {e}")

if __name__ == "__main__":
    import re
    inspect_search_form()