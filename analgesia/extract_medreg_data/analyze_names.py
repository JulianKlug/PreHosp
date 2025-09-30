#!/usr/bin/env python3
"""
Analyze the doctor names format in more detail
"""

import pandas as pd
import re

def analyze_name_format():
    """Analyze the format of doctor names"""
    file_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/prehospital/analgesia/data/Liste Notärzte-1.xlsx'
    
    df = pd.read_excel(file_path)
    
    print("=== NAME FORMAT ANALYSIS ===")
    print("Sample entries:")
    for i in range(min(20, len(df))):
        print(f"{i+1}: {df.iloc[i]['Mitglieder mit Einsatzfunktion']}")
    
    # Extract unique name patterns
    names = df['Mitglieder mit Einsatzfunktion'].unique()
    print(f"\nTotal unique entries: {len(names)}")
    
    # Analyze the pattern - seems to be "FirstName LastName (Role)"
    pattern = r'^([^(]+)\s*\([^)]+\)$'
    clean_names = []
    
    print("\nExtracting clean names:")
    for name in names[:10]:  # Show first 10
        match = re.match(pattern, name)
        if match:
            clean_name = match.group(1).strip()
            clean_names.append(clean_name)
            print(f"'{name}' -> '{clean_name}'")
        else:
            print(f"No match for: '{name}'")
    
    print(f"\nUnique doctors (first 20):")
    unique_clean_names = list(set(clean_names))
    for i, name in enumerate(unique_clean_names[:20]):
        print(f"{i+1}: {name}")
    
    return unique_clean_names

if __name__ == "__main__":
    analyze_name_format()