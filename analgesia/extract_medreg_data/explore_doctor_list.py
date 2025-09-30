#!/usr/bin/env python3
"""
Explore the doctor list Excel file to understand the data structure
"""

import pandas as pd
import os

def explore_doctor_list():
    """Explore the structure of the doctor list"""
    file_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/prehospital/analgesia/data/Liste Notärzte-1.xlsx'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        print("=== DOCTOR LIST EXPLORATION ===")
        print(f"File path: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\n--- First 5 rows ---")
        print(df.head())
        print("\n--- Data types ---")
        print(df.dtypes)
        print("\n--- Non-null counts ---")
        print(df.info())
        
        # Check for name columns
        name_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['name', 'nom', 'vorname', 'prenom', 'firstname', 'lastname'])]
        print(f"\n--- Potential name columns: {name_cols} ---")
        if name_cols:
            for col in name_cols:
                print(f"{col}: {df[col].head().tolist()}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    explore_doctor_list()