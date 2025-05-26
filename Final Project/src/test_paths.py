#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify file paths after reorganization
"""

import os
import pandas as pd

def test_data_paths():
    """Test if data files can be accessed from src folder"""
    print("Testing data file paths...")
    
    # Test data file paths
    data_path1 = "../data/fina_project_data01.xlsx"
    data_path2 = "../data/fina_project_data02.xlsx"
    
    print(f"Checking: {data_path1}")
    if os.path.exists(data_path1):
        print("✓ Data file 1 found")
        try:
            df1 = pd.read_excel(data_path1)
            print(f"✓ Data file 1 loaded: {df1.shape}")
        except Exception as e:
            print(f"✗ Error loading data file 1: {e}")
    else:
        print("✗ Data file 1 not found")
    
    print(f"Checking: {data_path2}")
    if os.path.exists(data_path2):
        print("✓ Data file 2 found")
        try:
            df2 = pd.read_excel(data_path2)
            print(f"✓ Data file 2 loaded: {df2.shape}")
        except Exception as e:
            print(f"✗ Error loading data file 2: {e}")
    else:
        print("✗ Data file 2 not found")

def test_image_paths():
    """Test if images folder is accessible"""
    print("\nTesting image folder path...")
    
    images_dir = "../images"
    if os.path.exists(images_dir):
        print("✓ Images folder found")
        print(f"✓ Images folder path: {os.path.abspath(images_dir)}")
    else:
        print("✗ Images folder not found")

if __name__ == "__main__":
    print("=== Path Testing ===")
    print(f"Current working directory: {os.getcwd()}")
    test_data_paths()
    test_image_paths()
    print("\n=== Test Complete ===")
