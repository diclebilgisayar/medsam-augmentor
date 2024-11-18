# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:33:59 2024

@author: cezerilab
"""

import os

def print_directory_structure(startpath):
    """
    Print the directory structure starting from the given path.
    
    Parameters:
    startpath (str): The root directory path to start scanning
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        print(f'{indent}├── {os.path.basename(root)}/')
        subindent = '│   ' * (level + 1)
        for file in files:
            if not file.startswith('.'):  # Skip hidden files
                print(f'{subindent}├── {file}')

def scan_project_structure(project_path):
    """
    Scan and display the project structure, excluding common unnecessary directories and files.
    
    Parameters:
    project_path (str): Path to the project root directory
    """
    exclude_dirs = {'.git', '__pycache__', '.idea', '.vscode', 'venv', 'env'}
    exclude_files = {'.DS_Store', '.gitignore', '*.pyc'}
    
    print("\nProject Structure:")
    print(f"{project_path}/")
    
    for root, dirs, files in os.walk(project_path):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        level = root.replace(project_path, '').count(os.sep)
        indent = '│   ' * level
        
        # Print current directory
        if level > 0:
            print(f'{indent}├── {os.path.basename(root)}/')
        
        # Print files in current directory
        subindent = '│   ' * (level + 1)
        for file in sorted(files):
            if not any(file.endswith(ext) for ext in exclude_files) and not file.startswith('.'):
                print(f'{subindent}├── {file}')

if __name__ == "__main__":
    # Get the project path from user
    project_path = input("Enter the path to your project directory: ")
    
    # Check if path exists
    if os.path.exists(project_path):
        scan_project_structure(project_path)
    else:
        print("Invalid path. Please enter a valid directory path.")