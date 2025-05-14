#!/usr/bin/env python3

import os
import re
import glob

"""Add Content Security Policy to HTML reports to ensure proper display"""

def fix_html_report(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if it's already fixed
    if '<meta http-equiv="Content-Security-Policy"' in content:
        print(f"File {filename} already has CSP, skipping.")
        return
    
    # Add CSP meta tag to the head
    csp_meta = '<meta http-equiv="Content-Security-Policy" content="default-src *; style-src * \'unsafe-inline\'; script-src * \'unsafe-inline\'; img-src * data:; connect-src *;">'
    
    # Insert CSP meta tag after the <head> tag
    if '<head>' in content:
        modified = content.replace('<head>', '<head>\n    ' + csp_meta)
        
        # Write the modified content back
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(modified)
        
        print(f"Fixed CSP in {filename}")
    else:
        print(f"Warning: Could not find <head> tag in {filename}")

def main():
    # Find all HTML files in results directory
    report_files = glob.glob('results/*.html')
    
    for filename in report_files:
        try:
            fix_html_report(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Processed {len(report_files)} report files.")

if __name__ == '__main__':
    main() 