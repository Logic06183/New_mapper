import pandas as pd
from variable_mapper import VariableMapper
import os
from pathlib import Path
import sys

# Set console encoding to UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

def clean_text(text):
    """Clean text for display"""
    if not isinstance(text, str):
        return str(text)
    return text.encode('ascii', 'replace').decode('ascii')

def print_mapping_results(mappings):
    """Print mapping results in a readable format"""
    for var_name, mapping in mappings.items():
        print("\n" + "=" * 50)
        print(f"Source Variable: {clean_text(var_name)}")
        if 'source_description' in mapping:
            print(f"Description: {clean_text(mapping['source_description'])}")
        print(f"Source Type: {clean_text(mapping['source_type'])}")
        print(f"Status: {clean_text(mapping['mapping_status'])}")
        print()
        
        if mapping['suggested_matches']:
            print("Suggested Matches:\n")
            for match in mapping['suggested_matches']:
                print(f"  Match: {clean_text(match['variable'])}")
                print(f"  Abbreviation: {clean_text(match['abbreviation'])}")
                print(f"  Confidence: {match['score']}%")
                print(f"  Description: {clean_text(match['description'])}")
                print(f"  Units: {clean_text(str(match['units']))}")
                print(f"  Reason: {clean_text(match['match_explanation'])}\n")
                
            if mapping['transformations']:
                print("Required Transformations:")
                for transform in mapping['transformations']:
                    print(f"  - {clean_text(transform)}")
        else:
            print("No matches found in codebook")

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize paths
    codebook_path = os.path.join(current_dir, 'codebook.xlsx')
    variables_path = os.path.join(current_dir, 'variables (2).csv')
    data_path = os.path.join(current_dir, 'iLiNS-DYAD-M 18mo development 2014-11-12 (3).csv')
    
    # Get list of study documentation files
    study_docs = []
    for ext in ['*.pdf', '*.doc', '*.docx', '*.txt']:
        study_docs.extend([str(p) for p in Path(current_dir).glob(ext)])
        
    if not study_docs:
        print("Warning: No study documentation files found")
    else:
        print(f"Found {len(study_docs)} study documentation files")
    
    # Create mapper instance with study documentation
    mapper = VariableMapper(
        codebook_path=codebook_path,
        variables_path=variables_path,
        study_docs=study_docs
    )
    
    # Load sample data
    data = pd.read_csv(data_path)
    
    # Get mappings
    mappings = mapper.map_variables(data)
    
    # Print results
    print_mapping_results(mappings)

if __name__ == "__main__":
    main()
