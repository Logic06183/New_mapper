# Variable Mapping Assistant

A Streamlit app that helps map variables between datasets using semantic matching and study documentation.

## Features

- Upload and preview datasets in CSV or Excel format
- Upload codebooks with variable descriptions
- Upload study documentation (PDF, DOCX, TXT) for enhanced matching
- Interactive 3D visualization of semantic relationships
- Feedback system to improve matching over time
- Export mapping results

## Installation

```bash
# Clone the repository
git clone https://github.com/Logic06183/New_mapper.git
cd New_mapper

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage

1. Upload your dataset (CSV or Excel)
2. Upload your codebook (CSV or Excel)
3. Upload study documentation (PDF, DOCX, TXT)
4. Review suggested variable mappings
5. Provide feedback to improve future matches
6. Export your mappings

## File Format Requirements

### Dataset
- CSV or Excel file
- First row should contain variable names

### Codebook
- CSV or Excel file
- Should contain variable names and descriptions

### Study Documentation
- PDF, DOCX, or TXT files
- Should contain relevant variable descriptions and context

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
