import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from variable_mapper import VariableMapper, StudyContext, MappingStatus
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import os
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime
from pathlib import Path

# Set page config
st.set_page_config(page_title="Variable Mapping Assistant", layout="wide")

# Initialize SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def compute_embeddings(texts):
    """Compute embeddings for a list of texts"""
    return model.encode(texts)

def plot_semantic_space(variable_embeddings, codebook_embeddings, study_doc_embeddings,
                       variable_names, codebook_vars, doc_texts):
    """Create interactive 3D plot of semantic space"""
    # Combine all embeddings
    all_embeddings = np.vstack([variable_embeddings, codebook_embeddings, study_doc_embeddings])
    
    # Apply t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(all_embeddings)
    
    # Split back into separate arrays
    n_vars = len(variable_names)
    n_codebook = len(codebook_vars)
    
    var_coords = embeddings_3d[:n_vars]
    codebook_coords = embeddings_3d[n_vars:n_vars+n_codebook]
    doc_coords = embeddings_3d[n_vars+n_codebook:]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add variables
    fig.add_trace(go.Scatter3d(
        x=var_coords[:, 0], y=var_coords[:, 1], z=var_coords[:, 2],
        text=variable_names,
        mode='markers',
        name='Dataset Variables',
        marker=dict(size=8, color='blue', opacity=0.8)
    ))
    
    # Add codebook variables
    fig.add_trace(go.Scatter3d(
        x=codebook_coords[:, 0], y=codebook_coords[:, 1], z=codebook_coords[:, 2],
        text=codebook_vars,
        mode='markers',
        name='Codebook Variables',
        marker=dict(size=8, color='red', opacity=0.8)
    ))
    
    # Add study documentation embeddings
    fig.add_trace(go.Scatter3d(
        x=doc_coords[:, 0], y=doc_coords[:, 1], z=doc_coords[:, 2],
        text=doc_texts,
        mode='markers',
        name='Study Documentation',
        marker=dict(size=8, color='green', opacity=0.8)
    ))
    
    fig.update_layout(
        title="3D Semantic Space Visualization",
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3"
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def load_data(file, file_type="dataset"):
    """Load data from CSV or Excel file with error handling"""
    try:
        # Get file extension
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Try different encodings and delimiters for CSV
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            delimiters = [',', ';', '\t']
            
            # Store error messages for debugging
            errors = []
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(
                            file, 
                            encoding=encoding, 
                            sep=delimiter,
                            on_bad_lines='warn'  # More permissive parsing
                        )
                        if len(df.columns) > 1:  # Check if we got meaningful data
                            st.success(f"Successfully loaded {file_type} using {encoding} encoding and '{delimiter}' delimiter")
                            return df
                    except Exception as e:
                        errors.append(f"Attempt with {encoding}, {delimiter}: {str(e)}")
                        continue
            
            # If all attempts fail, try pandas' auto-detection
            try:
                df = pd.read_csv(file, engine='python')
                if len(df.columns) > 1:
                    st.success(f"Successfully loaded {file_type} using auto-detection")
                    return df
            except Exception as e:
                errors.append(f"Auto-detection attempt: {str(e)}")
            
            # If we get here, show detailed error information
            st.error(f"Error loading {file_type}. Attempted the following:")
            for error in errors:
                st.text(error)
            return None
            
        elif file_extension in ['xls', 'xlsx']:
            try:
                # Try reading with default sheet
                df = pd.read_excel(file)
                if len(df.columns) > 1:
                    st.success(f"Successfully loaded {file_type} from Excel file")
                    return df
                
                # If first attempt fails, try listing sheets and let user choose
                xls = pd.ExcelFile(file)
                if len(xls.sheet_names) > 1:
                    sheet_name = st.selectbox(
                        f"Multiple sheets found in {file_type}. Please select one:",
                        xls.sheet_names
                    )
                    df = pd.read_excel(file, sheet_name=sheet_name)
                    if len(df.columns) > 1:
                        st.success(f"Successfully loaded {file_type} from sheet: {sheet_name}")
                        return df
                
                st.error(f"No valid data found in {file_type}")
                return None
                
            except Exception as e:
                st.error(f"Error loading Excel {file_type}: {str(e)}")
                return None
        else:
            st.error(f"Unsupported file format for {file_type}: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error loading {file_type}: {str(e)}")
        return None

def create_3d_visualization(mappings, data, codebook):
    """Create 3D visualization of semantic relationships"""
    # Get embeddings for dataset variables
    dataset_texts = [f"{var}: {var}" for var in data.columns]
    variable_embeddings = compute_embeddings(dataset_texts)
    
    # Get embeddings for codebook variables
    codebook_texts = []
    for _, row in codebook.iterrows():
        var_name = row[codebook.columns[0]]  # Use first column as variable name
        description = row[codebook.columns[1]] if len(codebook.columns) > 1 else ""
        codebook_texts.append(f"{var_name}: {description}")
    codebook_embeddings = compute_embeddings(codebook_texts)
    
    # Combine all embeddings for visualization
    all_embeddings = np.vstack([variable_embeddings, codebook_embeddings])
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(all_embeddings)
    
    # Split back into separate arrays
    n_vars = len(dataset_texts)
    var_coords = embeddings_3d[:n_vars]
    codebook_coords = embeddings_3d[n_vars:]
    
    # Create scatter plots
    fig = go.Figure()
    
    # Dataset variables (blue)
    fig.add_trace(go.Scatter3d(
        x=var_coords[:, 0],
        y=var_coords[:, 1],
        z=var_coords[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='blue', opacity=0.7),
        text=data.columns,
        name='Dataset Variables'
    ))
    
    # Codebook variables (red)
    fig.add_trace(go.Scatter3d(
        x=codebook_coords[:, 0],
        y=codebook_coords[:, 1],
        z=codebook_coords[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='red', opacity=0.7),
        text=[t.split(':')[0] for t in codebook_texts],
        name='Codebook Variables'
    ))
    
    # Update layout
    fig.update_layout(
        title='Semantic Relationships in 3D Space',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        showlegend=True
    )
    
    st.plotly_chart(fig)

def save_mapping_feedback(mapping: Dict[str, Any], feedback: bool):
    """Save mapping feedback to improve future matches"""
    # Create feedback directory if it doesn't exist
    feedback_dir = "mapping_feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Load existing feedback or create new
    feedback_file = os.path.join(feedback_dir, "mapping_feedback.json")
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = {"positive": [], "negative": []}
    
    # Add new feedback
    feedback_entry = {
        "dataset_variable": mapping["dataset_variable"],
        "matches": mapping["matches"],
        "timestamp": datetime.now().isoformat()
    }
    
    if feedback:
        feedback_data["positive"].append(feedback_entry)
    else:
        feedback_data["negative"].append(feedback_entry)
    
    # Save updated feedback
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)

def display_mappings(mappings: List[Dict[str, Any]], data: pd.DataFrame, codebook: pd.DataFrame):
    """Display variable mappings with interactive elements"""
    st.header("Variable Mappings")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Visualization"])
    
    with tab1:
        # Create a table of mappings
        for mapping in mappings:
            var_name = mapping['dataset_variable']
            matches = mapping['matches']
            status = mapping['status']
            
            # Create an expander for each variable
            with st.expander(f" {var_name}"):
                # Show variable info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Info:**")
                    if var_name in data.columns:
                        st.write(f"Type: {data[var_name].dtype}")
                        st.write(f"Sample values: {', '.join(map(str, data[var_name].head(3)))}")
                
                # Show matches in a table
                st.write("**Suggested Matches:**")
                matches_df = pd.DataFrame([
                    {
                        'Codebook Variable': match['codebook_variable'],
                        'Similarity': f"{match['similarity_score']:.2%}",
                        'Description': match['description']
                    }
                    for match in matches
                ])
                st.dataframe(matches_df)
                
                # Add mapping controls
                col1, col2 = st.columns(2)
                with col1:
                    status = st.selectbox(
                        "Status",
                        options=["Mapped", "Unmapped"],
                        key=f"status_{var_name}"
                    )
                
                # Add feedback buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(" Correct Match", key=f"correct_{var_name}"):
                        save_mapping_feedback(mapping, True)
                        st.success("Feedback saved! This will help improve future matches.")
                with col2:
                    if st.button(" Incorrect Match", key=f"incorrect_{var_name}"):
                        save_mapping_feedback(mapping, False)
                        st.success("Feedback saved! This will help improve future matches.")
    
    with tab2:
        try:
            # Create 3D visualization
            create_3d_visualization(mappings, data, codebook)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.error("Please ensure all required data is loaded correctly.")

def main():
    st.title("Variable Mapping Assistant")
    st.markdown("""
    This app demonstrates how study documentation enhances semantic matching for variable mapping.
    The 3D visualization shows the semantic relationships between:
    - Dataset variables (blue)
    - Codebook variables (red)
    - Study documentation context (green)
    """)
    
    # File upload section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Upload Dataset")
        data_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            key="data"
        )
        
    with col2:
        st.subheader("Upload Codebook")
        codebook_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            key="codebook"
        )
        
    with col3:
        st.subheader("Upload Study Docs")
        study_docs = st.file_uploader(
            "Choose documentation files", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
    
    if data_file and codebook_file:
        # Load data with error handling
        data = load_data(data_file, "dataset")
        codebook = load_data(codebook_file, "codebook")
        
        if data is not None and codebook is not None:
            # Create study context
            study_context = StudyContext()
            
            # Process study documents if uploaded
            if study_docs:
                doc_dir = Path("study_docs")
                doc_dir.mkdir(exist_ok=True)
                
                for doc in study_docs:
                    doc_path = doc_dir / doc.name
                    with open(doc_path, "wb") as f:
                        f.write(doc.getvalue())
                    try:
                        study_context.add_document(str(doc_path))
                    except Exception as e:
                        st.warning(f"Error processing document {doc.name}: {str(e)}")
            
            # Initialize mapper
            mapper = VariableMapper(codebook, study_context)
            
            # Display data preview
            st.subheader("Dataset Preview")
            st.dataframe(data.head())
            
            st.subheader("Codebook Preview")
            st.dataframe(codebook.head())
            
            # Get embeddings
            variable_texts = [f"{var}: {study_context.get_variable_context(var)}" 
                            for var in data.columns]
            codebook_texts = [f"{row['Variable Name']}: {row.get('Description', '')}" 
                            for _, row in codebook.iterrows()]
            doc_texts = study_context.get_all_contexts() if study_docs else []
            
            variable_embeddings = compute_embeddings(variable_texts)
            codebook_embeddings = compute_embeddings(codebook_texts)
            study_doc_embeddings = compute_embeddings(doc_texts) if doc_texts else np.array([])
            
            # Create visualization
            if len(study_doc_embeddings) > 0:
                fig = plot_semantic_space(
                    variable_embeddings, 
                    codebook_embeddings,
                    study_doc_embeddings,
                    data.columns,
                    codebook['Variable Name'].tolist(),
                    doc_texts
                )
                st.plotly_chart(fig)
            
            # Show mappings
            st.subheader("Variable Mappings")
            mappings = mapper.map_variables(data)
            
            # Display mappings in an interactive table
            display_mappings(mappings, data, codebook)

if __name__ == "__main__":
    main()
