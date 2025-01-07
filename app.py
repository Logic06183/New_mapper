import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from variable_mapper import VariableMapper, StudyContext, MappingStatus
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path
import gc

# Initialize session state
if 'mapper' not in st.session_state:
    st.session_state['mapper'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'codebook' not in st.session_state:
    st.session_state['codebook'] = None
if 'mappings' not in st.session_state:
    st.session_state['mappings'] = None

# Set page config
st.set_page_config(
    page_title="Variable Mapping Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def compute_embeddings(texts: List[str]) -> np.ndarray:
    model = get_sentence_transformer()
    return model.encode(texts, convert_to_tensor=False)

def load_file(uploaded_file, file_type="dataset"):
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    return pd.read_csv(uploaded_file, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            st.error(f"Could not read {file_type} with any of the attempted encodings")
            return None
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format for {file_type}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_type}: {str(e)}")
        return None

def save_mapping_feedback(mapping: Dict[str, Any], feedback: bool):
    """Save mapping feedback to improve future matches"""
    feedback_dir = "mapping_feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    feedback_file = os.path.join(feedback_dir, "mapping_feedback.json")
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = {"positive": [], "negative": []}
    
    feedback_entry = {
        "dataset_variable": mapping["dataset_variable"],
        "matches": mapping["matches"],
        "timestamp": datetime.now().isoformat()
    }
    
    if feedback:
        feedback_data["positive"].append(feedback_entry)
    else:
        feedback_data["negative"].append(feedback_entry)
    
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
                        options=[s.value for s in MappingStatus],
                        key=f"status_{var_name}"
                    )
                
                # Add feedback buttons
                col1, col2 = st.columns(2)
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
            create_3d_visualization(mappings, data, codebook)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.error("Please ensure all required data is loaded correctly.")

def create_3d_visualization(mappings, data, codebook):
    """Create 3D visualization of semantic relationships"""
    with st.spinner("Generating 3D visualization..."):
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

def main():
    st.title("Variable Mapping Assistant")
    st.markdown("""
    Upload your dataset and codebook to start mapping variables. 
    The system will suggest matches based on semantic similarity.
    """)

    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.file_uploader("Upload Dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        if dataset:
            st.session_state['data'] = load_file(dataset, "dataset")
            if st.session_state['data'] is not None:
                st.write("Dataset Preview:")
                st.dataframe(st.session_state['data'].head())
    
    with col2:
        codebook = st.file_uploader("Upload Codebook (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        if codebook:
            st.session_state['codebook'] = load_file(codebook, "codebook")
            if st.session_state['codebook'] is not None:
                st.write("Codebook Preview:")
                st.dataframe(st.session_state['codebook'].head())

    # Optional study documentation
    study_docs = st.file_uploader("Upload Study Documentation (Optional)", 
                                 type=['pdf', 'docx', 'txt'],
                                 accept_multiple_files=True)

    # Process mappings
    if st.session_state['data'] is not None and st.session_state['codebook'] is not None:
        if st.button("Generate Mappings"):
            with st.spinner("Generating variable mappings..."):
                try:
                    if 'mapper' not in st.session_state or st.session_state['mapper'] is None:
                        st.session_state['mapper'] = VariableMapper(st.session_state['codebook'])
                    
                    # Create mappings
                    st.session_state['mappings'] = st.session_state['mapper'].map_variables(
                        st.session_state['data']
                    )
                    
                    # Display mappings
                    display_mappings(
                        st.session_state['mappings'],
                        st.session_state['data'],
                        st.session_state['codebook']
                    )
                    
                    # Clean up memory
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"Error generating mappings: {str(e)}")
        
        # Display existing mappings if available
        elif st.session_state['mappings'] is not None:
            display_mappings(
                st.session_state['mappings'],
                st.session_state['data'],
                st.session_state['codebook']
            )

if __name__ == "__main__":
    main()
