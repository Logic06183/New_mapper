import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from pathlib import Path
import PyPDF2
import docx
from docx import Document
from sentence_transformers.util import pytorch_cos_sim

class MappingStatus(Enum):
    TO_DO = "To do"
    SUCCESSFULLY_MAPPED = "Successfully mapped"
    MARKED_TO_RECONSIDER = "Marked to reconsider"
    MARKED_UNMAPPABLE = "Marked unmappable"

class StudyContext:
    """Class to handle study documentation and extract relevant context"""
    
    def __init__(self):
        """Initialize an empty study context"""
        self.documents = {}
        self.contexts = {}
        
    def add_document(self, file_path: str):
        """Add a document to the study context"""
        try:
            text = self._extract_text(file_path)
            self.documents[file_path] = text
            self._process_document(text)
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from a document file"""
        ext = file_path.lower().split('.')[-1]
        
        if ext == 'pdf':
            return self._extract_from_pdf(file_path)
        elif ext == 'docx':
            return self._extract_from_docx(file_path)
        elif ext == 'txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _process_document(self, text: str):
        """Process document text to extract variable contexts"""
        # Split text into paragraphs
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            # Look for variable definitions
            matches = re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*[:=-]\s*([^.\n]+)', paragraph)
            for match in matches:
                var_name = match.group(1).strip()
                description = match.group(2).strip()
                if var_name not in self.contexts:
                    self.contexts[var_name] = []
                self.contexts[var_name].append(description)
            
            # Look for variable mentions with context
            matches = re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s+(?:is|was|represents|measures|indicates|refers to)\s+([^.\n]+)', paragraph)
            for match in matches:
                var_name = match.group(1).strip()
                description = match.group(2).strip()
                if var_name not in self.contexts:
                    self.contexts[var_name] = []
                self.contexts[var_name].append(description)
    
    def get_variable_context(self, variable_name: str) -> str:
        """Get context for a specific variable"""
        contexts = self.contexts.get(variable_name, [])
        return ' | '.join(contexts) if contexts else ""
    
    def get_all_contexts(self) -> List[str]:
        """Get all unique contexts from study documentation"""
        all_contexts = []
        for var_name, contexts in self.contexts.items():
            context = f"{var_name}: {' | '.join(contexts)}"
            all_contexts.append(context)
        return all_contexts

    def get_document_text(self, file_path: str) -> str:
        """Get the full text of a specific document"""
        return self.documents.get(file_path, "")

class VariableMapper:
    """Class to map variables between datasets using semantic matching"""
    
    def __init__(self, codebook: Union[str, pd.DataFrame], study_context: Optional[StudyContext] = None):
        """Initialize with codebook and optional study context"""
        self.study_context = study_context
        
        # Load codebook
        if isinstance(codebook, str):
            self.codebook = self._load_codebook(codebook)
        elif isinstance(codebook, pd.DataFrame):
            self.codebook = codebook
        else:
            raise ValueError("Codebook must be either a file path or pandas DataFrame")
            
        # Identify column names
        self._identify_columns()
            
        # Create embeddings for codebook variables
        self.variable_embeddings = {}
        self._create_codebook_embeddings()
        
    def _identify_columns(self):
        """Identify the variable and description columns in the codebook"""
        # Print column names for debugging
        print("Available columns in codebook:", self.codebook.columns.tolist())
        
        # Common names for variable column
        var_columns = ['Variable', 'variable', 'Variable Name', 'variable_name', 'name', 'var', 'VARIABLE']
        # Common names for description column
        desc_columns = ['Description', 'description', 'Variable Description', 'desc', 'DESCRIPTION']
        
        # Find variable column
        self.var_column = None
        for col in var_columns:
            if col in self.codebook.columns:
                self.var_column = col
                break
                
        # Find description column
        self.desc_column = None
        for col in desc_columns:
            if col in self.codebook.columns:
                self.desc_column = col
                break
                
        if self.var_column is None:
            # If no standard name found, use the first column
            self.var_column = self.codebook.columns[0]
            print(f"Using '{self.var_column}' as variable column")
            
        if self.desc_column is None and len(self.codebook.columns) > 1:
            # If no standard name found, use the second column
            self.desc_column = self.codebook.columns[1]
            print(f"Using '{self.desc_column}' as description column")
        
    def _load_codebook(self, file_path: str) -> pd.DataFrame:
        """Load codebook from file"""
        ext = Path(file_path).suffix.lower()
        
        try:
            if ext == '.csv':
                return pd.read_csv(file_path)
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported codebook format: {ext}")
        except Exception as e:
            raise ValueError(f"Error loading codebook: {str(e)}")
    
    def _get_variable_description(self, variable_name: str) -> str:
        """Get description for a variable from codebook and study context"""
        # Try to get description from codebook
        codebook_desc = ""
        if self.desc_column:
            desc_row = self.codebook[self.codebook[self.var_column] == variable_name]
            if not desc_row.empty:
                codebook_desc = desc_row[self.desc_column].iloc[0]
        
        # Try to get context from study documentation
        study_context = ""
        if self.study_context:
            study_context = self.study_context.get_variable_context(variable_name)
        
        # Combine descriptions
        descriptions = []
        if codebook_desc and pd.notna(codebook_desc):
            descriptions.append(str(codebook_desc))
        if study_context:
            descriptions.append(study_context)
            
        return ' | '.join(descriptions) if descriptions else variable_name
    
    def _create_codebook_embeddings(self):
        """Create embeddings for codebook variables"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for _, row in self.codebook.iterrows():
            var_name = row[self.var_column]
            
            # Skip if variable name is missing or NaN
            if pd.isna(var_name):
                continue
                
            # Get description from codebook and study context
            description = self._get_variable_description(var_name)
            
            # Create embedding from variable name and description
            text = f"{var_name} {description}"
            embedding = model.encode(text, convert_to_tensor=True)
            self.variable_embeddings[var_name] = embedding
    
    def map_variables(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Map variables from input data to codebook variables"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        mappings = []
        
        for column in data.columns:
            # Get description and create embedding
            description = self._get_variable_description(column)
            query_text = f"{column} {description}"
            query_embedding = model.encode(query_text, convert_to_tensor=True)
            
            # Calculate cosine similarity with all codebook variables
            similarities = {}
            for var_name, var_embedding in self.variable_embeddings.items():
                similarity = pytorch_cos_sim(query_embedding, var_embedding).item()
                similarities[var_name] = similarity
            
            # Sort by similarity and get top matches
            sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_matches = sorted_matches[:5]  # Get top 5 matches
            
            # Create mapping entry
            mapping = {
                'dataset_variable': column,
                'matches': [
                    {
                        'codebook_variable': var_name,
                        'similarity_score': score,
                        'description': self._get_variable_description(var_name)
                    }
                    for var_name, score in top_matches
                ],
                'status': MappingStatus.TO_DO
            }
            mappings.append(mapping)
        
        return mappings
