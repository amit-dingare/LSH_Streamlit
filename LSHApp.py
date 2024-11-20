import streamlit as st
import numpy as np
from typing import Set, List, Dict, Tuple
import os
import json
import pandas as pd
import PyPDF2
import io
import pdfplumber
from datasketch import MinHash, MinHashLSH
from collections import defaultdict

class FileProcessor:
    """Handle different file types and extract text content"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file with robust error handling"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            texts = []
            
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        texts.append(text)
                except Exception as e:
                    st.warning(f"Skipped page {page_num + 1} due to error: {str(e)}")
                    continue
            
            if not texts:
                st.warning("No text could be extracted from the PDF")
                return ""
                
            return "\n".join(texts).strip()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_csv(file_content: bytes) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            # Convert all columns to string and join
            text = "\n".join(
                df.astype(str).apply(
                    lambda x: " ".join(x.dropna().values), axis=1
                )
            )
            return text.strip()
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_json(file_content: bytes) -> str:
        """Extract text from JSON file"""
        try:
            json_data = json.loads(file_content.decode('utf-8'))
            # Recursively extract all values from JSON
            def extract_values(obj):
                if isinstance(obj, dict):
                    return ' '.join(str(v) for v in obj.values())
                elif isinstance(obj, list):
                    return ' '.join(str(extract_values(item)) for item in obj)
                else:
                    return str(obj)
            return extract_values(json_data)
        except Exception as e:
            st.error(f"Error reading JSON: {str(e)}")
            return ""

    @staticmethod
    def process_file(file) -> str:
        """Process file based on its extension and return text content"""
        try:
            file_content = file.read()
            file_extension = os.path.splitext(file.name)[1].lower()
            
            # Reset file pointer for potential reuse
            file.seek(0)
            
            if file_extension == '.pdf':
                return FileProcessor.extract_text_from_pdf(file_content)
            elif file_extension == '.csv':
                return FileProcessor.extract_text_from_csv(file_content)
            elif file_extension == '.json':
                return FileProcessor.extract_text_from_json(file_content)
            else:  # .txt, .md, .py files
                return file_content.decode('utf-8')
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            return ""

class DocumentSimilarity:
    """Handle document similarity using datasketch's MinHash LSH"""
    
    def __init__(self, num_perm=128, threshold=0.5):
        """
        Initialize LSH with given parameters
        
        Args:
            num_perm (int): Number of permutations for MinHash
            threshold (float): Jaccard similarity threshold
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}
        self.documents = {}

    def get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Convert text to k-shingles"""
        # Clean and normalize text
        text = ' '.join(text.split())  # Normalize whitespace
        return {text[i:i+k].encode('utf-8') for i in range(len(text) - k + 1)}

    def insert(self, doc_id: str, text: str) -> bool:
        """Insert a document into the LSH index"""
        try:
            # Generate shingles
            shingles = self.get_shingles(text)
            if not shingles:
                return False

            # Create MinHash
            minhash = MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                minhash.update(shingle)

            # Store document and its minhash
            self.documents[doc_id] = text
            self.minhashes[doc_id] = minhash

            # Insert into LSH
            self.lsh.insert(doc_id, minhash)
            return True
        except Exception as e:
            st.error(f"Error inserting document {doc_id}: {str(e)}")
            return False

    def find_similar(self, doc_id: str) -> List[Tuple[str, float]]:
        """Find similar documents to the given document"""
        if doc_id not in self.minhashes:
            return []

        try:
            # Query LSH for similar documents
            similar_docs = self.lsh.query(self.minhashes[doc_id])
            similar_docs.remove(doc_id)  # Remove self from results

            # Calculate actual similarities
            results = []
            for similar_id in similar_docs:
                similarity = self.minhashes[doc_id].jaccard(self.minhashes[similar_id])
                results.append((similar_id, similarity))

            # Sort by similarity
            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as e:
            st.error(f"Error finding similar documents: {str(e)}")
            return []

def main():
    st.set_page_config(
        page_title="Multi-format LSH Document Similarity Finder",
        layout="wide"
    )
    
    st.title("🔍 Multi-format LSH Document Similarity Finder")
    st.write("""
    This app uses Locality Sensitive Hashing (LSH) with MinHash to find similar documents.
    Supports PDF, TXT, CSV, MD, PY, and JSON files.
    """)

    # Initialize session state
    if 'doc_similarity' not in st.session_state:
        st.session_state.doc_similarity = DocumentSimilarity()
        st.session_state.uploaded_files = {}
        st.session_state.file_types = {}

    # LSH Parameters in sidebar
    with st.sidebar:
        st.markdown("### LSH Parameters")
        num_perm = st.slider(
            "Number of Permutations",
            min_value=64,
            max_value=256,
            value=128,
            step=32,
            help="More permutations = higher accuracy but slower processing"
        )
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum Jaccard similarity for documents to be considered similar"
        )
        
        if st.button("Update LSH Parameters"):
            st.session_state.doc_similarity = DocumentSimilarity(
                num_perm=num_perm,
                threshold=threshold
            )
            # Reinsert all documents with new parameters
            for doc_id, text in st.session_state.uploaded_files.items():
                st.session_state.doc_similarity.insert(doc_id, text)
            st.success("Parameters updated!")

    # File uploader
    with st.expander("Upload Files", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'csv', 'md', 'py', 'json'],
            accept_multiple_files=True,
            help="Upload files to compare (PDF, TXT, CSV, MD, PY, JSON)"
        )

        # Process uploaded files
        if uploaded_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                if file.name not in st.session_state.uploaded_files:
                    with st.spinner(f'Processing {file.name}...'):
                        text_content = FileProcessor.process_file(file)
                        if text_content:  # Only add if text extraction was successful
                            success = st.session_state.doc_similarity.insert(file.name, text_content)
                            if success:
                                st.session_state.uploaded_files[file.name] = text_content
                                st.session_state.file_types[file.name] = os.path.splitext(file.name)[1]
                progress_bar.progress((i + 1) / len(uploaded_files))
            progress_bar.empty()

    # Show uploaded files and similarity search
    if st.session_state.uploaded_files:
        st.write(f"📚 Uploaded Files: {len(st.session_state.uploaded_files)}")
        
        selected_file = st.selectbox(
            "Select a file to compare",
            options=list(st.session_state.uploaded_files.keys()),
            help="Choose the reference file to find similar documents",
            format_func=lambda x: f"{x} ({st.session_state.file_types[x]})"
        )

        if st.button("Find Similar Documents"):
            if selected_file:
                with st.spinner("Finding similar documents..."):
                    similar_docs = st.session_state.doc_similarity.find_similar(selected_file)

                if similar_docs:
                    st.write("### 📋 Similar Documents Found")
                    for doc_id, similarity in similar_docs:
                        with st.expander(
                            f"{doc_id} ({st.session_state.file_types[doc_id]}) - Similarity: {similarity:.2%}"
                        ):
                            preview_text = st.session_state.uploaded_files[doc_id]
                            if len(preview_text) > 1000:
                                preview_text = preview_text[:1000] + "..."
                            st.text_area(
                                "Content Preview",
                                preview_text,
                                height=150,
                                disabled=True
                            )
                else:
                    st.info("No similar documents found with the current threshold.")

        # Add option to clear all files
        if st.button("Clear All Files"):
            st.session_state.doc_similarity = DocumentSimilarity(
                num_perm=num_perm,
                threshold=threshold
            )
            st.session_state.uploaded_files = {}
            st.session_state.file_types = {}
            st.experimental_rerun()

    # Display help information
    with st.sidebar:
        st.markdown("""
        ### Supported File Types
        - PDF (`.pdf`)
        - Text files (`.txt`)
        - CSV files (`.csv`)
        - Markdown (`.md`)
        - Python files (`.py`)
        - JSON files (`.json`)
        
        ### How to Use
        1. Upload multiple files using the file uploader
        2. Select a reference file from the dropdown
        3. Adjust LSH parameters if needed
        4. Click "Find Similar Documents" to see results
        
        ### Processing Details
        - PDFs: Extracts text from all pages
        - CSVs: Combines all cell values
        - JSON: Extracts all values recursively
        - TXT/MD/PY: Uses raw text content
        """)

if __name__ == "__main__":
    main()