from pathlib import Path
from typing import List , Any
from langchain_community.document_loaders import PyPDFLoader , PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader,TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """ 
     load all supported files from the data directory  and convert them into Langchain documents structure.
     Supported : csv, excel, pdf, JSON, TXT, word 
     """
    
    # use project data root folder 
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path {data_path}")
    documents = []

    # PDF files 
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} pdf files{[str(f) for f in pdf_files]}")
    
    # for loading pdf files
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} from pdf_files {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    
    # for loading txt files 
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"[DEBUG] Found {len(txt_files)} txt_files {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
            continue
    
    # for loading csv files 
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} csv_files {[str(f) for f in csv_files]} ")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    # for loading excel files
    excel_files = list(data_path.glob("**/*.xlsx"))
    print(f"[DEBUG] Found {len(excel_files)} excel_files {[str(f) for f in excel_files]}")
    for excel_file in excel_files:
        try:
            loader = UnstructuredExcelLoader(str(excel_file), mode="elements")
            loaded = loader.load()
            print(f"Loaded {len(loaded)} from {excel_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading {excel_file}: {e}")
            continue
    
    # for loading JSON files

    json_files = list(data_path.glob("**/*.json"))
    print(f"[DEBUG] Found {len(json_files)} json files {[str(f) for f in json_files]}")
    for json_file in json_files:
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    # for loading word files
    word_files = list(data_path.glob("**/*.docx"))
    print(f"[DEBUG] Found {len(word_files)} word files {[str(f) for f in word_files]}")
    for word_file in word_files:
        try:
            loader = Docx2txtLoader(str(word_file))
            loaded = loader.load()
            print(f"Loaded {len(loaded)} from {word_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"Error loading {word_file}: {e}")
            continue
    
    return documents

            