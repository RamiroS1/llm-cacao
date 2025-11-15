import dspy
import os
import argparse
import json
from pathlib import Path

from rag_creation.utils import *

parser = argparse.ArgumentParser('extract text from .pdf files', add_help=False)
parser.add_argument('--path_root', default="./llm_cacao-dragro", type=str, help='path to root data folder')
parser.add_argument('--remove_headers', default=True, type=bool, help='flag to remove header text from pdf')
parser.add_argument('--flag_rewrite', default=True, type=bool, help='flag to rewrite from exixting Q&A dataset')
parser.add_argument('--skip_savedata', default=False, type=bool, help='flag to rewrite from exixting Q&A dataset')
args = parser.parse_args()

def get_folders(path="."):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_docs(folder_path):
    return list(Path(folder_path).rglob("*.docx"))


lm = dspy.LM('ollama_chat/mistral',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0,
    model_kwargs={
        "format": "json",          # <- fuerza salida JSON en Ollama
        "options": {"num_ctx": 8192}  # opcional
    }
)
dspy.configure(lm=lm)

path_docs = os.path.join(args.path_root, "dataDocs_small.jsonl")

# Extract Q&A samples
if not args.skip_savedata:
    if not Path(path_docs).exists() or args.flag_rewrite: 
        containers = get_folders(args.path_root)
        
        for conti, container in enumerate(containers):
            subject = re.match(r"[A-Za-z]+", container).group()
            docs_files = get_docs(os.path.join(args.path_root, container))
            
            for i, doc in enumerate(docs_files):
                print("Processing document:", doc)
                
                # Extract text
                text, tables = word_to_markdown(doc)
                
                # Extract paragraphs    
                paragraphs = split_by_paragraphs(text)
                paragraphs = split_long_paragraphs(paragraphs)
                paragraphs = [c for c in paragraphs if is_content_chunk(c)]
                
                
                docs = []
                if len(tables) > 0:
                    for ii, table in enumerate(tables):
                        dic_doc = {"id": doc.stem + f"_{ii:05d}_table", "source": "simple_LLM", "subject": subject,
                                "doc_name": doc.stem, "paragraph": table}
                        
                        docs.append(dic_doc)
            
                
                for ii, prg in enumerate(paragraphs):
                    
                    dic_doc = {"id": doc.stem + f"_{ii:05d}", "source": "simple_LLM", "subject": subject, "doc_name": doc.stem,
                            "paragraph": prg}
                    
                    docs.append(dic_doc)
                    
                # save DOCS full dataset
                with open(path_docs, "a", encoding="utf-8") as f:
                    for entry in docs:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print("number of DOCs:", len(docs))
                
    else:
        print("already processed docs dataset")  
        
          
# RAG module
embedder = SentenceTransformer('intfloat/multilingual-e5-large')

path_fais = os.path.join(args.path_root, "profiles_index.faiss")
path_colbert = os.path.join(args.path_root, "profiles_corpus.tsv")
path_docsp = os.path.join(args.path_root, "profiles_docs.pkl")

if Path(path_docs).exists():
    
    # Load Q&As from the JSON file 
    with open(path_docs) as f:
        data = [json.loads(line) for line in f]
        
    documents = [{"paragraph": item["paragraph"], "subject": item["subject"]} for item in data]
    paragraphs = [f"passage: {item['paragraph']}" for item in data]
    
    print(f"Total paragraphs for RAG: {len(paragraphs)}")
    
    # Create a FAISS index
    if not Path(path_fais).exists():
        # Embed
        embeddings = normalize(embedder.encode(paragraphs))
        
        # Store in FAISS
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Save your index and documents
        faiss.write_index(index, path_fais)
        print("Indexing is done!")
    else:
        print("Index documents have already been calculated.")
        
    if not Path(path_docsp).exists():   
        with open(path_docsp, "wb") as f:
            pickle.dump(documents, f)
    else:
        print("Pickle document have already been calculated.")

            
            
    

