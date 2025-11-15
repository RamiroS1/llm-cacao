import dspy
import os
import argparse
import json
from pathlib import Path

from rag_creation.utils import *
from rag_creation.signatures import *

from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import faiss

parser = argparse.ArgumentParser('extract text from .pdf files', add_help=False)
parser.add_argument('--path_root', default="./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads", type=str, help='path to root data folder')
parser.add_argument('--save_qa', default="dataQA.jsonl", type=str, help='file name to save Q&A dataset')
parser.add_argument('--remove_headers', default=True, type=bool, help='flag to remove header text from pdf')
parser.add_argument('--flag_rewrite', default=True, type=bool, help='flag to rewrite from exixting Q&A dataset')
parser.add_argument('--skip_savedata', default=False, type=bool, help='flag to rewrite from exixting Q&A dataset')
args = parser.parse_args()


def get_folders(path="."):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_pdfs(folder_path):
    return list(Path(folder_path).rglob("*.pdf"))

#lm = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='', temperature=0)

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

# Signatures
# Instantiate the Q&A generator and evaluators
# qa_generator = QAGenerator()
qa_generator = QAGenerator_safe()
# qa_generator = QAGenerator_v1()

# path_qa = os.path.join(args.path_root, "dataQA.jsonl")
path_qa = os.path.join(args.path_root, args.save_qa)

# number of filles found  
containers = get_folders(args.path_root)
number_files = 0
for container in containers:
    pdf_files = get_pdfs(os.path.join(args.path_root, container, "pdf"))
    number_files += len(pdf_files)
    print(f"Found {len(pdf_files)} PDF files in container '{container}'")
print(f"Total number of PDF files found: {number_files}")

# Extract Q&A samples
if not args.skip_savedata:

    if not Path(path_qa).exists() or args.flag_rewrite:    
        containers = get_folders(args.path_root)

        num_pdf = 0
        for conti, container in enumerate(containers):
            subject = re.match(r"[A-Za-z]+", container).group()
        
            pdf_files = get_pdfs(os.path.join(args.path_root, container, "pdf"))
            
            # Iterate through PDFs
            for i, pdf in enumerate(pdf_files):
                num_pdf += 1
                
                if exists_qa(Path(pdf).stem, path_qa):
                    print(f"(PDF: {num_pdf}) Q&A for {pdf.stem} already exists. Skipping...")
                    continue
                
                print(f"Processing document: {pdf.stem}")
                
                text, tables = extract_text_from_pdf(pdf, remove_h=args.remove_headers)
                text = text.encode("utf-8", errors="replace").decode("utf-8")
                tables = drop_fuzzy_duplicate_tables(tables, similarity_threshold=0.92)
                    
                # scanned pdf
                if text == '\n' or text == "" or not text.strip():
                    text, tables = extract_text_from_scanned_pdf_2(pdf)
                    
                # Extract paragraphs    
                paragraphs = split_by_paragraphs(text)
                paragraphs = split_long_paragraphs(paragraphs)
                paragraphs = [c for c in paragraphs if is_content_chunk(c)]
                    
                # Generate Q&As
                qas = []
                    
                # Q&A from tables
                if len(tables) > 0:
                    for ii, table in enumerate(tables):
                        # Generate Q&A using zero-shot LLM
                        if table:
                            qa_pair = qa_generator(entry=table, subject=subject)
                                
                        if not qa_pair.is_valid:
                            continue
                            
                        question = clean_llm_output(qa_pair.question)
                        answer = clean_llm_output(qa_pair.answer)
                        evidence = clean_llm_output(qa_pair.evidence)
                            
                        dic_qa = {"id": pdf.stem + f"_{ii:05d}_table", "source": "simple_LLM", "subject": subject,
                                "doc_name": pdf.stem, "paragraph": tables[ii], "question": question, "answer": answer,
                                "evidence": evidence}
                            
                        qas.append(dic_qa)

                # Q&A from paragraphs
                for ii, prg in enumerate(paragraphs):
                    
                    # print("oooooooo", prg)
                    
                    if prg:
                        qa_pair = qa_generator(entry=prg, subject=subject)
                            
                    if not qa_pair.is_valid:
                        continue
                        
                    question = clean_llm_output(qa_pair.question)
                    answer = clean_llm_output(qa_pair.answer)
                    evidence = clean_llm_output(qa_pair.evidence)
                        
                    dic_qa = {"id": pdf.stem + f"_{ii:05d}", "source": "simple_LLM", "subject": subject, "doc_name": pdf.stem,
                            "paragraph": prg, "question": question, "answer": answer, "evidence": qa_pair.evidence}     
                        
                    qas.append(dic_qa)
                        
                # save Q&A_UIS full dataset
                with open(path_qa, "a", encoding="utf-8") as f:
                    for entry in qas:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"number of Q&As: {len(qas)} for PDF: {pdf.stem}")
                print(f"(PDF: {num_pdf}) Processed...")

    else:
        print("already processed Q&A dataset")        

# RAG module

embedder = SentenceTransformer('intfloat/multilingual-e5-large')

path_fais = os.path.join(args.path_root, "university_index_clean.faiss")
path_colbert = os.path.join(args.path_root, "uis_corpus_clean.tsv")
path_docs = os.path.join(args.path_root, "university_docs_clean.pkl")

if Path(path_qa).exists():
    
    # Load Q&As from the JSON file 
    with open(path_qa) as f:
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
        
    if not Path(path_docs).exists():   
        with open(path_docs, "wb") as f:
            pickle.dump(documents, f)
    else:
        print("Pickle document have already been calculated.")

    


