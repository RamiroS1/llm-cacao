import dspy
import os
import argparse
import json
import time
import psutil
import datetime
import re
import pickle
from pathlib import Path

# Intentar importar GPUtil para métricas de GPU
try:
    import GPUtil
    has_gpu_mon = True
except ImportError:
    has_gpu_mon = False

from rag_creation.utils import *
# Aseguramos que las librerías necesarias estén importadas
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN DE LOGS ---
LOG_FILE_PATH = "system_logs_docs.json"

def get_system_metrics():
    """Captura el estado actual de CPU, RAM y GPU."""
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_usage_percent": psutil.cpu_percent(interval=None),
        "ram_usage_mb": psutil.virtual_memory().used / (1024 * 1024),
    }
    
    if has_gpu_mon:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    "gpu_usage_percent": gpu.load * 100,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "temperature_gpu_c": gpu.temperature,
                    "device": gpu.name
                })
        except:
            metrics["gpu_error"] = "Could not retrieve GPU stats"
    
    return metrics

def log_interaction(log_data):
    """Escribe el log en el archivo JSON."""
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
# --- FIN CONFIGURACIÓN DE LOGS ---

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

# Configuración del modelo (Se mantiene aunque no se use en el loop de extracción actual)
lm = dspy.LM('ollama_chat/mistral',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0,
    model_kwargs={
        "format": "json",
        "options": {"num_ctx": 8192}
    }
)
dspy.configure(lm=lm)

path_docs = os.path.join(args.path_root, "dataDocs_small.jsonl")

# Extract Q&A samples
if not args.skip_savedata:
    if not Path(path_docs).exists() or args.flag_rewrite: 
        containers = get_folders(args.path_root)
        
        # Inicializar log si no existe
        if not os.path.exists(LOG_FILE_PATH):
            print(f"Creating log file at {LOG_FILE_PATH}")
        
        for conti, container in enumerate(containers):
            match = re.match(r"[A-Za-z]+", container)
            subject = match.group() if match else "General"
            
            docs_files = get_docs(os.path.join(args.path_root, container))
            
            for i, doc in enumerate(docs_files):
                print("Processing document:", doc)
                
                # --- MONITOREO DE EXTRACCIÓN (CPU Heavy) ---
                start_time = time.time()
                sys_metrics_start = get_system_metrics()
                
                # Extract text
                text, tables = word_to_markdown(doc)
                
                # Extract paragraphs    
                paragraphs = split_by_paragraphs(text)
                paragraphs = split_long_paragraphs(paragraphs)
                paragraphs = [c for c in paragraphs if is_content_chunk(c)]
                
                docs = []
                # Procesar tablas
                if len(tables) > 0:
                    for ii, table in enumerate(tables):
                        dic_doc = {"id": doc.stem + f"_{ii:05d}_table", "source": "simple_LLM", "subject": subject,
                                "doc_name": doc.stem, "paragraph": table}
                        docs.append(dic_doc)
            
                # Procesar párrafos
                for ii, prg in enumerate(paragraphs):
                    dic_doc = {"id": doc.stem + f"_{ii:05d}", "source": "simple_LLM", "subject": subject, "doc_name": doc.stem,
                            "paragraph": prg}
                    docs.append(dic_doc)
                
                # Métricas post-procesamiento
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # LOGGING DE EXTRACCIÓN
                log_entry = {
                    "log_type": "extraction_metric",
                    "timestamp": sys_metrics_start["timestamp"],
                    "request_id": f"extract_{doc.stem}",
                    "doc_name": doc.stem,
                    
                    # Métricas de Proceso
                    "latency_ms": round(latency_ms, 2),
                    "items_extracted": len(docs), # Cuántos chunks salieron
                    "char_count_total": len(text),
                    
                    # Estado del Sistema
                    "cpu_usage_percent": sys_metrics_start["cpu_usage_percent"],
                    "ram_usage_mb": round(sys_metrics_start["ram_usage_mb"], 2)
                }
                log_interaction(log_entry)
                    
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
        print("Starting Embedding process (GPU intensive)...")
        
        # --- MONITOREO DE EMBEDDING (GPU Heavy) ---
        sys_metrics_emb = get_system_metrics()
        start_emb = time.time()
        
        # Embed
        embeddings = normalize(embedder.encode(paragraphs))
        
        end_emb = time.time()
        latency_emb = (end_emb - start_emb) * 1000
        
        # LOGGING DE EMBEDDING
        log_emb = {
            "log_type": "embedding_batch_metric",
            "timestamp": sys_metrics_emb["timestamp"],
            "model_name": "multilingual-e5-large",
            
            # Rendimiento
            "latency_total_ms": round(latency_emb, 2),
            "total_paragraphs": len(paragraphs),
            "ms_per_paragraph": round(latency_emb / len(paragraphs), 2) if len(paragraphs) > 0 else 0,
            
            # Hardware (Crítico aquí)
            "gpu_usage_percent": sys_metrics_emb.get("gpu_usage_percent", 0),
            "gpu_memory_mb": sys_metrics_emb.get("gpu_memory_used_mb", 0),
            "temperature_gpu_c": sys_metrics_emb.get("temperature_gpu_c", 0)
        }
        log_interaction(log_emb)
        # ------------------------------------------
        
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