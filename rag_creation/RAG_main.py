import dspy
import os
import argparse
import json
import time
import psutil
import datetime
import re
import pickle # Faltaba importar pickle
from pathlib import Path

# Intentar importar GPUtil para métricas de NVIDIA, si falla, se maneja el error
try:
    import GPUtil
    has_gpu_mon = True
except ImportError:
    has_gpu_mon = False

from rag_creation.utils import *
from rag_creation.signatures import *

from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN DE LOGS ---
LOG_FILE_PATH = "system_logs.json"

def get_system_metrics():
    """Captura el estado actual de CPU, RAM y GPU."""
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_usage_percent": psutil.cpu_percent(interval=None),
        "ram_usage_mb": psutil.virtual_memory().used / (1024 * 1024),
        "ram_total_mb": psutil.virtual_memory().total / (1024 * 1024),
    }
    
    if has_gpu_mon:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0] # Asumimos la primera GPU
                metrics.update({
                    "gpu_usage_percent": gpu.load * 100,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "temperature_gpu_c": gpu.temperature,
                    "device": gpu.name
                })
        except:
            metrics["gpu_error"] = "Could not retrieve GPU stats"
    
    return metrics

def estimate_tokens(text):
    """Estimación cruda de tokens (aprox 4 caracteres por token)."""
    if not text: return 0
    return len(text) // 4

def log_interaction(log_data):
    """Escribe el log en el archivo JSON."""
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        # Se escribe como JSON lines para no romper el archivo si el script falla
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

# --- FIN CONFIGURACIÓN DE LOGS ---

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

# Configuración del modelo
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

qa_generator = QAGenerator_safe()
path_qa = os.path.join(args.path_root, args.save_qa)

# Conteo inicial de archivos
containers = get_folders(args.path_root)
number_files = 0
for container in containers:
    pdf_files = get_pdfs(os.path.join(args.path_root, container, "pdf"))
    number_files += len(pdf_files)
print(f"Total number of PDF files found: {number_files}")

# --- PROCESAMIENTO PRINCIPAL ---
if not args.skip_savedata:

    if not Path(path_qa).exists() or args.flag_rewrite:    
        containers = get_folders(args.path_root)
        num_pdf = 0
        
        # Inicializar archivo de logs si no existe
        if not os.path.exists(LOG_FILE_PATH):
            print(f"Creating log file at {LOG_FILE_PATH}")

        for conti, container in enumerate(containers):
            # Extraer el sujeto del nombre de la carpeta (ej. Fisica, Quimica)
            match = re.match(r"[A-Za-z]+", container)
            subject = match.group() if match else "General"
        
            pdf_files = get_pdfs(os.path.join(args.path_root, container, "pdf"))
            
            for i, pdf in enumerate(pdf_files):
                num_pdf += 1
                
                if exists_qa(Path(pdf).stem, path_qa):
                    print(f"(PDF: {num_pdf}) Q&A for {pdf.stem} already exists. Skipping...")
                    continue
                
                print(f"Processing document: {pdf.stem}")
                
                text, tables = extract_text_from_pdf(pdf, remove_h=args.remove_headers)
                text = text.encode("utf-8", errors="replace").decode("utf-8")
                tables = drop_fuzzy_duplicate_tables(tables, similarity_threshold=0.92)
                    
                if text == '\n' or text == "" or not text.strip():
                    text, tables = extract_text_from_scanned_pdf_2(pdf)
                    
                paragraphs = split_by_paragraphs(text)
                paragraphs = split_long_paragraphs(paragraphs)
                paragraphs = [c for c in paragraphs if is_content_chunk(c)]
                    
                qas = []
                
                # --- LOGIC WRAPPER PARA LLM (Función auxiliar) ---
                def process_llm_call(entry_data, entry_type, index):
                    """Maneja la llamada al LLM y el registro de logs"""
                    
                    # 1. Captura métricas del sistema ANTES
                    sys_metrics = get_system_metrics()
                    start_time = time.time()
                    
                    # 2. Llamada al LLM
                    qa_pair = qa_generator(entry=entry_data, subject=subject)
                    
                    # 3. Calcular Latencia
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    if not qa_pair.is_valid:
                        return None

                    question = clean_llm_output(qa_pair.question)
                    answer = clean_llm_output(qa_pair.answer)
                    evidence = clean_llm_output(qa_pair.evidence)
                    
                    # 4. Calcular Tokens (Estimación)
                    input_tokens = estimate_tokens(str(entry_data))
                    output_tokens = estimate_tokens(question + answer + evidence)
                    
                    # 5. CONSTRUIR EL LOG UNIFICADO (Lo que pediste)
                    log_entry = {
                        "log_type": "generation_metric",
                        "timestamp": sys_metrics["timestamp"],
                        "request_id": f"{pdf.stem}_{index}_{entry_type}",
                        
                        # Datos del Modelo
                        "model_name": "mistral",
                        "latency_ms": round(latency_ms, 2),
                        "tokens_input_est": input_tokens,
                        "tokens_output_est": output_tokens,
                        "throughput_tps": round(output_tokens / (latency_ms/1000), 2) if latency_ms > 0 else 0,
                        
                        # Interacción (Simulada: Entrada -> Salida)
                        "input_snippet": str(entry_data)[:100] + "...", # Recortado para no llenar el log
                        "output_snippet": question[:100] + "...",
                        "context_length": len(str(entry_data)),
                        
                        # Datos del Sistema
                        "gpu_usage_percent": sys_metrics.get("gpu_usage_percent", 0),
                        "gpu_memory_mb": sys_metrics.get("gpu_memory_used_mb", 0),
                        "cpu_usage_percent": sys_metrics.get("cpu_usage_percent", 0),
                        "ram_usage_mb": round(sys_metrics.get("ram_usage_mb", 0), 2)
                    }
                    
                    # Guardar Log
                    log_interaction(log_entry)
                    
                    return {
                        "id": pdf.stem + f"_{index:05d}_{entry_type}", 
                        "source": "simple_LLM", 
                        "subject": subject,
                        "doc_name": pdf.stem, 
                        "paragraph": entry_data, 
                        "question": question, 
                        "answer": answer,
                        "evidence": evidence
                    }

                # Procesar Tablas
                if len(tables) > 0:
                    for ii, table in enumerate(tables):
                        if table:
                            res = process_llm_call(table, "table", ii)
                            if res: qas.append(res)

                # Procesar Párrafos
                for ii, prg in enumerate(paragraphs):
                    if prg:
                        res = process_llm_call(prg, "paragraph", ii)
                        if res: qas.append(res)
                        
                # Guardar Q&A
                with open(path_qa, "a", encoding="utf-8") as f:
                    for entry in qas:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"number of Q&As: {len(qas)} for PDF: {pdf.stem}")
                print(f"(PDF: {num_pdf}) Processed...")

    else:
        print("already processed Q&A dataset")        

# --- RAG MODULE (Sin cambios mayores, solo imports) ---

embedder = SentenceTransformer('intfloat/multilingual-e5-large')

path_fais = os.path.join(args.path_root, "university_index_clean.faiss")
path_colbert = os.path.join(args.path_root, "uis_corpus_clean.tsv")
path_docs = os.path.join(args.path_root, "university_docs_clean.pkl")

if Path(path_qa).exists():
    with open(path_qa) as f:
        data = [json.loads(line) for line in f]
        
    documents = [{"paragraph": item["paragraph"], "subject": item["subject"]} for item in data]
    paragraphs = [f"passage: {item['paragraph']}" for item in data]
    
    print(f"Total paragraphs for RAG: {len(paragraphs)}")
    
    if not Path(path_fais).exists():
        embeddings = normalize(embedder.encode(paragraphs))
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, path_fais)
        print("Indexing is done!")
    else:
        print("Index documents have already been calculated.")
        
    if not Path(path_docs).exists():   
        with open(path_docs, "wb") as f:
            pickle.dump(documents, f)
    else:
        print("Pickle document have already been calculated.")