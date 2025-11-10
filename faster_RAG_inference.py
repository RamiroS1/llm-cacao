import os
import argparse
import dspy

from utils import RerankedFaissRetriever, UniversityRAGChain, clean_output

# ¿Podrías explicarme qué es el cultivo agroforestal?
# Quisiera saber más sobre la cadena de cacao
# ¿Cuáles son las principales zonas productoras de cacao en Colombia?

parser = argparse.ArgumentParser('Run RAG-inference', add_help=False)
parser.add_argument('--path_root', default="./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads", type=str, help='path to root data folder')
parser.add_argument('--question', default="¿Cuáles son las principales zonas productoras de cacao en Colombia?", type=str, help='user question')
args = parser.parse_args()


lm = dspy.LM(
    'ollama_chat/mistral',                    # keep your model
    api_base='http://localhost:11434',
    api_key='',
    temperature=0,
    model_kwargs={
        "format": "json",
        "options": {
            # ↓ decoding / output size
            "num_ctx": 4096,                 # smaller context = faster KV ops
            "num_predict": 256,              # HARD cap on output tokens (biggest win)
            "top_k": 30, "top_p": 0.9,       # stable decoding
            "repeat_penalty": 1.05,

            # ↓ runtime throughput (tune to your box)
            "num_thread": os.cpu_count(),    # CPU threads if you’re CPU-bound
            "num_batch": 512,                # larger = fewer passes (needs VRAM)
            "seed": 0,

            # ↓ GPU offload (if you have a GPU with enough VRAM)
            # -1 = offload as many layers as possible
            "num_gpu": -1
        }
    }
)
dspy.configure(lm=lm)


path_fais = os.path.join(args.path_root, "university_index.faiss")
path_docs = os.path.join(args.path_root, "university_docs.pkl")

name_model_match = 'intfloat/multilingual-e5-large'
# name_model_match = 'sentence-transformers/all-mpnet-base-v2'
# name_model_match = 'intfloat/multilingual-e5-base'
retriever = RerankedFaissRetriever(path_fais, path_docs, model_match=name_model_match)

if hasattr(retriever, "top_k"):
    retriever.top_k = 20

if hasattr(retriever, "k_rerank"):
    retriever.k_rerank = 8

if hasattr(retriever, "use_reranker"):
    retriever.use_reranker = False


chain = UniversityRAGChain(retriever=retriever, model_match=name_model_match)

user_question = args.question

response, context = chain(user_question, ext_context="")
print(f"PREGUNTA: {user_question} \n\n RESPUESTA {clean_output(response["answer"])}")