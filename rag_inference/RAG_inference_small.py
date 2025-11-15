import os
import argparse
import dspy

from rag_creation.utils import RerankedFaissRetriever, UniversityRAGChain, clean_output

parser = argparse.ArgumentParser('Run RAG-inference-small', add_help=False)
parser.add_argument('--path_root', default="./llm_cacao-dragro", type=str, help='path to root data folder')
parser.add_argument('--question', default="¿Cuáles son las principales zonas productoras de cacao en Colombia?", type=str, help='user question')
args = parser.parse_args()

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


path_fais = os.path.join(args.path_root, "profiles_index.faiss")
path_docs = os.path.join(args.path_root, "profiles_docs.pkl")

retriever = RerankedFaissRetriever(path_fais, path_docs)
chain = UniversityRAGChain(retriever=retriever)

user_question = args.question

response, context = chain(user_question, ext_context="")
print(f"PREGUNTA: {user_question} \n\n RESPUESTA {clean_output(response["answer"])}")