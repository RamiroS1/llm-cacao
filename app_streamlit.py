"""
Dr. agro - Sistema Multi-RAG con Streamlit (Minimalista y Rápido)
"""

import os
import streamlit as st
import dspy
from utils import RerankedFaissRetriever, UniversityRAGChain, faster_UniversityRAGChain, clean_output
from datetime import datetime
import traceback

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(
    page_title="Dr. agro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS MODERNO Y MINIMALISTA
# ============================================
st.markdown("""
<style>
    /* Tema general limpio */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar verde minimalista */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5e9 0%, #f1f8f4 100%);
    }
    
    /* Ocultar elementos innecesarios */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat messages */
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Input box */
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding-top: 16px;
    }
    
    /* Botones */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    /* Headers limpios */
    h1 {
        color: #2e7d32;
        font-weight: 700;
    }
    
    h3 {
        color: #43a047;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURACIÓN DE RAGs
# ============================================
RAG_CONFIG = {
    "📚 BAC (Cacao)": {
        "path_root": "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": False,
        "top_n": 10
    },
    "🌱 Dr. agro Específico": {
        "path_root": "./llm_cacao-dragro",
        "index_name": "profiles_index.faiss",
        "docs_name": "profiles_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": False,
        "top_n": 10
    },
    "⚡ Configurado (Rápido)": {
        "path_root": "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": True,
        "top_n": 4
    }
}

# ============================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "lm" not in st.session_state:
    st.session_state.lm = None

if "chains" not in st.session_state:
    st.session_state.chains = {}

if "current_rag" not in st.session_state:
    st.session_state.current_rag = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

# ============================================
# FUNCIONES PRINCIPALES
# ============================================
@st.cache_resource
def initialize_llm():
    """Inicializa el modelo de lenguaje (solo una vez)"""
    try:
        lm = dspy.LM(
            'ollama_chat/mistral',
            api_base='http://localhost:11434',
            api_key='',
            temperature=0,
            model_kwargs={
                "format": "json",
                "options": {
                    "num_ctx": 3072,
                    "num_predict": 160,
                    "top_k": 30,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "num_thread": os.cpu_count(),
                    "num_batch": 1024,
                    "seed": 0,
                    "num_gpu": -1,
                    "keep_alive": "30m",
                    "stop": ["\n}\n", "\n\n", "</s>"]
                }
            }
        )
        dspy.configure(lm=lm)
        return lm, True, "✅ Conectado"
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

@st.cache_resource
def load_rag(rag_name):
    """Carga un RAG (con caché)"""
    try:
        config = RAG_CONFIG[rag_name]
        path_root = config["path_root"]
        path_faiss = os.path.join(path_root, config["index_name"])
        path_docs = os.path.join(path_root, config["docs_name"])
        
        if not os.path.exists(path_faiss) or not os.path.exists(path_docs):
            return None, False
        
        retriever = RerankedFaissRetriever(
            path_faiss, 
            path_docs, 
            model_match=config["model_match"]
        )
        
        if config["use_faster"]:
            chain = faster_UniversityRAGChain(
                retriever=retriever,
                model_match=config["model_match"],
                top_n=config["top_n"]
            )
        else:
            chain = UniversityRAGChain(
                retriever=retriever,
                model_match=config["model_match"]
            )
        
        return chain, True
    
    except Exception as e:
        st.error(f"Error cargando RAG: {str(e)}")
        return None, False

def consultar_rag(message):
    """Procesa una consulta"""
    if not st.session_state.lm:
        return "❌ Modelo no inicializado"
    
    if not st.session_state.current_rag:
        return "❌ Selecciona un RAG primero"
    
    chain = st.session_state.chains.get(st.session_state.current_rag)
    if not chain:
        return "❌ RAG no disponible"
    
    try:
        response, context = chain(message, ext_context=st.session_state.system_prompt)
        respuesta = clean_output(response["answer"])
        
        timestamp = datetime.now().strftime("%H:%M")
        return f"{respuesta}\n\n*{timestamp} • {st.session_state.current_rag}*"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🌱 Dr. agro")
    st.caption("Sistema Multi-RAG para Cacao")
    
    # Inicializar LLM
    if st.session_state.lm is None:
        with st.spinner("Inicializando modelo..."):
            lm, success, msg = initialize_llm()
            if success:
                st.session_state.lm = lm
                st.success(msg, icon="✅")
            else:
                st.error(msg, icon="❌")
    else:
        st.success("✅ Modelo listo", icon="🤖")
    
    st.divider()
    
    # Selector de RAG
    st.subheader("📂 RAG Activo")
    rag_seleccionado = st.selectbox(
        "Selecciona una base de conocimiento:",
        options=list(RAG_CONFIG.keys()),
        label_visibility="collapsed"
    )
    
    # Cargar RAG si cambió
    if rag_seleccionado != st.session_state.current_rag:
        with st.spinner(f"Cargando {rag_seleccionado}..."):
            if rag_seleccionado not in st.session_state.chains:
                chain, success = load_rag(rag_seleccionado)
                if success:
                    st.session_state.chains[rag_seleccionado] = chain
            
            st.session_state.current_rag = rag_seleccionado
            st.rerun()
    
    # Indicador visual
    if st.session_state.current_rag:
        st.info(f"🟢 *{st.session_state.current_rag}*", icon="📊")
    
    st.divider()
    
    # System Prompt (colapsable)
    with st.expander("🧠 System Prompt", expanded=False):
        prompt = st.text_area(
            "Instrucciones para el modelo:",
            value=st.session_state.system_prompt,
            height=100,
            placeholder="Ej: Eres un experto en agronomía...",
            label_visibility="collapsed"
        )
        if st.button("💾 Guardar", use_container_width=True):
            st.session_state.system_prompt = prompt
            st.success("Guardado", icon="✅")
    
    st.divider()
    
    # Botón de limpiar
    if st.button("🗑️ Limpiar Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Info compacta
    with st.expander("ℹ️ Info", expanded=False):
        st.caption(f"""
        *RAGs cargados:* {len(st.session_state.chains)}/3  
        *Modelo:* Mistral (Ollama)  
        *Estado:* {'🟢 Activo' if st.session_state.lm else '🔴 Inactivo'}
        """)

# ============================================
# ÁREA PRINCIPAL - CHAT
# ============================================
st.title("💬 Chat")

# Mensaje de bienvenida
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant", avatar="🌱"):
        st.markdown("👋 ¡Hola! Soy *Dr. agro*. Selecciona un RAG y hazme tu consulta sobre cacao.")

# Mostrar historial
for message in st.session_state.messages:
    avatar = "🧑‍🌾" if message["role"] == "user" else "🌱"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Ejemplo: ¿Cuáles son las zonas productoras de cacao?"):
    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner("Pensando..."):
            response = consultar_rag(prompt)
            st.markdown(response)
    
    # Guardar respuesta
    st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================
# FOOTER MINIMALISTA
# ============================================
st.markdown("---")
st.caption("🌱 Dr. agro • Sistema Multi-RAG • Powered by Streamlit + DSPy")