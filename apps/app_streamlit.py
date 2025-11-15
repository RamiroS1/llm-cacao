"""
Sistema Multi-RAG con Streamlit
"""

import os
import streamlit as st
import dspy
from rag_creation.utils import RerankedFaissRetriever, UniversityRAGChain, faster_UniversityRAGChain, clean_output
from datetime import datetime
import traceback
import PIL
import requests
import torch
import gc

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
icon_path = "assets/isologo_agrosavia.png"
page_icon_obj = "🌱"

if os.path.exists(icon_path):
    try:
        page_icon_obj = PIL.Image.open(icon_path)
    except Exception:
        page_icon_obj = "🌱"

st.set_page_config(
    page_title="Asistente IA para cacao",
    page_icon=page_icon_obj,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS MODERNO Y MINIMALISTA
# ============================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5e9 0%, #f1f8f4 100%);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding-top: 16px;
    }
    
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
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
# CONFIGURACIÓN OPTIMIZADA DE RAGs
# ============================================
RAG_CONFIG = {
    "🌱 Dr. agro Específico": {
        "path_root": "./llm_cacao-dragro",
        "index_name": "profiles_index.faiss",
        "docs_name": "profiles_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": False,
        "llm_config": {
            "num_ctx": 8192,  # Contexto grande para Dr. agro
            "num_predict": 256,
            "num_gpu": -1
        },
        "retriever_config": None,  # Sin config especial
        "description": "Información especializada de enfermedades y deficiencias"
    },
    "⚡ Configurado (Óptimo)": {
        "path_root": "./data",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": True,
        "top_n": 4,
        "llm_config": {
            "num_ctx": 3072,  # Contexto reducido para velocidad
            "num_predict": 160,  # Límite estricto de salida
            "top_k": 30,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_thread": os.cpu_count(),
            "num_batch": 1024,
            "seed": 0,
            "num_gpu": -1,
            "keep_alive": "30m",
            "stop": ["\n}\n", "\n\n", "</s>"]
        },
        "retriever_config": {
            "top_k": 20,
            "k_rerank": 8,
            "use_reranker": False
        },
        "description": "Versión optimizada para respuestas rápidas"
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

if "lm_initialized" not in st.session_state:
    st.session_state.lm_initialized = False

if "current_llm_config" not in st.session_state:
    st.session_state.current_llm_config = None

# ============================================
# FUNCIONES PRINCIPALES
# ============================================

def liberar_memoria_cuda():
    """Libera memoria CUDA y ejecuta garbage collection"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        st.warning(f"Advertencia al liberar memoria: {str(e)}")

def initialize_llm(config_name="default"):
    """Inicializa el modelo de lenguaje con configuración específica"""
    try:
        # Verificar si Ollama está disponible
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code != 200:
            return None, False, "❌ Ollama no responde"
        
        # Obtener configuración del RAG si existe
        llm_config = {}
        if config_name in RAG_CONFIG:
            llm_config = RAG_CONFIG[config_name]["llm_config"]
        else:
            # Config por defecto (Dr. agro)
            llm_config = {
                "num_ctx": 8192,
                "format": "json"
            }
        
        lm = dspy.LM(
            'ollama_chat/mistral',
            api_base='http://localhost:11434',
            api_key='',
            temperature=0,
            model_kwargs={
                "format": "json",
                "options": llm_config
            }
        )
        dspy.configure(lm=lm)
        return lm, True, f"✅ Conectado ({config_name})"
    except requests.exceptions.RequestException:
        return None, False, "❌ Ollama no está ejecutándose"
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

def verificar_rag_disponible(rag_name):
    """Verifica si los archivos necesarios del RAG existen"""
    config = RAG_CONFIG[rag_name]
    path_faiss = os.path.join(config["path_root"], config["index_name"])
    path_docs = os.path.join(config["path_root"], config["docs_name"])
    
    errores = []
    if not os.path.exists(path_faiss):
        errores.append(f"❌ Falta: {path_faiss}")
    if not os.path.exists(path_docs):
        errores.append(f"❌ Falta: {path_docs}")
    
    return len(errores) == 0, errores

@st.cache_resource
def load_rag(rag_name):
    """Carga un RAG con configuración optimizada específica"""
    try:
        # Liberar memoria antes de cargar
        liberar_memoria_cuda()
        
        # Verificar disponibilidad
        disponible, errores = verificar_rag_disponible(rag_name)
        if not disponible:
            st.error(f"Archivos faltantes para {rag_name}:")
            for error in errores:
                st.error(error)
            return None, False
        
        config = RAG_CONFIG[rag_name]
        path_root = config["path_root"]
        path_faiss = os.path.join(path_root, config["index_name"])
        path_docs = os.path.join(path_root, config["docs_name"])
        
        # Crear retriever con configuración específica
        retriever = RerankedFaissRetriever(
            path_faiss, 
            path_docs, 
            model_match=config["model_match"]
        )
        
        # Aplicar configuración de retriever si existe
        if config.get("retriever_config"):
            retriever_conf = config["retriever_config"]
            if hasattr(retriever, "top_k"):
                retriever.top_k = retriever_conf["top_k"]
            if hasattr(retriever, "k_rerank"):
                retriever.k_rerank = retriever_conf["k_rerank"]
            if hasattr(retriever, "use_reranker"):
                retriever.use_reranker = retriever_conf["use_reranker"]
        
        # Crear chain según configuración
        if config["use_faster"]:
            chain = faster_UniversityRAGChain(
                retriever=retriever,
                model_match=config["model_match"],
                top_n=config.get("top_n", 4)
            )
        else:
            chain = UniversityRAGChain(
                retriever=retriever,
                model_match=config["model_match"]
            )
        
        return chain, True
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            st.error("❌ **Memoria GPU insuficiente**")
            st.info("""
            **Posibles soluciones:**
            1. Cierra otros procesos que usen GPU
            2. Reinicia la aplicación
            3. Ejecuta: `nvidia-smi` para ver procesos activos
            """)
        return None, False
    except Exception as e:
        st.error(f"Error cargando RAG {rag_name}: {str(e)}")
        return None, False

def consultar_rag(message):
    """Procesa una consulta con manejo robusto de errores"""
    # Validaciones
    if not st.session_state.lm:
        return "❌ **Error**: Modelo no inicializado. Por favor reinicia la aplicación."
    
    if not st.session_state.current_rag:
        return "❌ **Error**: Selecciona un RAG primero."
    
    chain = st.session_state.chains.get(st.session_state.current_rag)
    if not chain:
        return f"❌ **Error**: RAG '{st.session_state.current_rag}' no disponible."
    
    try:
        response, context = chain(message, ext_context=st.session_state.system_prompt)
        respuesta = clean_output(response.get("answer", "Sin respuesta"))
        
        return respuesta
    
    except KeyError as e:
        return f"❌ **Error en respuesta**: Clave faltante - {str(e)}"
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower():
            return "❌ **Error de conexión**: El modelo no responde. Verifica que Ollama esté ejecutándose."
        return f"❌ **Error**: {error_msg}"

# ============================================
# INICIALIZACIÓN AUTOMÁTICA DEL LLM
# ============================================
if not st.session_state.lm_initialized:
    with st.spinner("Inicializando modelo..."):
        # Inicializar con config por defecto
        lm, success, message = initialize_llm("default")
        st.session_state.lm = lm
        st.session_state.lm_initialized = True
        
        if not success:
            st.error(f"{message}. Por favor, asegúrate de que Ollama esté ejecutándose.")
            st.info("Para iniciar Ollama: `ollama serve`")

# ============================================
# SIDEBAR OPTIMIZADO
# ============================================
with st.sidebar:
    st.title("🌱 Dr. agro")
    st.caption("Asistente IA para cacao")
    st.divider()
    
    # Estado de conexión del modelo
    if st.session_state.lm:
        st.success("🟢 Modelo conectado", icon="🤖")
    else:
        st.error("🔴 Modelo no disponible", icon="⚠️")
        if st.button("🔄 Reintentar conexión"):
            st.session_state.lm_initialized = False
            st.rerun()
    
    st.divider()
    
    # Selector de RAG con validación
    st.subheader("📂 Base de Conocimiento")
    
    # Verificar RAGs disponibles
    rags_validos = {}
    for nombre, config in RAG_CONFIG.items():
        disponible, _ = verificar_rag_disponible(nombre)
        if disponible:
            rags_validos[nombre] = config
    
    if not rags_validos:
        st.error("❌ No hay RAGs disponibles")
        st.info("Verifica que existan los archivos .faiss y .pkl en las carpetas configuradas")
    else:
        rag_seleccionado = st.selectbox(
            "Selecciona:",
            options=list(rags_validos.keys()),
            label_visibility="collapsed",
            key="rag_selector"
        )
        
        # Cargar RAG si cambió
        if rag_seleccionado != st.session_state.current_rag:
            with st.spinner(f"Cargando {rag_seleccionado}..."):
                # Liberar RAG anterior si existe
                if st.session_state.current_rag and st.session_state.current_rag in st.session_state.chains:
                    del st.session_state.chains[st.session_state.current_rag]
                    liberar_memoria_cuda()
                
                # Reconfigurar LLM con la config del RAG seleccionado
                if st.session_state.current_llm_config != rag_seleccionado:
                    lm, success, message = initialize_llm(rag_seleccionado)
                    if success:
                        st.session_state.lm = lm
                        st.session_state.current_llm_config = rag_seleccionado
                
                # Cargar nuevo RAG
                if rag_seleccionado not in st.session_state.chains:
                    chain, success = load_rag(rag_seleccionado)
                    if success:
                        st.session_state.chains[rag_seleccionado] = chain
                        st.session_state.current_rag = rag_seleccionado
                        st.success(f"✅ {rag_seleccionado} cargado")
                    else:
                        st.error(f"❌ Error al cargar {rag_seleccionado}")
                else:
                    st.session_state.current_rag = rag_seleccionado
                st.rerun()
        
        # Indicador visual mejorado
        if st.session_state.current_rag:
            config_actual = RAG_CONFIG[st.session_state.current_rag]
            
            st.success(f"**{st.session_state.current_rag}**", icon="✅")
            
            # Mostrar configuración del RAG actual
            with st.expander("ℹ️ Detalles"):
                st.caption(config_actual["description"])
                st.text(f"Contexto: {config_actual['llm_config'].get('num_ctx', 'N/A')}")
                st.text(f"Max tokens: {config_actual['llm_config'].get('num_predict', 'N/A')}")
                if config_actual["use_faster"]:
                    st.text(f"Top N: {config_actual.get('top_n', 'N/A')}")
                st.text(f"Modo: {'⚡ Rápido' if config_actual['use_faster'] else '📚 Completo'}")
    
    st.divider()
    
    # Estadísticas de sesión
    if st.session_state.messages:
        st.metric("💬 Conversación", f"{len(st.session_state.messages) // 2} mensajes")
    
    st.divider()
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpiar", use_container_width=True, help="Limpia el historial de chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("♻️ Reset", use_container_width=True, help="Reinicia toda la aplicación"):
            # Liberar recursos antes de reset
            if st.session_state.chains:
                st.session_state.chains.clear()
            liberar_memoria_cuda()
            st.session_state.clear()
            st.rerun()

# ============================================
# ÁREA PRINCIPAL - CHAT
# ============================================

# Mensaje de bienvenida
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant", avatar="🌱"):
        st.markdown("👋 Selecciona un RAG y hazme tu consulta sobre cacao.")

# Mostrar historial
for message in st.session_state.messages:
    avatar = "🧑‍🌾" if message["role"] == "user" else "🌱"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Ejemplo: ¿Cuáles son las zonas productoras de cacao?"):
    # Validar que el sistema esté listo
    if not st.session_state.lm:
        st.error("⚠️ El modelo no está inicializado. Por favor reinicia la aplicación.")
        st.stop()
    
    if not st.session_state.current_rag:
        st.warning("⚠️ Por favor selecciona un RAG antes de consultar.")
        st.stop()
    
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