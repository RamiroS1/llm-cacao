"""
Dr. agro - Sistema Multi-RAG y Multimodal Integrado
Chat unificado estilo ChatGPT con soporte para texto, imágenes y videos
"""
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import traceback
import gc
import tempfile
from pathlib import Path
import re

# ============================================
# GESTIÓN DE RUTAS (PATH)
# ============================================
# Usamos Pathlib para hacerlo compatible con Windows/Linux/Mac automáticamente
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent / 'models_images'))
sys.path.append(str(current_dir.parent / 'models_video'))

import streamlit as st
import dspy
import requests
import torch
import PIL
from PIL import Image
from datetime import datetime

# Importaciones del RAG (Asegúrate de que estas rutas existan en tu proyecto)
from rag_creation.utils import RerankedFaissRetriever, UniversityRAGChain, faster_UniversityRAGChain, clean_output

# ============================================
# IMPORTACIÓN SEGURA DE MODELOS
# ============================================
try:
    from Llava_LDD import LlavaPlantDiseaseDetector
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    print("⚠️ Módulo LLaVA no encontrado o con errores.")

try:
    from VideoLlama3 import VideoLlamaAgriculturalAnalyzer
    VIDEOLLAMA_AVAILABLE = True
except ImportError:
    VIDEOLLAMA_AVAILABLE = False
    print("⚠️ Módulo VideoLLaMA3 no encontrado.")

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
    page_title="Dr. agro - Asistente IA",
    page_icon=page_icon_obj,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS MODERNO
# ============================================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #e8f5e9 0%, #f1f8f4 100%); }
    .stChatMessage { background: white; border-radius: 12px; padding: 16px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    h1 { color: #2e7d32; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURACIÓN DE RAGs
# ============================================
RAG_CONFIG = {
    "🌱 Dr. agro Específico": {
        "path_root": "../llm_cacao-dragro",
        "index_name": "profiles_index.faiss",
        "docs_name": "profiles_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": False,
        "llm_config": {
            "num_ctx": 4096,
            "num_predict": 256,
            "num_gpu": 1,
            "keep_alive": "10m"
        },
        "description": "Información especializada de enfermedades y deficiencias"
    },
    "⚡ Configurado (Óptimo)": {
        "path_root": "../data",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": True,
        "top_n": 4,
        "llm_config": {
            "num_ctx": 2048,
            "num_predict": 128,
            "top_k": 30,
            "top_p": 0.9,
            "num_gpu": 1,
            "keep_alive": "10m"
        },
        "description": "Versión optimizada para respuestas rápidas"
    }
}

# ============================================
# GESTIÓN DE ESTADO (SESSION STATE)
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_rag" not in st.session_state:
    st.session_state.current_rag = None
if "chains" not in st.session_state:
    st.session_state.chains = {}
if "llava_loaded" not in st.session_state:
    st.session_state.llava_loaded = False
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "llava_detector" not in st.session_state:
    st.session_state.llava_detector = None
if "video_analyzer" not in st.session_state:
    st.session_state.video_analyzer = None
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_video" not in st.session_state:
    st.session_state.pending_video = None
if "pending_video_path" not in st.session_state:
    st.session_state.pending_video_path = None

# ============================================
# FUNCIONES CRUDAS (MEMORIA Y VERIFICACIÓN)
# ============================================

def liberar_memoria_cuda():
    """Libera memoria CUDA de forma segura"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass

def liberar_modelo(model_name):
    """Descarga un modelo específico de la GPU"""
    try:
        if model_name == "llava" and st.session_state.llava_detector:
            if hasattr(st.session_state.llava_detector, 'unload_model'):
                st.session_state.llava_detector.unload_model()
            st.session_state.llava_detector = None
            st.session_state.llava_loaded = False
            
        elif model_name == "video" and st.session_state.video_analyzer:
            if hasattr(st.session_state.video_analyzer, 'unload_model'):
                st.session_state.video_analyzer.unload_model()
            st.session_state.video_analyzer = None
            st.session_state.video_loaded = False
        
        liberar_memoria_cuda()
        return True
    except Exception as e:
        st.error(f"Error al liberar {model_name}: {str(e)}")
        return False

def verificar_ollama():
    """Verifica si Ollama está corriendo"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return response.status_code == 200, "✅ Ollama conectado"
    except requests.exceptions.ConnectionError:
        return False, "❌ Ollama no está corriendo"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# ============================================
# FUNCIONES CORE (DSPy + RAG + MODELOS)
# ============================================

# 🔥 SOLUCIÓN AL BUG DE DSPY: @st.cache_resource 🔥
# Esto asegura que el objeto LM se cree UNA sola vez por sesión de servidor
# y no intente reconfigurar DSPy en cada recarga de la página.
@st.cache_resource(show_spinner=False)
def get_ollama_lm(config_name="default"):
    """
    Crea y configura la instancia de DSPy LM.
    Cacheado para evitar errores de hilos.
    """
    llm_config = {}
    if config_name in RAG_CONFIG:
        llm_config = RAG_CONFIG[config_name]["llm_config"]
    else:
        llm_config = {"num_ctx": 2048, "num_predict": 128}

    lm = dspy.LM(
        'ollama_chat/mistral',
        api_base='http://localhost:11434',
        api_key='',
        temperature=0,
        model_kwargs={"format": "json", "options": llm_config}
    )
    
    # Intentamos configurar DSPy. Si falla por hilos, lo ignoramos porque significa que ya está configurado.
    try:
        dspy.configure(lm=lm)
    except Exception:
        pass # Ignorar error si ya estaba configurado por otro hilo
        
    return lm

def verificar_intencion_chat(text):
    """
    Filtro rápido para preguntas comunes que no requieren RAG.
    Retorna la respuesta (str) o None si debe pasar al RAG.
    """
    if not text: return None
    
    # Limpieza básica
    t = text.lower().strip()
    # Eliminamos signos de puntuación para facilitar la coincidencia
    t = re.sub(r'[^\w\s]', '', t)
    
    # 1. IDENTIDAD / CREADOR
    # Preguntas sobre quién es o quién lo hizo
    triggers_identidad = ['quien eres', 'que eres', 'como te llamas']
    triggers_creador = ['quien te hizo', 'quien te creo', 'quien te desarrollo', 'quienes son tus creadores']
    
    if any(x in t for x in triggers_creador):
        return (
            "🛠️ Fui desarrollado por el equipo de investigación de **AGROSAVIA** "
            "(Corporación colombiana de investigación agropecuaria), diseñado para apoyar "
            "a productores y técnicos en el diagnóstico y manejo de cultivos."
        )
    
    if any(x in t for x in triggers_identidad):
        return (
            "🤖 Soy **Dr. agro**, un asistente virtual basado en Inteligencia Artificial. "
            "Estoy entrenado para responder preguntas sobre agricultura, identificar enfermedades "
            "en imágenes y analizar videos de cultivos."
        )

    # 2. SALUDOS Y DESPEDIDAS
    # Solo respondemos si el mensaje es corto (menos de 6 palabras)
    # Ejemplo: "Hola" -> Responde saludo.
    # Ejemplo: "Hola tengo una plaga" -> Pasa al RAG (es una consulta).
    words = t.split()
    if len(words) < 6:
        saludos = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'hi', 'holi']
        despedidas = ['adios', 'chao', 'hasta luego', 'nos vemos', 'gracias', 'muchas gracias']
        
        if any(x in t for x in saludos):
            return "👋 **¡Hola!** Soy Dr. agro. ¿En qué puedo ayudarte hoy con tu cultivo?"
            
        if any(x in t for x in despedidas):
            return "🤝 **¡Con gusto!** Espero haber sido de ayuda. Revisa tus cultivos frecuentemente."

    return None

def descargar_rags_memoria():
    """
    Fuerza la eliminación de los modelos de Embeddings de los RAGs
    para dejar espacio al modelo visual LLaVA.
    """
    if "chains" in st.session_state and st.session_state.chains:
        print("🧹 Limpiando RAGs para liberar VRAM...")
        # Iteramos sobre las cadenas cargadas
        keys_to_remove = []
        for name, chain in st.session_state.chains.items():
            # Intentamos acceder al modelo dentro del retriever y borrarlo
            try:
                if hasattr(chain, 'retriever'):
                    if hasattr(chain.retriever, 'model'):
                        del chain.retriever.model
                    if hasattr(chain.retriever, 'model_match'):
                        del chain.retriever.model_match
            except Exception:
                pass
            keys_to_remove.append(name)
        
        # Borramos las referencias
        st.session_state.chains = {}
        st.session_state.current_rag = None
        
        # Limpieza profunda
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()




def initialize_llm_safe(config_name="default"):
    """Wrapper seguro para inicializar LLM"""
    is_running, msg = verificar_ollama()
    if not is_running:
        return None, False, msg
    
    try:
        lm = get_ollama_lm(config_name)
        # Re-asegurar configuración local si es necesario
        try:
            dspy.settings.configure(lm=lm)
        except:
            pass
        return lm, True, "Conectado"
    except Exception as e:
        return None, False, f"Error DSPy: {str(e)}"

@st.cache_resource
def load_rag_cached(rag_name):
    """Carga el RAG y lo guarda en caché"""
    config = RAG_CONFIG[rag_name]
    path_root = config["path_root"]
    path_faiss = os.path.join(path_root, config["index_name"])
    path_docs = os.path.join(path_root, config["docs_name"])
    
    if not os.path.exists(path_faiss) or not os.path.exists(path_docs):
        return None, False, "Archivos de índice no encontrados"

    try:
        retriever = RerankedFaissRetriever(path_faiss, path_docs, model_match=config["model_match"])
        
        # Configurar retriever si existe config
        if config.get("retriever_config"):
            ret_conf = config["retriever_config"]
            for k, v in ret_conf.items():
                if hasattr(retriever, k):
                    setattr(retriever, k, v)

        if config["use_faster"]:
            chain = faster_UniversityRAGChain(retriever=retriever, model_match=config["model_match"], top_n=config.get("top_n", 4))
        else:
            chain = UniversityRAGChain(retriever=retriever, model_match=config["model_match"])
            
        return chain, True, "Cargado"
    except Exception as e:
        return None, False, str(e)

def cargar_llava():
    """Carga LLaVA usando el nuevo script optimizado"""
    if not LLAVA_AVAILABLE: return False, "Módulo no instalado"
    
    # 1. Si hay video, fuera.
    if st.session_state.video_loaded:
        liberar_modelo("video")
    
    # 2. IMPORTANTE: Sacar los RAGs de la GPU
    descargar_rags_memoria()
    
    # 3. Limpieza final
    liberar_memoria_cuda()
    
    if st.session_state.llava_detector is None:
        st.session_state.llava_detector = LlavaPlantDiseaseDetector()
    
    # 4. Intentar cargar
    try:
        success, msg = st.session_state.llava_detector.load_model()
        if success:
            st.session_state.llava_loaded = True
        return success, msg
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, "❌ Error VRAM: Cierra otros programas o reinicia Ollama."
        return False, str(e)

def cargar_video():
    """Carga VideoLLaMA3"""
    if not VIDEOLLAMA_AVAILABLE: return False, "Módulo no instalado"
    
    if st.session_state.llava_loaded:
        liberar_modelo("llava")
    
    liberar_memoria_cuda()
    
    if st.session_state.video_analyzer is None:
        st.session_state.video_analyzer = VideoLlamaAgriculturalAnalyzer()
        
    success, msg = st.session_state.video_analyzer.load_model()
    if success:
        st.session_state.video_loaded = True
    return success, msg

# ============================================
# LÓGICA DE INTERFAZ (SIDEBAR)
# ============================================
with st.sidebar:
    st.title("🌱 Dr. agro")
    st.caption("Asistente IA para Agricultura")
    st.divider()
    
    # Estado Ollama
    lm, success, msg = initialize_llm_safe("default")
    if success:
        st.success(msg, icon="🤖")
        st.session_state.lm = lm
    else:
        st.error(f"Ollama Offline: {msg}")
        st.info("Ejecuta `ollama serve` en terminal")
    
    st.divider()
    
    # Carga de Modelos Visuales
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imagen**")
        if st.session_state.llava_loaded:
            if st.button("⬇️ Apagar", key="btn_off_llava", use_container_width=True):
                liberar_modelo("llava")
                st.rerun()
        else:
            if st.button("⬆️ Cargar", key="btn_on_llava", disabled=not LLAVA_AVAILABLE, use_container_width=True):
                with st.spinner("Cargando LLaVA..."):
                    ok, m = cargar_llava()
                    if not ok: st.error(m)
                    else: st.rerun()
    
    with col2:
        st.markdown("**Video**")
        if st.session_state.video_loaded:
            if st.button("⬇️ Apagar", key="btn_off_video", use_container_width=True):
                liberar_modelo("video")
                st.rerun()
        else:
            if st.button("⬆️ Cargar", key="btn_on_video", disabled=not VIDEOLLAMA_AVAILABLE, use_container_width=True):
                with st.spinner("Cargando VideoAI..."):
                    ok, m = cargar_video()
                    if not ok: st.error(m)
                    else: st.rerun()

    st.divider()
    
    # Selección de RAG
    rag_seleccionado = st.selectbox("🧠 Cerebro (RAG):", list(RAG_CONFIG.keys()))
    
    # Cambio de RAG dinámico
    if rag_seleccionado != st.session_state.current_rag:
        with st.spinner(f"Cargando conocimientos de {rag_seleccionado}..."):
            chain, ok, msg = load_rag_cached(rag_seleccionado)
            if ok:
                st.session_state.chains[rag_seleccionado] = chain
                st.session_state.current_rag = rag_seleccionado
                st.session_state.lm = get_ollama_lm(rag_seleccionado) # Actualizar config LLM
                st.rerun()
            else:
                st.error(f"Error RAG: {msg}")

    if torch.cuda.is_available():
        gb_used = torch.cuda.memory_allocated()/1024**3
        gb_total = torch.cuda.get_device_properties(0).total_memory/1024**3
        st.progress(gb_used/gb_total, text=f"VRAM: {gb_used:.1f}/{gb_total:.1f} GB")

    if st.button("🗑️ Limpiar Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================
# ÁREA PRINCIPAL
# ============================================
#st.header("Dr. agro")

# Historial
for msg in st.session_state.messages:
    avatar = "🧑‍🌾" if msg["role"] == "user" else "🌱"
    with st.chat_message(msg["role"], avatar=avatar):
        if "image" in msg: st.image(msg["image"], width=250)
        if "video" in msg: st.info(f"🎥 Analizando: {msg['video']}")
        st.markdown(msg["content"])

# ============================================
# ÁREA DE ADJUNTOS (Con claves de reseteo)
# ============================================
# Usamos una clave de reseteo en el session_state
if 'upload_key_img' not in st.session_state:
    st.session_state.upload_key_img = 0
if 'upload_key_vid' not in st.session_state:
    st.session_state.upload_key_vid = 0

col_adj1, col_adj2, col_adj3 = st.columns([1, 1, 3])
with col_adj1:
    upl_img = st.file_uploader(
        "📷", 
        type=["jpg","png","jpeg"], 
        label_visibility="collapsed",
        key=f"img_{st.session_state.upload_key_img}" # <--- CLAVE DE RESETEO
    )
    if upl_img: 
        st.session_state.pending_image = Image.open(upl_img)
        st.caption(f"📎 {upl_img.name}")

with col_adj2:
    upl_vid = st.file_uploader(
        "🎥", 
        type=["mp4"], 
        label_visibility="collapsed",
        key=f"vid_{st.session_state.upload_key_vid}" # <--- CLAVE DE RESETEO
    )
    if upl_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(upl_vid.read())
        st.session_state.pending_video = upl_vid.name
        st.session_state.pending_video_path = tfile.name
        st.caption(f"📎 {upl_vid.name}")

# Input
if prompt := st.chat_input("Describe tu problema o sube una foto..."):
    
    # 1. Preparar mensaje usuario
    user_msg = {"role": "user", "content": prompt}
    if st.session_state.pending_image: user_msg["image"] = st.session_state.pending_image
    if st.session_state.pending_video: user_msg["video"] = st.session_state.pending_video
    
    st.session_state.messages.append(user_msg)
    
    # Mostrar inmediato
    with st.chat_message("user", avatar="🧑‍🌾"):
        if st.session_state.pending_image: st.image(st.session_state.pending_image, width=250)
        st.markdown(prompt)
    
    # 2. Generar Respuesta
    with st.chat_message("assistant", avatar="🌱"):
        response_text = ""
        
        # CASO A: IMAGEN (Prioridad 1)
        if st.session_state.pending_image:
            if not st.session_state.llava_loaded:
                st.warning("⚠️ El motor visual (LLaVA) está apagado. Cárgalo en el menú lateral.")
                response_text = "Por favor, activa LLaVA en el menú lateral para que pueda ver tu imagen."
            else:
                with st.spinner("👁️ Analizando imagen..."):
                    ok, resp = st.session_state.llava_detector.analyze_image(
                        st.session_state.pending_image, prompt
                    )
                    response_text = resp if ok else f"Error visual: {resp}"

        # CASO B: VIDEO (Prioridad 2)
        elif st.session_state.pending_video:
            if not st.session_state.video_loaded:
                st.warning("⚠️ El motor de video está apagado. Cárgalo en el menú lateral.")
                response_text = "Por favor, activa VideoAI en el menú lateral."
            else:
                with st.spinner("🎬 Mirando video (esto toma tiempo)..."):
                    ok, resp = st.session_state.video_analyzer.analyze_video(
                        st.session_state.pending_video_path, prompt
                    )
                    response_text = resp if ok else f"Error video: {resp}"
                    # Limpieza
                    try: os.remove(st.session_state.pending_video_path)
                    except: pass

        # CASO C: TEXTO / RAG (Default)
        else:
            # ---------------------------------------------------------
            # 1. INTENTO DE CHAT RÁPIDO (Sin RAG)
            # ---------------------------------------------------------
            fast_response = verificar_intencion_chat(prompt)
            
            if fast_response:
                response_text = fast_response
            
            # ---------------------------------------------------------
            # 2. CONSULTA AL RAG (Si no es un saludo/pregunta simple)
            # ---------------------------------------------------------
            else:
                if not st.session_state.current_rag or st.session_state.current_rag not in st.session_state.chains:
                    st.warning("⚠️ RAG no cargado. Selecciona uno en el sidebar.")
                    response_text = "No tengo acceso a mi base de conocimientos. Por favor selecciona un 'Cerebro' en el menú lateral."
                else:
                    with st.spinner("📖 Consultando manuales..."):
                        try:
                            chain = st.session_state.chains[st.session_state.current_rag]
                            
                            # Contexto del sistema (experto agrícola)
                            sys_context = st.session_state.get("system_prompt", "Eres un experto agrícola de Agrosavia.")
                            
                            # Llamada al RAG
                            res, _ = chain(question=prompt, ext_context=sys_context) 
                            
                            response_text = clean_output(res.get("answer", "Sin respuesta"))
                        except Exception as e:
                            # Manejo de errores silencioso para el usuario
                            print(f"Error RAG: {str(e)}")
                            response_text = "Lo siento, tuve un problema técnico consultando mis manuales. ¿Podrías reformular la pregunta?"

        # Mostrar respuesta final (sea del Fast Path o del RAG)
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # A. Limpiar variables de estado internas
    st.session_state.pending_image = None
    st.session_state.pending_video = None
    st.session_state.pending_video_path = None

    # B. 🔥 RESETEAR EL WIDGET DE FILE UPLOADER 🔥
    # Al cambiar la clave, Streamlit trata el widget como uno nuevo.
    # El archivo subido desaparece visualmente, pero su contexto ya está en messages.
    st.session_state.upload_key_img += 1
    st.session_state.upload_key_vid += 1
    st.rerun()