"""
Dr. agro - Sistema Multi-RAG con Gradio (Diseño Moderno)
Sistema de consulta inteligente con múltiples bases de conocimiento
"""

import os
import base64
import gradio as gr
import dspy
from utils import RerankedFaissRetriever, UniversityRAGChain, faster_UniversityRAGChain, clean_output
from datetime import datetime
import traceback

# ============================================
# CARGAR IMAGEN COMO BASE64
# ============================================
def load_image_as_base64(path):
    """Convierte imagen a base64 para usar en CSS background"""
    try:
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except:
        return None

BG_IMAGE = load_image_as_base64("images/image.png")

# ============================================
# CONFIGURACIÓN DE RAGs
# ============================================
RAG_CONFIG = {
    "RAG 1 - BAC (Cacao)": {
        "path_root": "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "description": "📚 Base de conocimiento completa del BAC",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": False,  # Usar UniversityRAGChain
        "top_n": 10
    },
    "RAG 2 - Dr. agro Específico": {
        "path_root": "./llm_cacao-dragro",
        "index_name": "profiles_index.faiss",
        "docs_name": "profiles_docs.pkl",
        "description": "🌱 Base especializada: deficiencias, escoba de bruja, monilia, phytophthora",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": False,
        "top_n": 10
    },
    "RAG 3 - Zonas Productoras": {
        "path_root": "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "description": "🗺️ Información sobre zonas productoras y cultivos agroforestales",
        "model_match": "intfloat/multilingual-e5-large",
        "use_faster": True,  # Usar faster_UniversityRAGChain
        "top_n": 4
    }
}

# Variables globales
lm = None
retrievers = {}
chains = {}
current_rag = None
SYSTEM_PROMPT = ""

# ============================================
# FUNCIONES PRINCIPALES
# ============================================
def initialize_llm():
    """Inicializa el modelo de lenguaje Mistral con Ollama"""
    global lm
    if lm is not None:
        return True, "✅ Modelo ya inicializado"
    
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
        return True, "✅ Modelo Mistral inicializado correctamente"
    
    except ConnectionError:
        return False, "❌ Error: No se puede conectar con Ollama en localhost:11434"
    except Exception as e:
        traceback.print_exc()
        return False, f"❌ Error al inicializar: {type(e).__name__}: {str(e)}"

def load_rag(rag_name):
    """Carga un RAG específico con su configuración"""
    global retrievers, chains, current_rag
    
    # Si ya está cargado, solo cambiamos el actual
    if rag_name in chains:
        current_rag = rag_name
        config = RAG_CONFIG[rag_name]
        return True, f"✅ {rag_name}\n{config['description']}"
    
    try:
        config = RAG_CONFIG[rag_name]
        path_root = config["path_root"]
        path_faiss = os.path.join(path_root, config["index_name"])
        path_docs = os.path.join(path_root, config["docs_name"])
        
        # Verificar que existan los archivos
        if not os.path.exists(path_faiss):
            return False, f"❌ Archivo FAISS no encontrado: {path_faiss}"
        if not os.path.exists(path_docs):
            return False, f"❌ Archivo de documentos no encontrado: {path_docs}"
        
        # Crear retriever con configuración específica
        retriever = RerankedFaissRetriever(
            path_faiss, 
            path_docs, 
            model_match=config["model_match"]
        )
        
        # Configurar parámetros del retriever si existen
        if hasattr(retriever, "k"):
            retriever.k = 20  # Recuperar más candidatos iniciales
        
        # Crear la cadena RAG apropiada
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
        
        # Guardar en caché
        retrievers[rag_name] = retriever
        chains[rag_name] = chain
        current_rag = rag_name
        
        return True, f"✅ {rag_name} cargado exitosamente\n{config['description']}"
    
    except FileNotFoundError as e:
        return False, f"❌ Archivo no encontrado: {str(e)}"
    except Exception as e:
        traceback.print_exc()
        return False, f"❌ Error al cargar RAG: {type(e).__name__}: {str(e)}"

def set_system_prompt(text):
    """Actualiza el prompt del sistema"""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = text or ""
    return "🧠 Prompt del sistema actualizado correctamente."

def consultar_rag(message, history):
    """Procesa una consulta usando el RAG actual"""
    # Validaciones básicas
    if not message or not message.strip():
        return "⚠️ Por favor, escribe una pregunta."
    
    if lm is None:
        return "❌ El modelo de lenguaje no está inicializado. Por favor, recarga la página."
    
    if current_rag is None:
        return "❌ No hay ningún RAG seleccionado. Por favor, elige uno del panel lateral."
    
    chain = chains.get(current_rag)
    if not chain:
        return f"❌ El RAG '{current_rag}' no está disponible. Intenta cargarlo de nuevo."
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ejecutar la cadena RAG
        response, context = chain(message, ext_context=SYSTEM_PROMPT or "")
        
        # Validar respuesta
        if not response or "answer" not in response:
            return "❌ No se pudo generar una respuesta. Por favor, intenta reformular tu pregunta."
        
        respuesta = clean_output(response["answer"])
        
        # Agregar metadata
        rag_config = RAG_CONFIG[current_rag]
        footer = f"\n\n---\n📊 **RAG usado:** {current_rag}\n⏰ **Hora:** {timestamp}"
        
        return f"{respuesta}{footer}"
    
    except KeyError as e:
        print(f"[ERROR] Clave faltante en respuesta: {e}")
        traceback.print_exc()
        return "❌ Error en el formato de respuesta del modelo. Verifica los logs."
    
    except ConnectionError:
        return "❌ Error de conexión con Ollama. ¿Está corriendo en localhost:11434?"
    
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return f"❌ Error inesperado: {type(e).__name__}. Revisa los logs del servidor."

def obtener_info_sistema():
    """Genera información del estado actual del sistema"""
    info = "**Estado del Sistema**\n\n"
    info += "🟢 LLM Activo\n" if lm else "🔴 LLM no inicializado\n"
    info += f"\n**RAGs cargados:** {len(chains)}/{len(RAG_CONFIG)}\n\n"
    
    for name in RAG_CONFIG.keys():
        status = "🟢" if name in chains else "⚪"
        info += f"{status} {name}\n"
    
    if current_rag:
        info += f"\n**RAG activo:** {current_rag}"
    else:
        info += "\n**RAG activo:** Ninguno"
    
    return info

# ============================================
# INTERFAZ MODERNA CON GRADIO
# ============================================
def crear_interfaz_chat():
    """Crea la interfaz de usuario con Gradio"""
    
    # CSS Moderno
    custom_css = """
    /* Tema general */
    :root {
        --sidebar-bg: #f0f9f4;
        --sidebar-text: #2d5f3f;
        --sidebar-hover: #e6f4ea;
        --chat-bg: #f8f9fa;
        --primary-color: #4caf50;
        --message-user: #e3f2fd;
        --message-assistant: #ffffff;
    }
    
    /* Contenedor principal */
    .gradio-container {
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Panel lateral izquierdo - estilo claro verde */
    #sidebar {
        background: linear-gradient(180deg, #f0f9f4 0%, #e8f5e9 100%) !important;
        padding: 24px 20px !important;
        border-radius: 0 !important;
        border-right: 2px solid #c8e6c9 !important;
        min-height: 100vh !important;
        color: var(--sidebar-text) !important;
    }
    
    #sidebar h1 {
        color: #2e7d32 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    
    #sidebar p {
        color: #43a047 !important;
        font-size: 14px !important;
        margin-bottom: 24px !important;
        opacity: 0.9 !important;
    }
    
    /* Secciones del sidebar */
    #sidebar .label-wrap {
        color: #2d5f3f !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-top: 20px !important;
    }
    
    #sidebar .wrap {
        background: white !important;
        border: 1px solid #c8e6c9 !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    
    #sidebar button {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        width: 100% !important;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2) !important;
    }
    
    #sidebar button:hover {
        background: linear-gradient(135deg, #45a049 0%, #388e3c 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Área de chat principal */
    #chat_area {
        background: var(--chat-bg) !important;
        padding: 20px !important;
        min-height: 100vh !important;
    }
    
    #chatbot_main {
        height: calc(100vh - 240px) !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        background: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    
    /* Mensajes del chat */
    .message-wrap {
        padding: 16px 20px !important;
        margin: 8px 0 !important;
        border-radius: 12px !important;
    }
    
    .user .message-wrap {
        background: var(--message-user) !important;
        margin-left: auto !important;
        max-width: 80% !important;
    }
    
    .bot .message-wrap {
        background: var(--message-assistant) !important;
        border: 1px solid #e0e0e0 !important;
        max-width: 85% !important;
    }
    
    /* Input de mensaje */
    #input_row {
        margin-top: 16px !important;
        gap: 12px !important;
    }
    
    #prompt_box textarea {
        border: 2px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-size: 15px !important;
        transition: all 0.2s !important;
        background: white !important;
    }
    
    #prompt_box textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
    }
    
    /* Botones de acción */
    #btn_enviar {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3) !important;
        transition: all 0.3s !important;
        height: 56px !important;
    }
    
    #btn_enviar:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(76, 175, 80, 0.4) !important;
    }
    
    #btn_limpiar {
        background: #f5f5f5 !important;
        color: #666 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        height: 56px !important;
    }
    
    #btn_limpiar:hover {
        background: #eeeeee !important;
        border-color: #d0d0d0 !important;
    }
    
    /* Panel derecho - información */
    #history_panel {
        background: white !important;
        padding: 20px !important;
        border-left: 1px solid #e0e0e0 !important;
        min-height: 100vh !important;
    }
    
    #history_panel h3 {
        color: #333 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        margin-bottom: 16px !important;
        padding-bottom: 12px !important;
        border-bottom: 2px solid #4caf50 !important;
    }
    
    /* Accordion */
    .accordion {
        background: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
    }
    
    /* Scrollbar personalizado */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        #sidebar {
            min-height: auto !important;
        }
        #chatbot_main {
            height: 500px !important;
        }
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate"
        ),
        title="🌱 Dr. agro - Sistema Multi-RAG",
        css=custom_css
    ) as demo:
        
        with gr.Row(equal_height=False):
            # ========== SIDEBAR IZQUIERDO ==========
            with gr.Column(scale=2, elem_id="sidebar"):
                gr.Markdown("""
                # 🌱 Dr. agro
                <p>Sistema Multi-RAG para Cacao</p>
                """)
                
                # Estado del sistema
                gr.Markdown("### 📊 Estado del Sistema")
                estado_sistema = gr.Markdown(value="Inicializando...", elem_classes="status-box")
                
                # Selector de RAG
                gr.Markdown("### 🗂️ Seleccionar RAG")
                rag_selector = gr.Radio(
                    choices=list(RAG_CONFIG.keys()),
                    value=list(RAG_CONFIG.keys())[0],
                    label="",
                    show_label=False
                )
                btn_cambiar_rag = gr.Button("🔄 Cambiar RAG", size="sm")
                
                # Información del sistema
                gr.Markdown("### ⚙️ Información del Sistema")
                info_sistema = gr.Markdown(elem_classes="info-box")
                btn_actualizar_info = gr.Button("🔄 Actualizar Estado", size="sm")
                
                btn_actualizar_info.click(
                    fn=obtener_info_sistema,
                    outputs=info_sistema
                )
            
            # ========== ÁREA DE CHAT PRINCIPAL ==========
            with gr.Column(scale=5, elem_id="chat_area"):
                # Accordion para el prompt del sistema
                with gr.Accordion("🧠 Configuración del Prompt del Sistema", open=False):
                    prompt_tb = gr.Textbox(
                        placeholder="Ej: Eres un experto en agronomía especializado en cacao...",
                        lines=3,
                        label="System Prompt",
                        show_label=False
                    )
                    with gr.Row():
                        btn_prompt = gr.Button("💾 Guardar Prompt", variant="secondary", size="sm")
                        prompt_status = gr.Markdown()
                    
                    btn_prompt.click(
                        fn=set_system_prompt,
                        inputs=prompt_tb,
                        outputs=prompt_status
                    )
                
                # Chatbot principal
                default_msgs = [
                    {
                        "role": "assistant",
                        "content": "👋 ¡Hola! Soy Dr. agro, tu asistente especializado en cacao. Selecciona un RAG del panel lateral y hazme tu consulta."
                    }
                ]
                
                chatbot = gr.Chatbot(
                    value=default_msgs,
                    type="messages",
                    show_label=False,
                    elem_id="chatbot_main",
                    avatar_images=(None, "images/iconchatbot.png"),
                    bubble_full_width=False
                )
                
                # Input de usuario
                with gr.Row(elem_id="input_row"):
                    user_input = gr.Textbox(
                        placeholder="Ejemplo: ¿Cuáles son las principales zonas productoras de cacao en Colombia?",
                        lines=1,
                        elem_id="prompt_box",
                        label="",
                        show_label=False,
                        scale=8
                    )
                    btn_enviar = gr.Button(
                        "Enviar 🚀",
                        variant="primary",
                        elem_id="btn_enviar",
                        scale=1
                    )
                    btn_limpiar = gr.Button(
                        "🗑️",
                        elem_id="btn_limpiar",
                        scale=0,
                        min_width=56
                    )
                
                # Estado interno
                history_tuples = gr.State([])
                
                def submit(msg, msgs, tuples):
                    """Procesa el envío de un mensaje"""
                    if not msg or not msg.strip():
                        return msgs, tuples, ""
                    
                    resp = consultar_rag(msg, tuples)
                    new_msgs = (msgs or []) + [
                        {"role": "user", "content": msg},
                        {"role": "assistant", "content": resp}
                    ]
                    new_tuples = (tuples or []) + [(msg, resp)]
                    return new_msgs, new_tuples, ""
                
                def clear():
                    """Limpia el historial de chat"""
                    default_msgs = [
                        {
                            "role": "assistant",
                            "content": "👋 ¡Hola! Soy Dr. agro. ¿En qué puedo ayudarte hoy?"
                        }
                    ]
                    default_tuples = []
                    return default_msgs, default_tuples, ""
                
                # Conectar eventos
                btn_enviar.click(
                    submit,
                    [user_input, chatbot, history_tuples],
                    [chatbot, history_tuples, user_input]
                )
                
                user_input.submit(
                    submit,
                    [user_input, chatbot, history_tuples],
                    [chatbot, history_tuples, user_input]
                )
                
                btn_limpiar.click(
                    clear,
                    outputs=[chatbot, history_tuples, user_input]
                )
            
            # ========== PANEL DERECHO - INFORMACIÓN ==========
            with gr.Column(scale=2, elem_id="history_panel"):
                gr.Markdown("### 📚 RAGs Disponibles")
                
                for rag_name, config in RAG_CONFIG.items():
                    with gr.Accordion(rag_name, open=False):
                        gr.Markdown(f"""
                        **Descripción:**  
                        {config['description']}
                        
                        **Modelo:** `{config['model_match']}`  
                        **Top N:** {config['top_n']}  
                        **Tipo:** {'Rápido' if config['use_faster'] else 'Estándar'}
                        """)
                
                with gr.Accordion("ℹ️ Guía de Uso", open=False):
                    gr.Markdown("""
                    **Cómo usar el sistema:**
                    
                    1. **Selecciona un RAG** del panel lateral
                    2. **Click en "Cambiar RAG"** para cargarlo
                    3. **Escribe tu pregunta** en el campo de texto
                    4. **Presiona Enter** o click en "Enviar"
                    
                    **Ejemplos de preguntas:**
                    
                    - ¿Qué es la moniliasis del cacao?
                    - Síntomas de la escoba de bruja
                    - ¿Cuáles son las zonas productoras?
                    - Tratamiento para phytophthora
                    - ¿Qué es el cultivo agroforestal?
                    
                    **RAGs disponibles:**
                    
                    - **RAG 1:** Base completa del BAC
                    - **RAG 2:** Enfermedades específicas
                    - **RAG 3:** Zonas y cultivos (optimizado)
                    """)
        
        # ========== EVENTOS DE INICIALIZACIÓN ==========
        btn_cambiar_rag.click(
            fn=lambda r: load_rag(r)[1],
            inputs=rag_selector,
            outputs=estado_sistema
        )
        
        def init_all():
            """Inicializa el sistema completo"""
            # Inicializar LLM
            success, msg = initialize_llm()
            if not success:
                return msg, obtener_info_sistema()
            
            # Cargar el primer RAG por defecto
            rag_default = list(RAG_CONFIG.keys())[0]
            success, rag_msg = load_rag(rag_default)
            
            if not success:
                return f"⚠️ LLM inicializado pero RAG falló:\n{rag_msg}", obtener_info_sistema()
            
            return f"✅ Sistema inicializado\n{rag_msg}", obtener_info_sistema()
        
        demo.load(
            fn=init_all,
            outputs=[estado_sistema, info_sistema]
        )
    
    return demo

# ============================================
# LANZAR APLICACIÓN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Dr. agro - Sistema Multi-RAG")
    print("=" * 60)
    
    # Verificar imagen
    if os.path.exists("images/image.png"):
        print("\n✅ Imagen cargada: images/image.png")
    else:
        print("\n⚠️ Imagen no encontrada: images/image.png")
    
    # Verificar RAGs
    print("\n📋 Verificando RAGs disponibles...")
    for name, config in RAG_CONFIG.items():
        path_faiss = os.path.join(config["path_root"], config["index_name"])
        path_docs = os.path.join(config["path_root"], config["docs_name"])
        
        faiss_ok = "✅" if os.path.exists(path_faiss) else "❌"
        docs_ok = "✅" if os.path.exists(path_docs) else "❌"
        
        print(f"\n{name}:")
        print(f"  {faiss_ok} FAISS: {path_faiss}")
        print(f"  {docs_ok} DOCS:  {path_docs}")
    
    print("\n🌐 Lanzando Gradio...")
    print("=" * 60)
    print()
    
    demo = crear_interfaz_chat()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )