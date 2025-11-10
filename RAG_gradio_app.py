"""
Dr. Agro - Sistema RAG con Gradio (Diseño Moderno)
"""

import os
import base64
import gradio as gr
import dspy
from utils import RerankedFaissRetriever, UniversityRAGChain, clean_output
from datetime import datetime
import traceback

# ============================================
# CARGAR IMAGEN COMO BASE64
# ============================================
def load_image_as_base64(path):
    """Convierte imagen a base64 para usar en CSS background"""
    try:
        with open(path, "rb") as f:
            return f"data:images/png;base64,{base64.b64encode(f.read()).decode()}"
    except:
        return None

BG_IMAGE = load_image_as_base64("images/image.png")

# ============================================
# CONFIGURACIÓN DE CORPUS
# ============================================
CORPUS_CONFIG = {
    "BAC (Cacao)": {
        "path_root": "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads",
        "index_name": "university_index.faiss",
        "docs_name": "university_docs.pkl",
        "description": "📚 Base de conocimiento completa"
    },
    "Dr. Agro Específico": {
        "path_root": "./llm_cacao-dragro",
        "index_name": "profiles_index.faiss",
        "docs_name": "profiles_docs.pkl",
        "description": "🌱 Base especializada: deficiencias, escoba de bruja, monilia, phytophthora"
    }
}

# Variables globales
lm = None
retrievers = {}
chains = {}
current_corpus = None
SYSTEM_PROMPT = ""

# ============================================
# FUNCIONES PRINCIPALES
# ============================================
def initialize_llm():
    global lm
    if lm is not None:
        return True, ""
    try:
        lm = dspy.LM(
            'ollama_chat/mistral',
            api_base='http://localhost:11434',
            api_key='',
            temperature=0,
            model_kwargs={"format": "json", "options": {"num_ctx": 8192}}
        )
        dspy.configure(lm=lm)
        return True, "✅ Modelo Mistral inicializado"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def load_corpus(corpus_name):
    global retrievers, chains, current_corpus
    if corpus_name in chains:
        current_corpus = corpus_name
        config = CORPUS_CONFIG[corpus_name]
        return True, f"✅ {corpus_name}\n{config['description']}"
    try:
        config = CORPUS_CONFIG[corpus_name]
        path_root = config["path_root"]
        path_faiss = os.path.join(path_root, config["index_name"])
        path_docs = os.path.join(path_root, config["docs_name"])
        
        if not os.path.exists(path_faiss) or not os.path.exists(path_docs):
            return False, "❌ Archivos no encontrados"
        
        retriever = RerankedFaissRetriever(path_faiss, path_docs)
        chain = UniversityRAGChain(retriever=retriever)
        retrievers[corpus_name] = retriever
        chains[corpus_name] = chain
        current_corpus = corpus_name
        return True, f"✅ {corpus_name}\n{config['description']}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def set_system_prompt(text):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = text or ""
    return "🧠 Prompt actualizado."

def consultar_rag(message, history):
    if not message or not message.strip():
        return "⚠️ Escribe una pregunta."
    if lm is None or current_corpus is None:
        return "❌ Sistema no listo."
    
    chain = chains.get(current_corpus)
    if not chain:
        return "❌ Corpus no disponible."
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response, _ = chain(message, ext_context=SYSTEM_PROMPT or "")
        respuesta = clean_output(response["answer"])
        return f"{respuesta}\n\n<small style='color: #888; font-size: 0.85em;'>⏰ {timestamp}</small>"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def obtener_info_sistema():
    info = "**Estado del Sistema**\n\n"
    info += "🟢 LLM Activo\n" if lm else "🔴 LLM no inicializado\n"
    info += f"\n**Corpus cargados:** {len(chains)}/2\n\n"
    for name in CORPUS_CONFIG.keys():
        status = "🟢" if name in chains else "⚪"
        info += f"{status} {name}\n"
    if current_corpus:
        info += f"\n**Corpus activo:** {current_corpus}"
    return info

# ============================================
# INTERFAZ MODERNA
# ============================================
def crear_interfaz_chat():
    
    # CSS Moderno inspirado en la imagen
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
    
    /* Panel derecho - historial */
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
        title="🌱 Dr. agro - Tu asistente virtual para el Cacao",
        css=custom_css
    ) as demo:

        with gr.Row(equal_height=False):
            
            # ========== SIDEBAR IZQUIERDO ==========
            with gr.Column(scale=2, elem_id="sidebar"):
                
                gr.Markdown("""
                # 🌱 Dr. agro
                <p>Tu asistente virtual para el Cacao</p>
                """)
                
                # Estado del sistema
                gr.Markdown("### 📊 Estado del Sistema")
                estado_sistema = gr.Markdown(value="Inicializando...", elem_classes="status-box")
                
                # Selector de corpus
                gr.Markdown("### 🗂️ Base de Datos")
                corpus_selector = gr.Radio(
                    choices=list(CORPUS_CONFIG.keys()),
                    value=list(CORPUS_CONFIG.keys())[0],
                    label="",
                    show_label=False
                )
                
                btn_cambiar_corpus = gr.Button("🔄 Cambiar Corpus", size="sm")
                
                # Información del sistema
                gr.Markdown("### ⚙️ Información")
                info_sistema = gr.Markdown(elem_classes="info-box")
                btn_actualizar_info = gr.Button("🔄 Actualizar", size="sm")
                
                btn_actualizar_info.click(fn=obtener_info_sistema, outputs=info_sistema)

            # ========== ÁREA DE CHAT PRINCIPAL ==========
            with gr.Column(scale=5, elem_id="chat_area"):
                
                # Accordion para el prompt del sistema
                with gr.Accordion("🧠 Configuración del Prompt del Sistema", open=False):
                    prompt_tb = gr.Textbox(
                        placeholder="Define el comportamiento del asistente aquí...",
                        lines=3,
                        label="System Prompt",
                        show_label=False
                    )
                    with gr.Row():
                        btn_prompt = gr.Button("💾 Guardar Prompt", variant="secondary", size="sm")
                        prompt_status = gr.Markdown()
                    btn_prompt.click(fn=set_system_prompt, inputs=prompt_tb, outputs=prompt_status)

                # Chatbot principal
                default_msgs = [
                    {"role": "assistant", "content": "👋 ¡Hola! Soy Dr. Agro, tu asistente especializado en cacao. ¿En qué puedo ayudarte hoy?"}
                ]
                default_tuples = [(None, "👋 ¡Hola! Soy Dr. Agro, tu asistente especializado en cacao. ¿En qué puedo ayudarte hoy?")]

                chatbot = gr.Chatbot(
                    value=default_msgs,
                    type="messages",
                    show_label=False,
                    elem_id="chatbot_main",
                    avatar_images=(None, "🌱"),
                    bubble_full_width=False
                )

                # Input de usuario
                with gr.Row(elem_id="input_row"):
                    user_input = gr.Textbox(
                        placeholder="Escribe tu pregunta sobre cacao...",
                        lines=1,
                        elem_id="prompt_box",
                        label="",
                        show_label=False,
                        scale=8
                    )
                    btn_enviar = gr.Button("Enviar 🚀", variant="primary", elem_id="btn_enviar", scale=1)
                    btn_limpiar = gr.Button("🗑️", elem_id="btn_limpiar", scale=0, min_width=56)

                history_tuples = gr.State(default_tuples)

                def submit(msg, msgs, tuples):
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
                    default_msgs = [
                        {"role": "assistant", "content": "👋 ¡Hola! Soy Dr. Agro, tu asistente especializado en cacao. ¿En qué puedo ayudarte hoy?"}
                    ]
                    default_tuples = [(None, "👋 ¡Hola! Soy Dr. Agro, tu asistente especializado en cacao. ¿En qué puedo ayudarte hoy?")]
                    return default_msgs, default_tuples, ""

                btn_enviar.click(submit, [user_input, chatbot, history_tuples], [chatbot, history_tuples, user_input])
                user_input.submit(submit, [user_input, chatbot, history_tuples], [chatbot, history_tuples, user_input])
                btn_limpiar.click(clear, outputs=[chatbot, history_tuples, user_input])

            # ========== PANEL DERECHO - HISTORIAL ==========
            with gr.Column(scale=2, elem_id="history_panel"):
                
                gr.Markdown("### 📜 Configuración")
                
                gr.Markdown("""
                **Corpus Disponibles:**
                - 📚 BAC (Cacao)
                - 🌱 Dr. Agro Específico
                
                **Funciones:**
                - Consultas sobre enfermedades
                - Análisis de síntomas
                - Recomendaciones de tratamiento
                """)
                
                with gr.Accordion("ℹ️ Ayuda", open=False):
                    gr.Markdown("""
                    **Cómo usar:**
                    
                    1. Selecciona un corpus
                    2. Escribe tu pregunta
                    3. Presiona Enter o click en Enviar
                    
                    **Ejemplos:**
                    - ¿Qué es la moniliasis?
                    - Síntomas de la escoba de bruja
                    - Tratamiento para phytophthora
                    """)

        # Eventos de inicialización
        btn_cambiar_corpus.click(
            fn=lambda c: load_corpus(c)[1],
            inputs=corpus_selector,
            outputs=estado_sistema
        )

        def init_all():
            initialize_llm()
            load_corpus(list(CORPUS_CONFIG.keys())[0])
            return "", obtener_info_sistema()

        demo.load(fn=init_all, outputs=[estado_sistema, info_sistema])

    return demo

# ============================================
# LANZAR
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Dr. Agro - Sistema RAG (Interfaz Moderna)")
    print("=" * 60)
    
    if os.path.exists("images/image.png"):
        print("\n✅ Imagen cargada: images/image.png")
    else:
        print("\n⚠️ Imagen no encontrada: images/image.png")
    
    print("\n📋 Verificando corpus...")
    for name, config in CORPUS_CONFIG.items():
        path_faiss = os.path.join(config["path_root"], config["index_name"])
        print(f"  {'✅' if os.path.exists(path_faiss) else '❌'} {name}")
    
    print("\n🌐 Lanzando Gradio...\n")
    demo = crear_interfaz_chat()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)