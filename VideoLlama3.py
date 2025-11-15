import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import gc
from pathlib import Path

# ---------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------
device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

print(">>> Cargando modelo optimizado para GPU de 24GB...")

# Configuración de atención optimizada
try:
    from flash_attn import flash_attn_func
    attn_backend = "flash_attention_2"
    print("✓ Flash Attention 2 disponible")
except ImportError:
    attn_backend = "eager"
    print("⚠ Usando atención estándar (instala flash-attn para mejor rendimiento)")

# Cargar modelo con optimizaciones
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation=attn_backend,
    use_cache=True,
)

model.eval()

# Optimización con compilación (opcional)
if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 7:
    print(">>> Compilando modelo con torch.compile()...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠ No se pudo compilar: {e}")

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# ---------------------------------------------
# FUNCIÓN DE INFERENCIA OPTIMIZADA
# ---------------------------------------------
def analizar_video(
    video_path: str,
    pregunta: str,
    fps: float = 0.5,
    max_frames: int = 24,
    max_new_tokens: int = 256,
    use_sampling: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Analiza un video y responde preguntas en español.
    
    Args:
        video_path: Ruta al video
        pregunta: Pregunta sobre el video
        fps: Frames por segundo a extraer
        max_frames: Máximo de frames a procesar
        max_new_tokens: Tokens máximos en la respuesta
        use_sampling: Si usar sampling (True) o greedy (False)
        temperature: Control de creatividad (solo si use_sampling=True)
        top_p: Nucleus sampling (solo si use_sampling=True)
    """
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    
    # System prompt en español con instrucción explícita
    conversation = [
        {
            "role": "system",
            "content": """Eres un experto agrónomo especializado en el cultivo de cacao. 
Analiza videos e identifica plagas, enfermedades y condiciones del cultivo con precisión.
IMPORTANTE: Siempre responde en español, de forma clara y profesional."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": fps,
                        "max_frames": max_frames
                    }
                },
                {
                    "type": "text",
                    "text": f"{pregunta}\n\nResponde en español."
                },
            ]
        },
    ]
    
    print(f">>> Procesando video: {max_frames} frames @ {fps} fps")
    
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Mover a GPU
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    print(">>> Generando respuesta en español...")
    
    # Configuración de generación
    if use_sampling:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.15,  # Aumentado para evitar repeticiones
            "use_cache": True,
        }
    else:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.15,
            "use_cache": True,
        }
    
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_config)
    
    # Limpiar memoria
    del inputs
    torch.cuda.empty_cache()
    
    # Decodificar respuesta
    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()
    
    # Extraer solo la respuesta del asistente
    if "assistant" in response.lower():
        parts = response.split("assistant")
        response = parts[-1].strip()
        for prefix in [":", "\n", " "]:
            response = response.lstrip(prefix)
    
    return response

# ---------------------------------------------
# EJEMPLOS DE USO
# ---------------------------------------------
if __name__ == "__main__":
    video_path = "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads/cacao20250916133108/video/Ver_video_36367.mp4"
    
    # ============================================================
    # EJEMPLO 1: DETECCIÓN RÁPIDA DE PLAGAS
    # ============================================================
    print("\n" + "="*60)
    print("EJEMPLO 1: DETECCIÓN RÁPIDA DE PLAGAS")
    print("="*60)
    
    respuesta = analizar_video(
        video_path=video_path,
        pregunta="¿Qué plagas observas en el video?",
        fps=0.5,
        max_frames=16,
        max_new_tokens=200,
        use_sampling=False  # Modo rápido y determinista
    )
    print(f"\n>>> RESPUESTA:\n{respuesta}\n")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # ============================================================
    # EJEMPLO 2: ANÁLISIS DETALLADO
    # ============================================================
    print("\n" + "="*60)
    print("EJEMPLO 2: ANÁLISIS DETALLADO")
    print("="*60)
    
    respuesta_detallada = analizar_video(
        video_path=video_path,
        pregunta="""Analiza el video y describe:
- Plagas o enfermedades presentes
- Nivel de severidad
- Estado general del cultivo
- Recomendaciones de manejo""",
        fps=1,
        max_frames=24,
        max_new_tokens=400,
        use_sampling=True,
        temperature=0.6
    )
    print(f"\n>>> RESPUESTA DETALLADA:\n{respuesta_detallada}\n")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # ============================================================
    # EJEMPLO 3: IDENTIFICACIÓN DE ENFERMEDADES
    # ============================================================
    print("\n" + "="*60)
    print("EJEMPLO 3: IDENTIFICACIÓN DE ENFERMEDADES")
    print("="*60)
    
    respuesta_enfermedad = analizar_video(
        video_path=video_path,
        pregunta="¿Se observan síntomas de enfermedades en las plantas de cacao? Describe los síntomas visibles.",
        fps=0.75,
        max_frames=20,
        max_new_tokens=300,
        use_sampling=True,
        temperature=0.5
    )
    print(f"\n>>> RESPUESTA:\n{respuesta_enfermedad}\n")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # ============================================================
    # EJEMPLO 4: EVALUACIÓN DE ESTADO GENERAL
    # ============================================================
    print("\n" + "="*60)
    print("EJEMPLO 4: EVALUACIÓN DE ESTADO GENERAL")
    print("="*60)
    
    respuesta_estado = analizar_video(
        video_path=video_path,
        pregunta="Evalúa el estado general del cultivo de cacao en el video. ¿Qué observas?",
        fps=0.5,
        max_frames=16,
        max_new_tokens=250,
        use_sampling=False
    )
    print(f"\n>>> RESPUESTA:\n{respuesta_estado}\n")
    
    # ============================================================
    # ESTADÍSTICAS
    # ============================================================
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE MEMORIA GPU")
    print("="*60)
    print(f"Memoria asignada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memoria reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Memoria máxima usada: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")