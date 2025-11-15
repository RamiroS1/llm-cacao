import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from transformers import AutoProcessor
import av
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------
# VERIFICACIÓN
# ---------------------------------------------
print(">>> Verificando dependencias...")
try:
    import sentencepiece
    print(f"✓ SentencePiece: {sentencepiece.__version__}")
except ImportError:
    print("✗ SentencePiece no encontrado")
    exit(1)

import transformers
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")

# ---------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = "llava-hf/LLaVA-NeXT-Video-7B-hf"

print(f"\n>>> Usando device: {device}")

# ---------------------------------------------
# CUANTIZACIÓN 4-BIT
# ---------------------------------------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------------------------
# CARGAR MODELO Y PROCESSOR
# ---------------------------------------------
print("\n>>> Cargando modelo y processor...")

# Usar AutoProcessor que maneja correctamente videos
processor = AutoProcessor.from_pretrained(model_path)

print(">>> Cargando modelo con cuantización 4-bit...")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map='auto',
    low_cpu_mem_usage=True,
)

print("✓ Modelo y processor cargados exitosamente\n")

# ---------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------
def read_video_pyav(container, indices):
    """Extrae frames específicos del video"""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def get_video_duration(container):
    """Obtiene duración del video en segundos"""
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base)
    return duration

# ---------------------------------------------
# CARGAR VIDEO
# ---------------------------------------------
print(">>> Cargando video...")

try:
    video_path = "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads/cacao20250916133108/video/Ver_video_36367.mp4"
    print(f"✓ Usando video local")
except Exception as e:
    print(f"⚠ No se pudo descargar video de ejemplo: {e}")
    video_path = "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads/cacao20250916133108/video/Ver_video_36367.mp4"
    print(f"✓ Usando video local")

# ---------------------------------------------
# PROCESAR VIDEO
# ---------------------------------------------
print("\n>>> Procesando video...")

container = av.open(video_path)
total_frames = container.streams.video[0].frames
fps = container.streams.video[0].average_rate
duration = get_video_duration(container)

print(f"  • Total frames: {total_frames}")
print(f"  • FPS: {fps}")
print(f"  • Duración: {duration:.2f} segundos")

# Extraer 8 frames uniformemente
num_frames = 8
indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
video_frames = read_video_pyav(container, indices)

print(f"  • Frames extraídos: {video_frames.shape}")

# ---------------------------------------------
# PREPARAR ENTRADA CON PROCESSOR
# ---------------------------------------------
print("\n>>> Preparando entrada para el modelo...")

# Convertir a lista de imágenes PIL
clip = [Image.fromarray(frame) for frame in video_frames]

# Crear conversación en formato correcto
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Que plagas aparecen en el video?"},
            {"type": "video"},
        ],
    },
]

# Usar el processor con el formato correcto
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(f"  • Prompt generado")

# Procesar con el método correcto para videos
inputs = processor(
    text=prompt,
    videos=clip,  # Pasar la lista de frames directamente
    padding=True,
    return_tensors="pt"
)

# Mover a GPU
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Convertir a float16 para ahorrar memoria
if "pixel_values_videos" in inputs:
    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.float16)

print(f"  • Input keys: {list(inputs.keys())}")
print(f"  • Shape input_ids: {inputs['input_ids'].shape}")
if "pixel_values_videos" in inputs:
    print(f"  • Shape pixel_values_videos: {inputs['pixel_values_videos'].shape}")

# ---------------------------------------------
# GENERACIÓN
# ---------------------------------------------
print("\n>>> Generando respuesta...")
print("  (Esto puede tomar 30-60 segundos...)\n")

try:
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # ---------------------------------------------
    # DECODIFICAR RESPUESTA
    # ---------------------------------------------
    # Obtener solo los tokens generados (sin el prompt)
    generated_ids = [
        output_ids[i][len(inputs["input_ids"][i]):] 
        for i in range(len(output_ids))
    ]
    
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    # ---------------------------------------------
    # MOSTRAR RESULTADOS
    # ---------------------------------------------
    print("="*70)
    print(">>> INFORMACIÓN DEL VIDEO:")
    print("="*70)
    print(f"Ruta: {video_path}")
    print(f"Duración: {duration:.2f} segundos")
    print(f"Frames totales: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Frames analizados: {num_frames}")
    
    print("\n" + "="*70)
    print(">>> RESPUESTA DEL MODELO:")
    print("="*70)
    print(response)
    print("="*70)
    
except Exception as e:
    print(f"✗ Error durante la generación: {e}")
    import traceback
    traceback.print_exc()

# ---------------------------------------------
# LIMPIAR MEMORIA
# ---------------------------------------------
import gc
torch.cuda.empty_cache()
gc.collect()

print("\n✓ Proceso completado. Memoria GPU liberada.")

# ---------------------------------------------
# MÉTRICAS DE USO DE MEMORIA
# ---------------------------------------------
if torch.cuda.is_available():
    print("\n>>> Uso de memoria GPU:")
    print(f"  • Memoria asignada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  • Memoria reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"  • Memoria máxima: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    
def analyze_video(video_path, question, model, processor, device="cuda:0", num_frames=8):
    """
    Analiza un video con LLaVA-NeXT-Video
    
    Args:
        video_path: Ruta al archivo de video
        question: Pregunta sobre el video
        model: Modelo LLaVA cargado
        processor: AutoProcessor cargado
        device: Dispositivo (cuda/cpu)
        num_frames: Número de frames a extraer
    
    Returns:
        str: Respuesta del modelo
    """
    import av
    import numpy as np
    from PIL import Image
    
    def read_video_pyav(container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    # Extraer frames
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    video_frames = read_video_pyav(container, indices)
    
    # Convertir a PIL
    clip = [Image.fromarray(frame) for frame in video_frames]
    
    # Crear conversación
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video"},
            ],
        },
    ]
    
    # Procesar
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.float16)
    
    # Generar
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=False,
        )
    
    # Decodificar
    generated_ids = [output_ids[i][len(inputs["input_ids"][i]):] for i in range(len(output_ids))]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return response

# Ejemplo de uso:
# response = analyze_video(
#     video_path="./mi_video.mp4",
#     question="¿Qué enfermedades del cacao se observan en el video?",
#     model=model,
#     processor=processor,
#     num_frames=16  # Más frames para análisis detallado
# )    