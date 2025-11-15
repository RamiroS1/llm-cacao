import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from transformers import AutoProcessor
import av
import numpy as np
from PIL import Image
import time
from datetime import datetime
import json
import psutil
import gc
from pathlib import Path

# ---------------------------------------------
# VERIFICACIÓN DE DEPENDENCIAS
# ---------------------------------------------
print(">>> Verificando dependencias...")
try:
    import sentencepiece
    print(f"✓ SentencePiece: {sentencepiece.__version__}")
except ImportError:
    print("✗ SentencePiece no encontrado. Instalar: pip install sentencepiece")
    exit(1)

import transformers
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA disponible: {torch.cuda.is_available()}")

# ---------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# Directorio para guardar resultados JSON
OUTPUT_DIR = Path("./resultados_inferencia")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n>>> Usando device: {device}")
if torch.cuda.is_available():
    print(f">>> GPU: {torch.cuda.get_device_name(0)}")
    print(f">>> VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ---------------------------------------------
# CONFIGURACIÓN DE CUANTIZACIÓN OPTIMIZADA
# ---------------------------------------------
print("\n>>> Configurando cuantización 4-bit...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ---------------------------------------------
# CARGAR MODELO Y PROCESSOR
# ---------------------------------------------
print(">>> Cargando processor...")
processor = AutoProcessor.from_pretrained(model_path)

print(">>> Cargando modelo con cuantización 4-bit (esto puede tardar 1-2 minutos)...")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map='auto',
    low_cpu_mem_usage=True,
)

model.eval()

print("✓ Modelo cargado exitosamente\n")

# ---------------------------------------------
# UTILIDADES DE MONITOREO
# ---------------------------------------------

def obtener_nombre_gpu():
    """Obtiene el nombre de la GPU"""
    try:
        return torch.cuda.get_device_name(0)
    except:
        return "Unknown GPU"

def obtener_metricas_sistema():
    """Obtiene métricas del sistema en tiempo real"""
    try:
        gpu_usage = torch.cuda.utilization(0) if torch.cuda.is_available() else 0
        gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_memory_reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
        
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram_usage_mb = psutil.virtual_memory().used / (1024 ** 2)
        
        return {
            "gpu_usage_percent": round(gpu_usage, 2),
            "gpu_memory_mb": round(gpu_memory_mb, 2),
            "gpu_memory_reserved_mb": round(gpu_memory_reserved_mb, 2),
            "cpu_usage_percent": round(cpu_usage, 2),
            "ram_usage_mb": round(ram_usage_mb, 2),
        }
    except Exception as e:
        return {}

def calcular_hallucination_score(response: str) -> float:
    """
    Calcula score de alucinación basado en patrones.
    Rango: 0.0 (sin alucinación) a 1.0 (alta alucinación)
    """
    score = 0.0
    response_lower = response.lower()
    
    patrones_negativos = [
        "no puedo", "lo siento", "no tengo acceso", 
        "no puedo ver", "no es posible", "no dispongo",
        "como modelo de lenguaje", "como ia"
    ]
    
    patrones_incertidumbre = [
        "probablemente", "quizás", "tal vez", "posiblemente",
        "podría ser", "es posible que", "no estoy seguro"
    ]
    
    if len(response.split()) < 10:
        score += 0.3
    
    for patron in patrones_negativos:
        if patron in response_lower:
            score += 0.4
            break
    
    count_incertidumbre = sum(1 for patron in patrones_incertidumbre if patron in response_lower)
    if count_incertidumbre > 3:
        score += 0.2
    
    palabras = response.split()
    if len(palabras) > 20:
        palabras_unicas = len(set(palabras))
        ratio_repeticion = palabras_unicas / len(palabras)
        if ratio_repeticion < 0.5:
            score += 0.3
    
    return min(round(score, 2), 1.0)

def calcular_toxicity_score(response: str) -> float:
    """Calcula score básico de toxicidad"""
    palabras_toxicas = [
        "idiota", "estúpido", "tonto", "inútil", 
        "basura", "porquería", "mierda"
    ]
    
    response_lower = response.lower()
    toxicidad = sum(1 for palabra in palabras_toxicas if palabra in response_lower)
    
    return min(round(toxicidad * 0.2, 2), 1.0)

def generar_request_id():
    """Genera ID único para la request"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = f"{np.random.randint(1000, 9999)}"
    return f"req_{timestamp}_{random_suffix}"

def generar_response_id():
    """Genera ID único para la respuesta"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = f"{np.random.randint(1000, 9999)}"
    return f"resp_{timestamp}_{random_suffix}"

def guardar_resultado_json(respuesta: str, metricas: dict, video_path: str, pregunta: str):
    """
    Guarda el resultado completo en un archivo JSON
    
    Args:
        respuesta: Respuesta del modelo
        metricas: Diccionario con todas las métricas
        video_path: Ruta del video analizado
        pregunta: Pregunta realizada
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultado_{timestamp}_{metricas['comportamiento']['response_id']}.json"
    filepath = OUTPUT_DIR / filename
    
    resultado_completo = {
        "metadata": {
            "video_path": str(video_path),
            "pregunta": pregunta,
            "modelo": "LLaVA-NeXT-Video-7B",
            "timestamp": datetime.now().isoformat()
        },
        "respuesta": respuesta,
        "metricas": metricas
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(resultado_completo, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultado guardado en: {filepath}")
    return str(filepath)

# ---------------------------------------------
# FUNCIONES DE PROCESAMIENTO DE VIDEO
# ---------------------------------------------

def read_video_pyav(container, indices):
    """Extrae frames específicos del video de forma eficiente"""
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

def get_video_info(container):
    """Obtiene información del video"""
    stream = container.streams.video[0]
    total_frames = stream.frames
    fps = float(stream.average_rate)
    duration = float(stream.duration * stream.time_base) if stream.duration else 0
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration
    }

# ---------------------------------------------
# FUNCIÓN PRINCIPAL DE ANÁLISIS CON MÉTRICAS
# ---------------------------------------------

def analizar_video(
    video_path: str,
    pregunta: str,
    num_frames: int = 8,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    mostrar_metricas: bool = True,
    guardar_json: bool = True
):
    """
    Analiza un video agrícola con LLaVA-NeXT-Video y retorna respuesta con métricas.
    
    Args:
        video_path: Ruta al video
        pregunta: Pregunta sobre el video
        num_frames: Número de frames a extraer (8-32 recomendado)
        max_new_tokens: Máximo de tokens a generar
        do_sample: Si usar sampling (True) o greedy (False)
        temperature: Control de creatividad
        top_p: Nucleus sampling
        mostrar_metricas: Si mostrar métricas en consola
        guardar_json: Si guardar resultado en JSON
    
    Returns:
        tuple: (respuesta, diccionario_metricas, ruta_json)
    """
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    
    # IDs únicos
    request_id = generar_request_id()
    response_id = generar_response_id()
    
    # Timestamps
    timestamp_inicio = datetime.now()
    tiempo_inicio = time.time()
    
    # Métricas iniciales
    metricas_inicio = obtener_metricas_sistema()
    memoria_gpu_inicio = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
    
    if mostrar_metricas:
        print(f">>> Procesando video: {num_frames} frames")
    
    # ---------------------------------------------
    # PROCESAMIENTO DEL VIDEO
    # ---------------------------------------------
    tiempo_video_inicio = time.time()
    
    container = av.open(video_path)
    video_info = get_video_info(container)
    
    indices = np.linspace(0, video_info["total_frames"] - 1, num_frames).astype(int)
    video_frames = read_video_pyav(container, indices)
    
    clip = [Image.fromarray(frame) for frame in video_frames]
    
    tiempo_video = (time.time() - tiempo_video_inicio) * 1000
    
    if mostrar_metricas:
        print(f"  • Video: {video_info['duration']:.1f}s | {video_info['total_frames']} frames @ {video_info['fps']:.1f} fps")
        print(f"  • Frames extraídos: {num_frames} ({tiempo_video:.0f}ms)")
    
    # ---------------------------------------------
    # PREPARAR ENTRADA
    # ---------------------------------------------
    tiempo_prep_inicio = time.time()
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{pregunta}\n\nResponde en español de forma clara y precisa."},
                {"type": "video"},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        text=prompt,
        videos=clip,
        padding=True,
        return_tensors="pt"
    )
    
    tokens_input = inputs['input_ids'].shape[1]
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.float16)
    
    tiempo_prep = (time.time() - tiempo_prep_inicio) * 1000
    
    if mostrar_metricas:
        print(f"  • Preparación: {tiempo_prep:.0f}ms | Tokens input: {tokens_input}")
    
    # ---------------------------------------------
    # GENERACIÓN
    # ---------------------------------------------
    if mostrar_metricas:
        print(">>> Generando respuesta en español...")
    
    tiempo_gen_inicio = time.time()
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
    }
    
    if do_sample:
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
        })
    
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    
    tiempo_gen = (time.time() - tiempo_gen_inicio) * 1000
    
    tokens_output = output_ids.shape[1] - tokens_input
    
    memoria_gpu_final = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
    
    # ---------------------------------------------
    # DECODIFICACIÓN
    # ---------------------------------------------
    tiempo_dec_inicio = time.time()
    
    generated_ids = [output_ids[i][len(inputs["input_ids"][i]):] for i in range(len(output_ids))]
    
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()
    
    tiempo_dec = (time.time() - tiempo_dec_inicio) * 1000
    
    # Limpiar memoria
    del inputs, output_ids, generated_ids
    torch.cuda.empty_cache()
    gc.collect()
    
    # ---------------------------------------------
    # CÁLCULO DE MÉTRICAS
    # ---------------------------------------------
    tiempo_total = (time.time() - tiempo_inicio) * 1000
    latency_ms = round(tiempo_total, 2)
    throughput_tps = round(tokens_output / (tiempo_gen / 1000), 2) if tiempo_gen > 0 else 0
    
    metricas_final = obtener_metricas_sistema()
    
    hallucination_score = calcular_hallucination_score(response)
    toxicity_score = calcular_toxicity_score(response)
    safety_check = toxicity_score < 0.3
    
    # ---------------------------------------------
    # CONSTRUCCIÓN DE MÉTRICAS
    # ---------------------------------------------
    metricas_comportamiento = {
        "response_id": response_id,
        "request_id": request_id,
        "model_name": "LLaVA-NeXT-Video-7B",
        "source_dataset": "LLaVA-Video-Dataset",
        "prompt_template": "agricultural_video_analysis_v1",
        "safety_check_passed": safety_check,
        "bias_flag": False,
        "toxicity_score": toxicity_score,
        "hallucination_score": hallucination_score
    }
    
    metricas_tiempo = {
        "tiempo_total_ms": latency_ms,
        "tiempo_video_ms": round(tiempo_video, 2),
        "tiempo_preparacion_ms": round(tiempo_prep, 2),
        "tiempo_generacion_ms": round(tiempo_gen, 2),
        "tiempo_decodificacion_ms": round(tiempo_dec, 2),
        "timestamp_inicio": timestamp_inicio.isoformat(),
        "timestamp_fin": datetime.now().isoformat()
    }
    
    metricas_modelo = {
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "latency_ms": latency_ms,
        "throughput_tps": throughput_tps,
        "gpu_memory_mb": round(memoria_gpu_final, 2),
        "gpu_memory_delta_mb": round(memoria_gpu_final - memoria_gpu_inicio, 2),
        "model_name": "LLaVA-NeXT-Video-7B",
        "quantization": "4-bit",
        "mode": "inference",
        "temperature": temperature if do_sample else 0.0,
        "prompt_tokens": tokens_input,
        "response_tokens": tokens_output,
        "max_new_tokens": max_new_tokens,
        "num_frames": num_frames,
        "video_duration_s": round(video_info["duration"], 2),
        "video_fps": round(video_info["fps"], 2)
    }
    
    metricas_sistema = {
        "timestamp": datetime.now().isoformat(),
        "gpu_usage_percent": metricas_final.get("gpu_usage_percent", 0),
        "gpu_memory_mb": metricas_final.get("gpu_memory_mb", 0),
        "gpu_memory_reserved_mb": metricas_final.get("gpu_memory_reserved_mb", 0),
        "cpu_usage_percent": metricas_final.get("cpu_usage_percent", 0),
        "ram_usage_mb": metricas_final.get("ram_usage_mb", 0),
        "latency_ms": latency_ms,
        "tokens_processed": tokens_input + tokens_output,
        "throughput_tps": throughput_tps,
        "model_name": "LLaVA-NeXT-Video-7B",
        "model_version": "7B-4bit",
        "device": obtener_nombre_gpu(),
        "batch_size": 1,
        "dtype": "float16",
        "quantization": "4-bit-nf4"
    }
    
    metricas = {
        "comportamiento": metricas_comportamiento,
        "tiempo": metricas_tiempo,
        "modelo": metricas_modelo,
        "sistema": metricas_sistema
    }
    
    # ---------------------------------------------
    # IMPRIMIR MÉTRICAS
    # ---------------------------------------------
    if mostrar_metricas:
        print("\n" + "="*70)
        print("📊 MÉTRICAS DE INFERENCIA - LLaVA-NeXT-Video")
        print("="*70)
        
        print("\n1️⃣  COMPORTAMIENTO DEL MODELO:")
        print(json.dumps(metricas_comportamiento, indent=2, ensure_ascii=False))
        
        print("\n2️⃣  TIEMPO DE RESPUESTA:")
        print(json.dumps(metricas_tiempo, indent=2, ensure_ascii=False))
        
        print("\n3️⃣  DATOS DEL MODELO:")
        print(json.dumps(metricas_modelo, indent=2, ensure_ascii=False))
        
        print("\n4️⃣  MÉTRICAS DE SISTEMA:")
        print(json.dumps(metricas_sistema, indent=2, ensure_ascii=False))
        
        print("\n" + "="*70)
    
    # ---------------------------------------------
    # GUARDAR JSON
    # ---------------------------------------------
    json_path = None
    if guardar_json:
        json_path = guardar_resultado_json(response, metricas, video_path, pregunta)
    
    return response, metricas, json_path

# ---------------------------------------------
# EJEMPLOS DE USO
# ---------------------------------------------
if __name__ == "__main__":
    video_path = "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads/cacao20250916133108/video/Ver_video_36367.mp4"
    
    resultados = []
    
    # ============================================================
    # EJEMPLO 1: DETECCIÓN RÁPIDA
    # ============================================================
    print("\n" + "🔍"*35)
    print("EJEMPLO 1: DETECCIÓN RÁPIDA DE PROBLEMAS")
    print("🔍"*35)
    
    respuesta1, metricas1, json1 = analizar_video(
        video_path=video_path,
        pregunta="¿Qué problemas fitosanitarios observas en el cultivo?",
        num_frames=8,
        max_new_tokens=200,
        do_sample=False,
        mostrar_metricas=True,
        guardar_json=True
    )
    
    print(f"\n✅ RESPUESTA DEL MODELO:")
    print("-" * 70)
    print(respuesta1)
    print("-" * 70)
    
    resultados.append({"ejemplo": "Rápido", "json_path": json1})
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    # ============================================================
    # EJEMPLO 2: ANÁLISIS DETALLADO
    # ============================================================
    print("\n" + "📊"*35)
    print("EJEMPLO 2: ANÁLISIS DETALLADO DEL CULTIVO")
    print("📊"*35)
    
    respuesta2, metricas2, json2 = analizar_video(
        video_path=video_path,
        pregunta="""Analiza el video y describe:
- Problemas o afectaciones visibles
- Nivel de severidad
- Estado general de las plantas""",
        num_frames=16,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.6,
        mostrar_metricas=True,
        guardar_json=True
    )
    
    print(f"\n✅ RESPUESTA DEL MODELO:")
    print("-" * 70)
    print(respuesta2)
    print("-" * 70)
    
    resultados.append({"ejemplo": "Detallado", "json_path": json2})
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    # ============================================================
    # EJEMPLO 3: IDENTIFICACIÓN ESPECÍFICA
    # ============================================================
    print("\n" + "🌿"*35)
    print("EJEMPLO 3: IDENTIFICACIÓN DE PLAGAS O ENFERMEDADES")
    print("🌿"*35)
    
    respuesta3, metricas3, json3 = analizar_video(
        video_path=video_path,
        pregunta="¿Qué tipo de plagas o enfermedades se observan en las plantas? Describe los síntomas visibles.",
        num_frames=12,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.5,
        mostrar_metricas=True,
        guardar_json=True
    )
    
    print(f"\n✅ RESPUESTA DEL MODELO:")
    print("-" * 70)
    print(respuesta3)
    print("-" * 70)
    
    resultados.append({"ejemplo": "Específico", "json_path": json3})
    
    # ============================================================
    # RESUMEN COMPARATIVO
    # ============================================================
    print("\n" + "="*70)
    print("📈 RESUMEN COMPARATIVO DE EJEMPLOS")
    print("="*70)
    
    ejemplos = [
        ("Rápido (8 frames)", metricas1),
        ("Detallado (16 frames)", metricas2),
        ("Específico (12 frames)", metricas3)
    ]
    
    for nombre, metricas in ejemplos:
        print(f"\n{nombre}:")
        print(f"  • Tiempo total: {metricas['tiempo']['tiempo_total_ms']:.0f} ms")
        print(f"  • Tiempo video: {metricas['tiempo']['tiempo_video_ms']:.0f} ms")
        print(f"  • Tiempo generación: {metricas['tiempo']['tiempo_generacion_ms']:.0f} ms")
        print(f"  • Tokens generados: {metricas['modelo']['tokens_output']}")
        print(f"  • Throughput: {metricas['modelo']['throughput_tps']:.2f} tokens/s")
        print(f"  • Memoria GPU: {metricas['modelo']['gpu_memory_mb']:.0f} MB")
        print(f"  • Score alucinación: {metricas['comportamiento']['hallucination_score']}")
    
    # ============================================================
    # RESUMEN DE ARCHIVOS JSON GENERADOS
    # ============================================================
    print("\n" + "="*70)
    print("📁 ARCHIVOS JSON GENERADOS")
    print("="*70)
    for resultado in resultados:
        print(f"  • {resultado['ejemplo']}: {resultado['json_path']}")
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)