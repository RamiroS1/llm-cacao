import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import gc
from pathlib import Path
import time
from datetime import datetime
import json
import psutil
import numpy as np

# ---------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------
device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

# Directorio para guardar resultados JSON
OUTPUT_DIR = Path("./resultados_inferencia")
OUTPUT_DIR.mkdir(exist_ok=True)

print(">>> Cargando modelo optimizado para GPU de 24GB...")

# Configuración de atención optimizada
try:
    from flash_attn import flash_attn_func
    attn_backend = "flash_attention_2"
    print("✓ Flash Attention 2 disponible")
except ImportError:
    attn_backend = "eager"
    print("⚠ Usando atención estándar")

# Cargar modelo
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

# Compilación opcional
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
        # GPU
        gpu_usage = torch.cuda.utilization(0) if torch.cuda.is_available() else 0
        gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_memory_reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
        gpu_temp = 0  # Requeriría pynvml para temperatura real
        
        # CPU y RAM
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram_usage_mb = psutil.virtual_memory().used / (1024 ** 2)
        
        return {
            "gpu_usage_percent": round(gpu_usage, 2),
            "gpu_memory_mb": round(gpu_memory_mb, 2),
            "gpu_memory_reserved_mb": round(gpu_memory_reserved_mb, 2),
            "cpu_usage_percent": round(cpu_usage, 2),
            "ram_usage_mb": round(ram_usage_mb, 2),
            "temperature_gpu_c": gpu_temp
        }
    except Exception as e:
        return {}

def calcular_hallucination_score(response: str) -> float:
    """
    Calcula un score aproximado de alucinación basado en patrones.
    Rango: 0.0 (sin alucinación) a 1.0 (alta alucinación)
    """
    score = 0.0
    response_lower = response.lower()
    
    # Patrones que indican posible alucinación
    patrones_negativos = [
        "no puedo", "lo siento", "no tengo acceso", 
        "no puedo ver", "no es posible", "no dispongo"
    ]
    
    # Patrones de incertidumbre excesiva
    patrones_incertidumbre = [
        "probablemente", "quizás", "tal vez", "posiblemente",
        "podría ser", "es posible que"
    ]
    
    # Respuestas muy cortas pueden ser evasivas
    if len(response.split()) < 10:
        score += 0.3
    
    # Negaciones y rechazos
    for patron in patrones_negativos:
        if patron in response_lower:
            score += 0.4
            break
    
    # Incertidumbre excesiva
    count_incertidumbre = sum(1 for patron in patrones_incertidumbre if patron in response_lower)
    if count_incertidumbre > 3:
        score += 0.2
    
    # Repeticiones excesivas (posible loop)
    palabras = response.split()
    if len(palabras) > 20:
        palabras_unicas = len(set(palabras))
        ratio_repeticion = palabras_unicas / len(palabras)
        if ratio_repeticion < 0.5:
            score += 0.3
    
    return min(round(score, 2), 1.0)

def calcular_toxicity_score(response: str) -> float:
    """
    Calcula un score básico de toxicidad.
    En producción, usar modelos especializados como Detoxify.
    """
    palabras_toxicas = [
        "idiota", "estúpido", "tonto", "inútil", 
        "basura", "porquería", "mierda"
    ]
    
    response_lower = response.lower()
    toxicidad = sum(1 for palabra in palabras_toxicas if palabra in response_lower)
    
    return min(round(toxicidad * 0.2, 2), 1.0)

def generar_request_id():
    """Genera un ID único para la request"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = f"{np.random.randint(1000, 9999)}"
    return f"req_{timestamp}_{random_suffix}"

def generar_response_id():
    """Genera un ID único para la respuesta"""
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
    
    Returns:
        str: Ruta del archivo JSON creado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultado_{timestamp}_{metricas['comportamiento']['response_id']}.json"
    filepath = OUTPUT_DIR / filename
    
    resultado_completo = {
        "metadata": {
            "video_path": str(video_path),
            "pregunta": pregunta,
            "modelo": "VideoLLaMA3-7B",
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
# FUNCIÓN DE INFERENCIA CON MÉTRICAS
# ---------------------------------------------
def analizar_video(
    video_path: str,
    pregunta: str,
    fps: float = 0.5,
    max_frames: int = 24,
    max_new_tokens: int = 256,
    use_sampling: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    mostrar_metricas: bool = True,
    guardar_json: bool = True
):
    """
    Analiza un video y responde preguntas en español con métricas completas.
    
    Args:
        video_path: Ruta al video
        pregunta: Pregunta sobre el video
        fps: Frames por segundo a extraer
        max_frames: Máximo de frames a procesar
        max_new_tokens: Tokens máximos en la respuesta
        use_sampling: Si usar sampling (True) o greedy (False)
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
    
    # Timestamp inicial
    timestamp_inicio = datetime.now()
    tiempo_inicio = time.time()
    
    # Métricas iniciales del sistema
    metricas_inicio = obtener_metricas_sistema()
    memoria_gpu_inicio = torch.cuda.memory_allocated(0) / (1024 ** 2)
    
    # System prompt generalizado
    conversation = [
        {
            "role": "system",
            "content": """Eres un experto agrónomo. Analiza videos e identifica con precisión. IMPORTANTE: Siempre responde en español, de forma clara y profesional."""
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
    
    if mostrar_metricas:
        print(f">>> Procesando video: {max_frames} frames @ {fps} fps")
    
    # Procesamiento
    tiempo_procesamiento_inicio = time.time()
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    tiempo_procesamiento = (time.time() - tiempo_procesamiento_inicio) * 1000
    
    # Contar tokens de entrada
    tokens_input = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
    
    # Mover a GPU
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    if mostrar_metricas:
        print(">>> Generando respuesta en español...")
    
    # Configuración de generación
    if use_sampling:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.15,
            "use_cache": True,
        }
    else:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.15,
            "use_cache": True,
        }
    
    # Generación
    tiempo_generacion_inicio = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_config)
    tiempo_generacion = (time.time() - tiempo_generacion_inicio) * 1000
    
    # Contar tokens de salida
    tokens_output = output_ids.shape[1] - tokens_input
    
    # Memoria GPU después de generación
    memoria_gpu_final = torch.cuda.memory_allocated(0) / (1024 ** 2)
    
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
    
    # Tiempo total
    tiempo_total = (time.time() - tiempo_inicio) * 1000
    latency_ms = round(tiempo_total, 2)
    
    # Throughput (tokens por segundo)
    throughput_tps = round(tokens_output / (tiempo_generacion / 1000), 2) if tiempo_generacion > 0 else 0
    
    # Métricas finales del sistema
    metricas_final = obtener_metricas_sistema()
    
    # Análisis de la respuesta
    hallucination_score = calcular_hallucination_score(response)
    toxicity_score = calcular_toxicity_score(response)
    safety_check = toxicity_score < 0.3
    bias_flag = False
    
    # ============================================================
    # CONSTRUCCIÓN DE MÉTRICAS
    # ============================================================
    metricas_comportamiento = {
        "response_id": response_id,
        "request_id": request_id,
        "model_name": "VideoLLaMA3-7B",
        "source_dataset": "Video-LLaMA-Dataset",
        "prompt_template": "agricultural_video_analysis_v1",
        "safety_check_passed": safety_check,
        "bias_flag": bias_flag,
        "toxicity_score": toxicity_score,
        "hallucination_score": hallucination_score
    }
    
    metricas_tiempo = {
        "tiempo_total_ms": latency_ms,
        "tiempo_procesamiento_ms": round(tiempo_procesamiento, 2),
        "tiempo_generacion_ms": round(tiempo_generacion, 2),
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
        "model_name": "VideoLLaMA3-7B",
        "mode": "inference",
        "temperature": temperature if use_sampling else 0.0,
        "context_length": 4096,
        "prompt_tokens": tokens_input,
        "response_tokens": tokens_output,
        "max_new_tokens": max_new_tokens,
        "fps": fps,
        "max_frames": max_frames
    }
    
    metricas_sistema = {
        "timestamp": datetime.now().isoformat(),
        "gpu_usage_percent": metricas_final.get("gpu_usage_percent", 0),
        "gpu_memory_mb": metricas_final.get("gpu_memory_mb", 0),
        "gpu_memory_reserved_mb": metricas_final.get("gpu_memory_reserved_mb", 0),
        "cpu_usage_percent": metricas_final.get("cpu_usage_percent", 0),
        "ram_usage_mb": metricas_final.get("ram_usage_mb", 0),
        "temperature_gpu_c": metricas_final.get("temperature_gpu_c", 0),
        "latency_ms": latency_ms,
        "tokens_processed": tokens_input + tokens_output,
        "throughput_tps": throughput_tps,
        "model_name": "VideoLLaMA3-7B",
        "model_version": "7B",
        "device": obtener_nombre_gpu(),
        "batch_size": 1,
        "dtype": "bfloat16",
        "attn_implementation": attn_backend
    }
    
    metricas = {
        "comportamiento": metricas_comportamiento,
        "tiempo": metricas_tiempo,
        "modelo": metricas_modelo,
        "sistema": metricas_sistema
    }
    
    # ============================================================
    # IMPRESIÓN DE MÉTRICAS
    # ============================================================
    if mostrar_metricas:
        print("\n" + "="*70)
        print("📊 MÉTRICAS DE INFERENCIA - VideoLLaMA3")
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
    
    # ============================================================
    # GUARDAR JSON
    # ============================================================
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
        fps=0.5,
        max_frames=16,
        max_new_tokens=200,
        use_sampling=False,
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
    time.sleep(1)
    
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
- Nivel de severidad observado
- Estado general de las plantas""",
        fps=1,
        max_frames=24,
        max_new_tokens=400,
        use_sampling=True,
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
    time.sleep(1)
    
    # ============================================================
    # EJEMPLO 3: IDENTIFICACIÓN ESPECÍFICA
    # ============================================================
    print("\n" + "🌿"*35)
    print("EJEMPLO 3: IDENTIFICACIÓN DE PLAGAS O ENFERMEDADES")
    print("🌿"*35)
    
    respuesta3, metricas3, json3 = analizar_video(
        video_path=video_path,
        pregunta="¿Qué tipo de plagas o enfermedades se observan en las plantas? Describe los síntomas visibles.",
        fps=0.75,
        max_frames=20,
        max_new_tokens=300,
        use_sampling=True,
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
        ("Rápido (16 frames)", metricas1),
        ("Detallado (24 frames)", metricas2),
        ("Específico (20 frames)", metricas3)
    ]
    
    for nombre, metricas in ejemplos:
        print(f"\n{nombre}:")
        print(f"  • Tiempo total: {metricas['tiempo']['tiempo_total_ms']:.0f} ms")
        print(f"  • Tiempo procesamiento: {metricas['tiempo']['tiempo_procesamiento_ms']:.0f} ms")
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