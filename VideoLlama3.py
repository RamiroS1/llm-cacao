import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# ---------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------
device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

print(">>> Cargando modelo optimizado para GPU de 24GB...")

# Intentar usar Flash Attention 2 si está disponible
try:
    attn_backend = "flash_attention_2"
except:
    attn_backend = "eager"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",       # permite offload automático si se llena la VRAM
    attn_implementation=attn_backend,
)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# ---------------------------------------------
# CONFIGURACIÓN DEL VIDEO
# ---------------------------------------------
video_path = "./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads/cacao20250916133108/video/Ver_video_36367.mp4"


conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": {
                    "video_path": video_path,
                    "fps": 1,
                    "max_frames": 32   # 🔥 muy importante para evitar OOM
                }
            },
            {
                "type": "text",
                "text": "Que plagas aparecen en el video?"
            },
        ]
    },
]

# ---------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------
print(">>> Procesando entrada...")
inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Mover a GPU solo lo necesario
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device)

# Reducir VRAM (muy importante)
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

# ---------------------------------------------
# GENERACIÓN CON BAJO CONSUMO DE VRAM
# ---------------------------------------------
print(">>> Generando respuesta (modo optimizado)...")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=False,   # 🔥 evita ~4GB de VRAM extra
    )

# ---------------------------------------------
# DECODIFICACIÓN
# ---------------------------------------------
response = processor.batch_decode(
    output_ids,
    skip_special_tokens=True
)[0].strip()

print("\n>>> RESPUESTA DEL MODELO:")
print(response)
