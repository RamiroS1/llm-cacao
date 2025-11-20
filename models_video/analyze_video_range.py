import os
import torch
import json
import gc
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import AutoModelForCausalLM, AutoProcessor
from uuid import uuid4

# ==============================================================================
# 1. CEREBRO VISUAL (VIDEOLLAMA)
# ==============================================================================
class VisualBrain:
    def __init__(self, model_path="DAMO-NLP-SG/VideoLLaMA3-7B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self):
        print(f"👁️ Cargando VideoLLaMA3 en {self.device}...")
        try:
            attn = "flash_attention_2" if "cuda" in self.device else "eager"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="auto", attn_implementation=attn
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            return True
        except Exception as e:
            print(f"❌ Error carga visual: {e}")
            return False

    def analyze(self, video_path):
        prompt = (
            "ROL: Analista técnico agrícola.\n"
            "TAREA: Describe visualmente el clip.\n"
            "ENFOQUE: ¿Se ven cultivos (cuáles), primeros planos de personas hablando, diapositivas/texto o maquinaria?\n"
            "FORMATO: Español. Sé muy conciso."
        )

        conversation = [
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 32}},
                {"type": "text", "text": prompt},
            ]}
        ]

        try:
            inputs = self.processor(conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.2)
            
            res = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            final_text = res.split("assistant")[-1].strip() if "assistant" in res else res
            return final_text
        except Exception as e:
            print(f"⚠️ Error inferencia: {e}")
            return "[Error Análisis Visual]"

# ==============================================================================
# 2. HERRAMIENTA DE CORTE
# ==============================================================================
def create_temp_clip(input_path, start, end, temp_output):
    try:
        duration = end - start
        if duration < 1.0: end = start + 1.0
            
        with VideoFileClip(input_path) as video:
            if start > video.duration: return False
            real_end = min(end, video.duration)
            
            if hasattr(video, 'subclip'): clip = video.subclip(start, real_end)
            else: clip = video.subclipped(start, real_end)
            
            clip.write_videofile(temp_output, codec="libx264", audio=False, logger=None)
        return True
    except Exception as e:
        print(f"❌ Error cortando clip: {e}")
        return False

# ==============================================================================
# 3. UTILIDAD: MAPEO DE VIDEOS
# ==============================================================================
def map_videos_in_folder(folder_path):
    """
    Crea un diccionario {video_id: ruta_completa}
    Maneja archivos tipo '1_48oBYG8AKhU.mp4' extrayendo el ID real.
    """
    video_map = {}
    if not os.path.exists(folder_path):
        return video_map

    print(f"📂 Escaneando carpeta de videos: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            # Lógica: Dividir por el primer guion bajo '_'
            # Ejemplo: "1_48oBYG8AKhU.mp4" -> ["1", "48oBYG8AKhU.mp4"]
            parts = filename.split('_', 1)
            
            if len(parts) > 1:
                # Caso con prefijo (1_ID.mp4)
                vid_id = parts[1].replace(".mp4", "")
            else:
                # Caso sin prefijo (ID.mp4)
                vid_id = filename.replace(".mp4", "")
            
            full_path = os.path.join(folder_path, filename)
            video_map[vid_id] = full_path
            # print(f"   Found: {vid_id} -> {filename}") # Debug opcional
            
    print(f"✅ Se encontraron {len(video_map)} videos para procesar.")
    return video_map

# ==============================================================================
# 4. PROCESAMIENTO MASIVO
# ==============================================================================
def process_jsonl(input_file, output_file, video_folder_path, visual_brain):
    
    # 1. Crear mapa de videos disponibles
    available_videos = map_videos_in_folder(video_folder_path)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        infile.seek(0) # Volver al inicio

        print(f"\n🚀 Iniciando procesamiento de {total_lines} registros JSONL...\n")

        for line_num, line in enumerate(infile):
            try:
                data = json.loads(line)
                video_id = data.get("video_id")
                segments = data.get("segments", [])
                
                # Buscar si tenemos el video en nuestro mapa
                video_path = available_videos.get(video_id)

                # Si NO tenemos el video, guardamos la línea igual y saltamos
                if not video_path:
                    # print(f"⚠️ Video ID {video_id} no encontrado en carpeta. Saltando.")
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                # Si SÍ tenemos el video, procesamos
                print(f"🎬 [{line_num+1}/{total_lines}] Procesando: {video_id} | Archivo: {os.path.basename(video_path)}")
                data["subtitle_source"] = "visual_ai_description" # Actualizar metadata

                for i, seg in enumerate(segments):
                    start = seg.get("time_start", 0)
                    end = seg.get("time_end", 0)
                    
                    temp_vid = f"temp_{uuid4().hex[:6]}.mp4"
                    
                    # Feedback visual mínimo
                    print(f"   🔹 Seg {i+1}/{len(segments)} ({start}s)", end="\r", flush=True)

                    if create_temp_clip(video_path, start, end, temp_vid):
                        # Análisis
                        description = visual_brain.analyze(temp_vid)
                        seg["text"] = description
                        
                        # Limpieza archivo
                        if os.path.exists(temp_vid): os.remove(temp_vid)
                        
                        # Limpieza memoria VRAM periódica (cada 10 segmentos)
                        if i % 10 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                    else:
                        seg["text"] = "[Error Clip]"

                print(f"   ✅ Video {video_id} completado.")
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                outfile.flush()

            except json.JSONDecodeError:
                print(f"❌ Error decodificando línea {line_num}")

# ==============================================================================
# 5. EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    # Rutas configuradas
    INPUT_JSON = "../json/linkata_videos_whisper.jsonl"
    OUTPUT_JSON = "linkata_videos_visual_completo.jsonl"
    VIDEOS_DIR = "./videos" # Carpeta donde están los mp4

    # Validaciones
    if not os.path.exists(INPUT_JSON):
        exit(f"❌ No encuentro el JSON de entrada: {os.path.abspath(INPUT_JSON)}")
    if not os.path.exists(VIDEOS_DIR):
        exit(f"❌ No encuentro la carpeta de videos: {os.path.abspath(VIDEOS_DIR)}")

    # Cargar Modelo
    brain = VisualBrain()
    if brain.load():
        # Ejecutar proceso
        process_jsonl(INPUT_JSON, OUTPUT_JSON, VIDEOS_DIR, brain)
        print(f"\n🎉 TODO LISTO. Resultado guardado en: {OUTPUT_JSON}")