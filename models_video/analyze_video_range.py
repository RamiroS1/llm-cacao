import os
import torch
import argparse
import whisper
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import AutoModelForCausalLM, AutoProcessor
from uuid import uuid4
import gc

# ==============================================================================
# 1. CEREBRO AUDITIVO (WHISPER)
# ==============================================================================
class AudioBrain:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
    
    def transcribe(self, audio_path):
        print(f"👂 Escuchando audio (Modelo Whisper {self.model_size})...")
        try:
            self.model = whisper.load_model(self.model_size)
            result = self.model.transcribe(audio_path)
            
            # Liberar memoria inmediatamente después de usar
            del self.model
            torch.cuda.empty_cache()
            
            return result["text"]
        except Exception as e:
            return f"Error en transcripción: {e}"

# ==============================================================================
# 2. CEREBRO VISUAL (VIDEOLLAMA)
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
            "ROL: Analista técnico.\n"
            "TAREA: Describe qué se ve en la imagen.\n"
            "ENFOQUE: Cultivos, personas, diapositivas o maquinaria.\n"
            "FORMATO: Español, conciso."
        )

        conversation = [
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 32}},
                {"type": "text", "text": prompt},
            ]}
        ]

        inputs = self.processor(conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.2)
        
        res = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return res.split("assistant")[-1].strip() if "assistant" in res else res

# ==============================================================================
# 3. HERRAMIENTAS DE CORTE
# ==============================================================================
def process_media(input_path, start, end, temp_video, temp_audio):
    try:
        print(f"✂️ Procesando clip ({start}-{end}s)...")
        with VideoFileClip(input_path) as video:
            # Fix versiones MoviePy
            if hasattr(video, 'subclip'): clip = video.subclip(start, end)
            else: clip = video.subclipped(start, end)
            
            # Guardar video (sin audio para VideoLLaMA - más rápido)
            clip.write_videofile(temp_video, codec="libx264", audio=False, logger=None)
            
            # Guardar audio solo (para Whisper)
            clip.audio.write_audiofile(temp_audio, logger=None)
        return True
    except Exception as e:
        print(f"❌ Error procesando medios: {e}")
        return False

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.video): exit("❌ No encuentro el video")

    # Nombres temporales únicos
    uid = uuid4().hex
    temp_vid = f"temp_v_{uid}.mp4"
    temp_aud = f"temp_a_{uid}.mp3"

    if process_media(args.video, args.start, args.end, temp_vid, temp_aud):
        
        # 1. ANÁLISIS DE AUDIO (Whisper)
        audio_brain = AudioBrain(model_size="base") # Usa "small" o "medium" para más precisión
        texto_transcrito = audio_brain.transcribe(temp_aud)

        # 2. ANÁLISIS VISUAL (VideoLLaMA)
        visual_brain = VisualBrain()
        if visual_brain.load():
            descripcion_visual = visual_brain.analyze(temp_vid)

            # 3. RESULTADO FINAL
            print("\n" + "="*60)
            print("📢 LO QUE SE ESCUCHA (Transcripción Literal):")
            print(texto_transcrito.strip()) 
            print("-" * 60)
            print("👀 LO QUE SE VE (Análisis Visual):")
            print(descripcion_visual)
            print("="*60)

    # Limpieza
    for f in [temp_vid, temp_aud]:
        if os.path.exists(f): os.remove(f)