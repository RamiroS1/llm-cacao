"""
Módulo VideoLLaMA3 para Análisis de Videos Agrícolas
Versión modular simplificada para integración con Streamlit
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import gc

class VideoLlamaAgriculturalAnalyzer:
    def __init__(self, model_path="DAMO-NLP-SG/VideoLLaMA3-7B"):
        """
        Inicializa el analizador de videos agrícolas con VideoLLaMA3
        
        Args:
            model_path: Ruta o ID del modelo en HuggingFace
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.attn_backend = "eager"
    
    def load_model(self):
        """
        Carga el modelo VideoLLaMA3 en memoria
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            print(f">>> Cargando modelo VideoLLaMA3 desde {self.model_path}...")
            
            # Configuración de atención optimizada
            try:
                from flash_attn import flash_attn_func
                self.attn_backend = "flash_attention_2"
                print("✓ Flash Attention 2 disponible")
            except ImportError:
                self.attn_backend = "eager"
                print("⚠ Usando atención estándar")
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                attn_implementation=self.attn_backend,
                use_cache=True,
            )
            
            self.model.eval()
            
            # Cargar processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.is_loaded = True
            print("✓ Modelo VideoLLaMA3 cargado exitosamente")
            return True, "Modelo VideoLLaMA3 cargado correctamente"
            
        except Exception as e:
            self.is_loaded = False
            error_msg = f"Error al cargar VideoLLaMA3: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def analyze_video(
        self,
        video_path,
        question="¿Qué problemas fitosanitarios observas en el cultivo?",
        fps=0.5,
        max_frames=16,
        max_new_tokens=256,
        use_sampling=False,
        temperature=0.7
    ):
        """
        Analiza un video y responde la pregunta en español
        
        Args:
            video_path: Ruta al archivo de video
            question: Pregunta sobre el video
            fps: Frames por segundo a extraer
            max_frames: Máximo de frames a procesar
            max_new_tokens: Máximo de tokens a generar
            use_sampling: Si usar sampling (True) o greedy (False)
            temperature: Control de creatividad
            
        Returns:
            tuple: (success: bool, response: str)
        """
        if not self.is_loaded:
            return False, "Error: Modelo no cargado. Llama a load_model() primero."
        
        if not Path(video_path).exists():
            return False, f"Error: Video no encontrado: {video_path}"
        
        try:
            # Preparar conversación
            conversation = [
                {
                    "role": "system",
                    "content": "Eres un experto agrónomo. Analiza videos e identifica con precisión. Responde en español."
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
                            "text": f"{question}\n\nResponde en español."
                        },
                    ]
                },
            ]
            
            # Procesar inputs
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Mover a GPU
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            # Configuración de generación
            if use_sampling:
                generation_config = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 0.9,
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
            
            # Generar respuesta
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_config)
            
            # Limpiar inputs
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Decodificar respuesta
            response = self.processor.batch_decode(
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
            
            return True, response
            
        except Exception as e:
            error_msg = f"Error en análisis: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def unload_model(self):
        """
        Descarga el modelo de memoria
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
            self.is_loaded = False
            print("✓ Modelo VideoLLaMA3 descargado de memoria")
            return True, "Modelo descargado correctamente"
            
        except Exception as e:
            error_msg = f"Error al descargar modelo: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar analizador
    analyzer = VideoLlamaAgriculturalAnalyzer()
    
    # Cargar modelo
    success, message = analyzer.load_model()
    print(message)
    
    if success:
        # Analizar video
        test_video = "./data/test_video.mp4"
        
        success, response = analyzer.analyze_video(
            video_path=test_video,
            question="¿Qué observas en el video?",
            fps=0.5,
            max_frames=16,
            max_new_tokens=256
        )
        
        if success:
            print(f"\n📋 RESPUESTA:")
            print("-" * 70)
            print(response)
            print("-" * 70)
        else:
            print(f"Error: {response}")
        
        # Descargar modelo
        analyzer.unload_model()