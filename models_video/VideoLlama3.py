"""
Módulo VideoLLaMA3 Optimizado para Dr. agro
Especialidad: Análisis Dinámico de Cultivos y Plagas
Características:
- Formato de salida Markdown (Estilo Gemini)
- Filtro rápido de conversación (ahorro de GPU)
- Gestión de memoria VRAM y Flash Attention
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import gc
import re

class VideoLlamaAgriculturalAnalyzer:
    def __init__(self, model_path="DAMO-NLP-SG/VideoLLaMA3-7B"):
        """
        Inicializa el analizador de videos con perfil de Agrónomo Senior.
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.attn_backend = "eager"
        
        # ------------------------------------------------------------------
        # DICCIONARIOS PARA FILTRO RÁPIDO (FAST PATH)
        # ------------------------------------------------------------------
        self.GREETINGS = {
            'hola', 'buenos dias', 'buenas tardes', 'buenas noches', 
            'hi', 'hello', 'holi', 'que tal', 'saludos'
        }
        self.FAREWELLS = {
            'adios', 'chao', 'hasta luego', 'bye', 'nos vemos', 
            'gracias', 'muchas gracias', 'ok gracias'
        }
    
    def load_model(self):
        """
        Carga el modelo VideoLLaMA3 con optimizaciones de memoria.
        """
        try:
            print(f">>> Cargando VideoLLaMA3 (Modo Experto) desde {self.model_path}...")
            
            # Detección de Flash Attention para mayor velocidad
            try:
                from flash_attn import flash_attn_func
                self.attn_backend = "flash_attention_2"
                print("✓ Flash Attention 2 activado")
            except ImportError:
                self.attn_backend = "eager"
                print("⚠ Flash Attention no detectado. Usando modo estándar (más lento).")
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, # Mejor precisión/memoria que float16 para VideoLLaMA
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

    def _check_conversational_intent(self, text):
        """
        Ruta Rápida (CPU): Detecta intenciones simples.
        """
        if not text: return None
        
        clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = clean_text.split()
        
        if len(words) > 5:
            return None
            
        if any(word in self.GREETINGS for word in words):
            return (
                "👋 **¡Hola! Soy el especialista en video de Dr. agro.**\n\n"
                "Puedo analizar grabaciones de tus cultivos para detectar plagas en movimiento, "
                "problemas de riego o estado general del lote. Sube tu video y empezaré el análisis."
            )
            
        if any(word in self.FAREWELLS for word in words):
            return (
                "🤝 **¡Hasta pronto!**\n\n"
                "El monitoreo constante es clave. Estaré aquí si tienes más videos para analizar."
            )
            
        return None
    
    def analyze_video(
        self,
        video_path,
        question="¿Qué problemas fitosanitarios observas en el cultivo?",
        fps=0.5,
        max_frames=16, # Aumentado ligeramente para captar mejor el movimiento
        max_new_tokens=512, # Aumentado para permitir la plantilla Markdown
        temperature=0.2 # Temperatura baja para rigor científico
    ):
        """
        Analiza video aplicando Prompt Científico y Formato Markdown.
        """
        if not self.is_loaded:
            return False, "Error: Modelo no cargado. Llama a load_model() primero."
        
        if not Path(video_path).exists():
            return False, f"Error: Video no encontrado: {video_path}"

        # ---------------------------------------------------------
        # 1. FAST PATH
        # ---------------------------------------------------------
        intent_response = self._check_conversational_intent(question)
        if intent_response:
            return True, intent_response
        
        try:
            # ---------------------------------------------------------
            # 2. PROMPT ENGINEERÍA PARA VIDEO (Estilo Gemini)
            # ---------------------------------------------------------
            prompt_structure = (
                "TAREA: Actúa como Ingeniero Agrónomo Especialista en Monitoreo. Analiza el VIDEO ADJUNTO.\n"
                "IDIOMA: Español (Estrictamente).\n"
                "FORMATO: Usa Markdown para estructurar la respuesta. Sigue esta plantilla:\n\n"
                
                "### 📹 Observación Dinámica\n"
                "Describe el escenario general del video (tipo de cultivo, condiciones ambientales, movimiento observado):\n"
                "* [Observación 1]\n"
                "* [Observación 2]\n\n"

                "### 🔬 Diagnóstico Presuntivo\n"
                "**Problema Identificado:** [Nombre de la plaga/enfermedad/estrés]\n"
                "**Evidencia Visual:** Describe qué fotogramas o movimientos confirman el diagnóstico.\n\n"
                
                "### 🦠 Análisis Técnico\n"
                "**Posible Causa:** [Agente causal]\n"
                "**Severidad Aparente:** [Leve/Moderada/Severa] según la extensión vista en el video.\n\n"
                
                "### 🛡️ Recomendaciones de Manejo\n"
                "1. [Acción inmediata]\n"
                "2. [Medida preventiva a largo plazo]\n\n"
                
                f"PREGUNTA ESPECÍFICA: {question}"
            )

            conversation = [
                {
                    "role": "system",
                    "content": "Eres un asistente agrícola experto de Agrosavia. Responde siempre en español usando formato Markdown."
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
                            "text": prompt_structure
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
            
            # Mover a GPU y asegurar tipos de datos correctos
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            # Configuración de generación estricta
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature, # Baja creatividad para respetar formato
                "top_p": 0.9,
                "repetition_penalty": 1.2, # Evita repetir frases en descripciones largas
                "use_cache": True,
            }
            
            # Generar
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_config)
            
            # Limpiar tensores inmediatamente
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Decodificar
            response = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            # Limpieza de cadena (VideoLLaMA a veces deja basura del sistema al inicio)
            if "assistant" in response.lower():
                parts = response.split("assistant")
                response = parts[-1].strip()
                # Limpiar caracteres residuales comunes
                for prefix in [":", "\n", " ", "Sure", "Here"]:
                    response = response.lstrip(prefix)
            
            # Verificar si falló el idioma o la detección
            if "provide a video" in response.lower():
                return True, "⚠️ No pude procesar los cuadros del video correctamente. Intenta con un video más corto o con mejor iluminación."

            return True, response
            
        except Exception as e:
            error_msg = f"Error en análisis de video: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def unload_model(self):
        """
        Descarga el modelo de memoria para liberar recursos.
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

# ==========================================
# BLOQUE DE PRUEBA
# ==========================================
if __name__ == "__main__":
    print("--- Test VideoLLaMA3 Dr. agro ---")
    analyzer = VideoLlamaAgriculturalAnalyzer()
    
    # Test Fast Path
    print("Test Saludo:", analyzer._check_conversational_intent("Hola"))
    
    # Test Carga
    success, msg = analyzer.load_model()
    if success:
        print("Modelo listo. Para probar, descomenta las líneas de análisis abajo.")
        # analyzer.analyze_video("ruta_test.mp4", "Diagnostica esto")
        analyzer.unload_model()