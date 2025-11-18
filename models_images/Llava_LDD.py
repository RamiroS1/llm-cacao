"""
Módulo LLaVA Optimizado para Dr. agro
Especialidad: Detección de Enfermedades en Plantas
Características:
- Formato de salida Markdown
- Filtro rápido de conversación 
- Gestión manual de memoria VRAM
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import gc
import re

class LlavaPlantDiseaseDetector:
    def __init__(self, model_id="YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"):
        """
        Inicializa el detector con configuración para respuestas científicas y visuales.
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        # ------------------------------------------------------------------
        # DICCIONARIOS PARA FILTRO RÁPIDO (FAST PATH)
        # Evitan usar la GPU para saludos simples
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
        Carga el modelo LLaVA en memoria (float16 para ahorrar VRAM)
        """
        try:
            print(f">>> Cargando modelo LLaVA (Modo Experto) desde {self.model_id}...")
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True,
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self.is_loaded = True
            print("✓ Modelo LLaVA cargado exitosamente")
            return True, "Modelo LLaVA cargado correctamente"
            
        except Exception as e:
            self.is_loaded = False
            error_msg = f"Error al cargar LLaVA: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg

    def _check_conversational_intent(self, text):
        """
        Ruta Rápida (CPU): Detecta intenciones simples antes de invocar al modelo pesado.
        Retorna: str (respuesta) o None (si requiere análisis visual)
        """
        if not text: return None
        
        # Limpieza: minúsculas y quitar puntuación básica
        clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = clean_text.split()
        
        # Si el mensaje es largo (> 5 palabras), asumimos que es una consulta técnica
        # Ejemplo: "Hola, ¿por qué mi planta tiene hojas amarillas?" -> Pasa al modelo
        if len(words) > 5:
            return None
            
        # Verificar saludos
        if any(word in self.GREETINGS for word in words):
            return (
                "👋 **¡Hola! Soy Dr. agro.**\n\n"
                "Estoy listo para ayudarte. Por favor, sube una imagen clara de la hoja, "
                "tallo o fruto afectado y haré un diagnóstico técnico inmediato."
            )
            
        # Verificar despedidas
        if any(word in self.FAREWELLS for word in words):
            return (
                "🤝 **¡Hasta luego!**\n\n"
                "Recuerda monitorear tus cultivos frecuentemente. "
                "Estaré aquí si necesitas otra opinión técnica."
            )
            
        return None
    
    def analyze_image(self, image, question=None, max_new_tokens=750):
        """
        Analiza una imagen aplicando el System Prompt científico y formato Markdown.
        """
        if not self.is_loaded:
            return False, "Error: Modelo no cargado. Llama a load_model() primero."
        
        # ---------------------------------------------------------
        # 1. FAST PATH: Verificar si es solo un saludo
        # ---------------------------------------------------------
        intent_response = self._check_conversational_intent(question)
        if intent_response:
            return True, intent_response
        
        # ---------------------------------------------------------
        # 2. SLOW PATH: Análisis Neuronal Profundo
        # ---------------------------------------------------------
        try:
            # Procesar imagen
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Definir consulta base si está vacía
            user_query = question if question else "Realiza un diagnóstico técnico completo de esta planta."
            
            # ---------------------------------------------------------
            # PROMPT DE INGENIERÍA: Estructura Gemini + Persona Experta
            # ---------------------------------------------------------
            prompt_structure = (
                "TAREA: Actúa como un Fitopatólogo e Ingeniero Agrónomo Senior. Analiza la IMAGEN ADJUNTA.\n"
                "IDIOMA DE SALIDA: Español (Estrictamente).\n"
                "FORMATO: Usa Markdown para estructurar la respuesta visualmente. Sigue esta plantilla:\n\n"
                
                "### 🔬 Diagnóstico Identificado\n"
                "**Nombre Común:** [Nombre de la enfermedad/plaga]\n"
                "**Nombre Científico:** *[Género especie]* (Taxonomía)\n"
                "**Nivel de Confianza:** [Alto/Medio/Bajo] basado en signos visuales.\n\n"
                
                "### 🍂 Sintomatología Observada\n"
                "Describe técnicamente los signos patológicos visibles en la imagen:\n"
                "* [Signo visual 1: ej. Clorosis, Necrosis, Halo]\n"
                "* [Signo visual 2: ej. Patrón de manchas, Esporulación]\n\n"
                
                "### 🦠 Etiología (Causa Probable)\n"
                "**Tipo de Agente:** [Hongo / Bacteria / Virus / Insecto / Nutricional]\n"
                "[Explicación breve del mecanismo de acción del patógeno]\n\n"
                
                "### 🛡️ Recomendaciones de Manejo Integrado\n"
                "1. [Acción Cultural: ej. Poda, Riego]\n"
                "2. [Acción Química/Biológica sugerida]\n"
                "3. [Medida preventiva]\n\n"
                
                f"CONSULTA ESPECÍFICA DEL USUARIO: {user_query}"
            )

            # Construcción de la conversación
            # NOTA: Ponemos la imagen PRIMERO para asegurar atención visual
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_structure},
                    ],
                },
            ]
            
            prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                images=image, 
                text=prompt, 
                return_tensors='pt'
            ).to(self.device, torch.float16)
            
            # Generación con parámetros ajustados para rigor científico
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True,
                    temperature=0.2,       # Baja creatividad para seguir la plantilla
                    top_p=0.9,
                    repetition_penalty=1.15 # Evitar bucles en descripciones largas
                )
            
            # Decodificación y Limpieza
            response = self.processor.decode(output[0][2:], skip_special_tokens=True)
            
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            # Verificación de fallo de visión (común en LLaVA)
            if "proporcióname una imagen" in response.lower() or "no veo imagen" in response.lower():
                 return True, "⚠️ **Atención:** El modelo no pudo enfocar correctamente la imagen. Por favor intenta:\n1. Recortar la imagen para centrar la hoja/fruto.\n2. Usar una foto con mejor iluminación."
            
            # Liberar tensores de entrada inmediatamente
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True, response
            
        except Exception as e:
            error_msg = f"Error crítico en análisis visual: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def unload_model(self):
        """
        Descarga el modelo completamente para liberar VRAM para otros procesos (Video/RAG)
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Forzar recolección de basura de Python y CUDA
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            print("✓ Modelo LLaVA descargado de memoria")
            return True, "Modelo descargado correctamente"
            
        except Exception as e:
            error_msg = f"Error al descargar modelo: {str(e)}"
            return False, error_msg

# ==========================================
# BLOQUE DE PRUEBA (Solo se ejecuta si corres este archivo directamente)
# ==========================================
if __name__ == "__main__":
    print("--- Test de Llava_LDD ---")
    detector = LlavaPlantDiseaseDetector()
    
    # Prueba de Fast Path (Sin cargar modelo)
    print("Probando saludo:", detector._check_conversational_intent("Hola buenos dias"))
    
    # Cargar y Probar IA
    success, msg = detector.load_model()
    if success:
        # Reemplaza con una ruta real de tu PC para probar
        ruta_imagen = "test_leaf.jpg" 
        try:
            # Crear imagen dummy si no existe para evitar error en test
            img = Image.new('RGB', (100, 100), color = 'green')
            
            print("\nAnalizando imagen simulada...")
            ok, resp = detector.analyze_image(img, "Diagnostica esto")
            print("\nRESPUESTA GENERADA:\n")
            print(resp)
        except Exception as e:
            print(e)
        
        detector.unload_model()