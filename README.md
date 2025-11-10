## Sistema RAG para Cacao

Asistente cognitivo para extensionistas y productores de cacao basado en recuperación aumentada con generación (RAG), embeddings FAISS y despliegues locales con Ollama + dspy.

---

## Requisitos

- `python 3.10+`
- GPU NVIDIA opcional pero recomendada (para `torch`, `paddleocr`, modelos vision)
- Servidor Ollama en `http://localhost:11434` con el modelo `mistral` disponible
- Dependencias del proyecto: `pip install -r requirements.txt`
- Recursos externos:
  - Índices FAISS y pickles de documentos en `./04-Prototipar/4.1-Fuentes de Datos/4.1.1-GET_BAC_PUSH_BloB/downloads/`
  - Subconjunto especializado en `./llm_cacao-dragro/`

> Asegúrate de montar las carpetas de datos con las rutas originales o ajusta los argumentos `--path_root` en cada script.

---

## Preparación Rápida

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
ollama pull mistral
```

---

## Scripts Principales (orden sugerido de uso)

1. ### `RAG_main.py`
   - Construye el corpus completo del SNIA/BAC a partir de PDFs (`pdf/`), genera pares pregunta-respuesta vía LLM y crea los índices FAISS y pickles.
   - Entrada esperada: árbol `./04-Prototipar/.../downloads/{container}/pdf/*.pdf`.
   - Ejecutar:
     ```bash
     python RAG_main.py --path_root /ruta/a/descargas --remove_headers True --flag_rewrite True --skip_savedata False
     ```
   - Resultado: `dataQA.jsonl`, `university_index.faiss`, `university_docs.pkl`.

2. ### `Rag_main_small.py`
   - Flujo equivalente focalizado en los documentos especializados (`llm_cacao-dragro`). Procesa `.docx`, extrae párrafos/tablas y crea `profiles_index.faiss` + `profiles_docs.pkl`.
   - Ejecutar:
     ```bash
     python Rag_main_small.py --path_root /ruta/a/llm_cacao-dragro --flag_rewrite True
     ```

3. ### `RAG_inference.py`
   - Realiza consultas RAG sobre el corpus general usando el índice `university_index.faiss`.
   - Requiere Ollama activo y archivos generados en el paso 1.
   - Ejecutar:
     ```bash
     python RAG_inference.py --path_root /ruta/a/descargas --question "¿Cuál es la cadena de valor del cacao?"
     ```

4. ### `RAG_inference_small.py`
   - Variante de inferencia sobre el corpus especializado (`profiles_index.faiss`).
   - Ejecutar:
     ```bash
     python RAG_inference_small.py --path_root /ruta/a/llm_cacao-dragro --question "¿Síntomas de deficiencia de potasio?"
     ```

5. ### `faster_RAG_inference.py`
   - Igual que `RAG_inference.py` pero con parámetros de Ollama optimizados para latencia y throughput (control de `num_ctx`, `num_predict`, `num_thread`, etc.).
   - Ejecutar:
     ```bash
     python faster_RAG_inference.py --path_root /ruta/a/descargas --question "¿Cuáles son las principales zonas productoras de cacao en Colombia?"
     ```

6. ### `RAG_gradio_app.py`
   - Despliega una interfaz web (Gradio) con cambio dinámico de corpus, monitoreo del estado del modelo y prompt del sistema.
   - Requiere las rutas de corpus configuradas en `CORPUS_CONFIG` y un servidor Ollama activo.
   - Ejecutar:
     ```bash
     python RAG_gradio_app.py
     ```
   - Accede vía navegador a la URL que muestra Gradio (por defecto `http://127.0.0.1:7860`).

7. ### `showclean_QA.py`
   - Toma muestras aleatorias del `.jsonl` de QA y exporta los pares a un `.docx` para revisión manual.
   - Ejecutar:
     ```bash
     python showclean_QA.py ./ruta/dataQA.jsonl --output ./ruta/sampled_qas.docx --num 50
     ```

8. ### `Llava_LDD.py`
   - Demo de diagnóstico de enfermedades de hojas con `LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection`.
   - Requiere GPU (`torch.float16`) y una imagen accesible localmente o por URL.
   - Ajusta `image_file` antes de ejecutar:
     ```bash
     python Llava_LDD.py
     ```

---

## Módulos de Soporte

- `utils.py`: extracción de texto/tablas (pdfplumber, Tesseract, PaddleOCR), limpieza, embedding con `SentenceTransformer`, cadenas RAG (`RerankedFaissRetriever`, `UniversityRAGChain`).
- `signatures.py`: clases de `dspy.Signature` para estructurar prompts y validadores de QA (e.g., `QAGenerator_safe`).
- `faster_RAG_inference.py` reutiliza configuraciones de `utils.py`.

---

## Flujo Sugerido

- Generar o actualizar corpus (`RAG_main.py` y/o `Rag_main_small.py`).
- Validar QA con `showclean_QA.py`.
- Levantar inferencia puntual (`RAG_inference*.py` o `faster_RAG_inference.py`).
- Para demos interactivos, usar `RAG_gradio_app.py`.
- Para visión por computadora de hojas usar `Llava_LDD.py`.

---

## Contribuir

- Crea una rama: `git checkout -b feature/nueva-funcionalidad`
- Aplica cambios y pruebas locales.
- Envía PR con descripción del flujo ejecutado y artefactos generados.

---

## Licencia

Actualmente el repositorio no incluye archivo de licencia.

---

## Agradecimientos

Equipo de `Agrosavia` y colaboradores del SNIA por los insumos técnicos y documentales que alimentan el sistema.

---
