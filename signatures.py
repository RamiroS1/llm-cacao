import dspy
from typing import List
from typing_extensions import TypedDict

import re
import json

ALLOWED_QT = {"definicion","procedimiento","requisito","excepcion","numerica","fecha_plazo","comparativa","lista","condicional"}
ALLOWED_AT = {"extractiva","abstractive_breve"}
ALLOWED_DIFF = {"baja","media","alta"}

JSON_ONLY_INSTRUCTIONS = """\
Responde **ÚNICAMENTE** con un objeto JSON válido.
Empieza con { y termina con }. Sin texto extra, sin Markdown.

Campos y dominios (ES):
- question (str)
- answer (str)
- question_type (uno de: definicion, procedimiento, requisito, excepcion,
                 numerica, fecha_plazo, comparativa, lista, condicional)
- answer_type (uno de: extractiva, abstractive_breve)
- difficulty (uno de: baja, media, alta)
- evidence (str, ≤ 25 palabras EXACTAS del texto origen)
- evidence_char_start (int, -1 si no hay)
- evidence_char_end (int, -1 si no hay)
- is_valid (boolean)

Ejemplo:
{
  "question": "¿Qué entidad expide la resolución?",
  "answer": "La Rectoría de la UIS.",
  "question_type": "definicion",
  "answer_type": "extractiva",
  "difficulty": "baja",
  "evidence": "La Resolución fue expedida por la Rectoría de la UIS",
  "evidence_char_start": 120,
  "evidence_char_end": 165,
  "is_valid": true
}
"""

JSON_ONLY_INSTRUCTIONS_V1 = """\
Responde **ÚNICAMENTE** con un objeto JSON válido.
Empieza con { y termina con }.
(… el bloque de reglas y ejemplo que ya definiste …)
"""

JSON_ONLY_INSTRUCTIONS_V2 = """
DEVUELVE EXCLUSIVAMENTE UN OBJETO JSON VÁLIDO (sin texto extra, sin Markdown, sin comentarios).
Incluye SIEMPRE TODAS las claves, incluso si debes usar valores por defecto.

Esquema obligatorio:
{
  "question": "<str>",
  "answer": "<str>",
  "question_type": "<definicion|procedimiento|requisito|excepcion|numerica|fecha_plazo|comparativa|lista|condicional>",
  "answer_type": "<extractiva|abstractive_breve>",
  "difficulty": "<baja|media|alta>",
  "evidence": "<cita exacta (<= 25 palabras)>",
  "evidence_char_start": <int>,  // índice en entry, -1 si no aplica
  "evidence_char_end": <int>,    // índice en entry (exclusivo), -1 si no aplica
  "is_valid": <true|false>
}

Reglas duras:
- Sin texto fuera del JSON.
- NUNCA omitas claves.
- "evidence" debe ser un substring exacto de 'entry' (<= 25 palabras). Si no hay evidencia inequívoca, usa "" y pon índices en -1.
- Los índices son enteros; -1 indica "no disponible".
- Si no es posible una respuesta inequívoca, usa "is_valid": false y deja strings vacíos y -1 en índices.
"""

class GenerateQASignature_v2(dspy.Signature):
    """
    Genera UNA (1) pregunta de alta calidad (question) y su respuesta correcta (answer)
    a partir de un párrafo o tabla (entry) y su tema principal (subject).

    Reglas de redacción (ES):
    - La pregunta debe ser clara, específica y útil para evaluar comprensión o uso práctico (en español).
    - La respuesta debe ser precisa y fiel con el contenido dado.
    - No inventes: si el texto no permite una respuesta inequívoca, devuelve `is_valid = false`.
    - Prefiere preguntas que requieran inferencia ligera o integración de fragmentos del párrafo,
      evitando parafraseo trivial.
    - Si la fuente es una TABLA, prioriza preguntas que combinen 2–3 celdas con condiciones.

    Calidad y verificación:
    - Provee un fragmento EXACTO del texto (evidence) que respalde la respuesta (≤ 25 palabras).
    - Entrega índices de carácter start/end del fragmento en `evidence_char_start/end`.
    - Clasifica el tipo de pregunta (`question_type`) y el tipo de respuesta (`answer_type`).
    - Asigna una dificultad aproximada (`difficulty`).

    Para `question_type`, devuelve exactamente uno de:
    definicion | procedimiento | requisito | excepcion | numerica | fecha_plazo | comparativa | lista | condicional

    `evidence_char_start` y `evidence_char_end`: enteros **solo con dígitos** (sin texto); usa -1 si no hay evidencia.

    Campos:
    - is_valid=false si no es posible formular una pregunta respondible con el texto.
    """

    json_rule: str = dspy.InputField(desc="Instrucciones estrictas: responder SOLO JSON válido")
    
    # Entries
    entry: str = dspy.InputField(desc="Párrafo o tabla de origen (texto completo)")
    subject: str = dspy.InputField(desc="Tema general del parrafo")

    # Main outputs
    question: str = dspy.OutputField(desc="Pregunta en español, clara y concreta", default="")
    answer: str = dspy.OutputField(desc="Respuesta en español, sin adornos", default="")
    # answer: str = dspy.OutputField(desc="Respuesta breve y específica, sin adornos")

    # Quality attributes/metadata
    question_type: str = dspy.OutputField(desc="Ver lista permitida en el docstring", default="")

    answer_type: str = dspy.OutputField(desc="extractiva o abstractive_breve", default="")

    difficulty: str = dspy.OutputField(desc="baja | media | alta", default="")

    # Grounding (obligatorio para control de calidad)
    evidence: str = dspy.OutputField(
        desc="Cita EXACTA del texto que respalda la respuesta (≤ 25 palabras, sin corchetes)", default=""
    )

    evidence_char_start: int = dspy.OutputField(desc="Índice de carácter inicial del evidence en `entry`", default=-1)
    evidence_char_end: int = dspy.OutputField(desc="Índice de carácter final (exclusivo) del evidence en `entry`", default=-1)

    # Control
    is_valid: bool = dspy.OutputField(
        desc="(True | False), True si la pregunta es respondible de forma inequívoca con el texto; False en caso contrario", default=False
    )

class GenerateQASignature_v1(dspy.Signature):
    """ ... your docstring rules (ES) ... """
    paragraph = dspy.InputField()
    division  = dspy.InputField()
    question  = dspy.OutputField()
    answer    = dspy.OutputField()
    question_type = dspy.OutputField()
    answer_type   = dspy.OutputField()
    difficulty    = dspy.OutputField()
    evidence      = dspy.OutputField()
    evidence_char_start = dspy.OutputField()
    evidence_char_end   = dspy.OutputField()
    is_valid      = dspy.OutputField()
    
    
def _strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ``` fences if present
    s = re.sub(r"^\s*```(?:json)?\s*", "", s.strip(), flags=re.I)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.I)
    return s.strip()

def _extract_largest_json_object(s: str) -> str | None:
    """
    Return the substring of s that is the largest balanced top-level {...} object.
    If none, return None. Ignores braces inside quotes.
    """
    s = s.strip()
    if not s:
        return None

    start = None
    depth = 0
    in_str = False
    esc = False
    best = None
    best_len = -1

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        cand = s[start:i+1]
                        if len(cand) > best_len:
                            best = cand
                            best_len = len(cand)
                        start = None
    return best


def _word_trim(s: str, max_words=25):
    words = re.findall(r"\S+", s.strip())
    return " ".join(words[:max_words])

def coerce_and_validate(raw_json_str: str, entry: str) -> dict:
    """
    1) Parse JSON
    2) Ensure all keys exist
    3) Enforce enums and lengths
    4) Compute indices if possible
    """
    """try:
        obj = json.loads(raw_json_str)
    except Exception:
        # Last-ditch: try to extract {...} block
        m = re.search(r"\{.*\}", raw_json_str, flags=re.S)
        if not m:
            raise ValueError("No se pudo parsear JSON del LM.")
        obj = json.loads(m.group(0))"""

    # Ensure keys
    defaults = {
        "question": "",
        "answer": "",
        "question_type": "",
        "answer_type": "",
        "difficulty": "",
        "evidence": "",
        "evidence_char_start": -1,
        "evidence_char_end": -1,
        "is_valid": False,
    }
    
    if not raw_json_str or not raw_json_str.strip():
        # Return a safe default instead of raising
        return defaults
    
    # Strip code fences and try direct parse
    s = _strip_code_fences(raw_json_str)
    
    try:
        obj = json.loads(s)
    except Exception:
        # Try to extract the largest balanced {...} block
        block = _extract_largest_json_object(s)
        if not block:
            return defaults
        try:
            obj = json.loads(block)
        except Exception:
            return defaults
    
    for k,v in defaults.items():
        obj.setdefault(k, v)

    # Enforce enums
    if obj["question_type"] not in ALLOWED_QT:
        obj["question_type"] = ""
    if obj["answer_type"] not in ALLOWED_AT:
        obj["answer_type"] = ""
    if obj["difficulty"] not in ALLOWED_DIFF:
        obj["difficulty"] = ""

    # Enforce evidence length
    # obj["evidence"] = _word_trim(obj["evidence"], 25)
    obj["evidence"] = _word_trim(obj.get("evidence",""), 25)

    # Compute indices if possible
    ev = obj["evidence"].strip()
    if ev and entry:
        start = entry.find(ev)
        if start != -1:
            obj["evidence_char_start"] = int(start)
            obj["evidence_char_end"] = int(start + len(ev))
        else:
            # evidence not found verbatim → clear indices
            obj["evidence_char_start"] = -1
            obj["evidence_char_end"] = -1
    else:
        obj["evidence_char_start"] = -1
        obj["evidence_char_end"] = -1

    # Auto is_valid if obviously broken
    if not obj["question"] or not obj["answer"]:
        obj["is_valid"] = False

    return obj

class RawJSONSignature(dspy.Signature):
    instruction: str = dspy.InputField()
    entry: str = dspy.InputField()
    subject: str = dspy.InputField()
    json_text: str = dspy.OutputField()  # single string output

class RawJSONProbe(dspy.Module):
    def __init__(self):
        super().__init__()
        self.p = dspy.Predict(RawJSONSignature)

    def forward(self, entry, subject, instruction):
        # ask the LM to output ONLY the JSON string (no extra text)
        out = self.p(
            instruction=instruction,
            entry=entry,
            subject=subject,
            _lm_kwargs={"extra_body": {"format": "json"}},  # Ollama JSON mode
        )
        return out.json_text  # a raw string (hopefully pure JSON)

class QAGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # self.program = dspy.Predict(GenerateQASignature)
        self.program = dspy.Predict(GenerateQASignature_v2)

    def forward(self, entry, subject):
        qa = self.program(json_rule=JSON_ONLY_INSTRUCTIONS_V2, entry=entry, subject=subject)
        return qa
    
class QAGenerator_safe(dspy.Module):
    def __init__(self):
        super().__init__()
        # self.program = dspy.Predict(GenerateQASignature)
        self.program = dspy.Predict(GenerateQASignature_v2)
        self.probe = RawJSONProbe()

    def forward(self, entry, subject):
        # 1) Call predict with strict JSON rule
        # If you are using Ollama, pass format='json' so the server returns JSON tokens.
        lm_kwargs = {"extra_body": {"format": "json"}, "temperature": 0}  # works for ollama_chat/* backends
        
        try:
            qa = self.program(json_rule=JSON_ONLY_INSTRUCTIONS_V2, entry=entry, subject=subject, _lm_kwargs=lm_kwargs,)
            # 2) The adapter already tried to parse fields, BUT if the LM (or server) output
            #    included stray tokens, we still harden the result by rebuilding from the raw text.
            #    Get raw text from the last call:
            raw = json.dumps({
                "question": qa.question,
                "answer": qa.answer,
                "question_type": qa.question_type,
                "answer_type": qa.answer_type,
                "difficulty": qa.difficulty,
                "evidence": qa.evidence,
                "evidence_char_start": qa.evidence_char_start,
                "evidence_char_end": qa.evidence_char_end,
                "is_valid": qa.is_valid,
            }, ensure_ascii=False)
        except Exception:
            raw = self.probe(entry=entry, subject=subject, instruction=JSON_ONLY_INSTRUCTIONS_V1)
            
        # 3) Coerce and validate to guarantee fields are present & consistent
        coerced = coerce_and_validate(raw, entry)
        
        # 4) Return a typed object (or a simple dict if you prefer)
        class Obj: pass
        obj = Obj()
        for k, v in coerced.items():
            setattr(obj, k, v)
        return obj
     

class QAGenerator_v1(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = dspy.PromptTemplate(
            """{json_rule}

Texto fuente (entry):
{entry}

Tema (subject): {subject}

Devuelve **solo** el JSON solicitado."""
        )
        self.program = dspy.Predict(GenerateQASignature_v1, prompt=self.prompt)

    def forward(self, entry, subject):
        return self.program(entry=entry, subject=subject, json_rule=JSON_ONLY_INSTRUCTIONS)

class UniversityRAG(dspy.Signature):
    #"""Responda una pregunta (question) relacionada con la universidad utilizando el contexto (context) dado."""
    """
    Eres un asistente experto en reglamentos e informacion de la Universidad industrial de santander.
    Tu tarea es responder con claridad, profundidad y precisión, basándote exclusivamente en los textos proporcionados. Muy importante **No alucinar**. Responde siempre en español.

    Con base en los siguientes textos extraídos de documentos oficiales o internet:
    {context}

     Responda la pregunta del usuario:
    {question}

    Analiza el caso utilizando los reglamentos presentados y la informacion de internet. Tu respuesta debe seguir esta estructura:

    Responde a la siguiente pregunta siguiendo este esquema:
    - Razonamiento paso a paso, explicando qué normas del contexto aplican y cómo pueden ayudar a responder la pregunta (en español). No omitas pasos importantes ni información clave del contexto.
    - Respuesta final clara y fundamentada, basada en el razonamiento (en español). Responde de forma detallada y completa. No des respuestas generales ni superficiales. No digas "depende" ni "consulta con la universidad" o "consulta en el articulo", dar la informacion especifica.
    - Citas explícitas: indica artículo y reglamento correspondiente (en español).

    Devuelve:
    - reasoning: razonamíento lógico paso a paso (en español)
    - answer: respuesta final (en español)
    - citations: lista de artículos/reglamentos usados (en español)
    """
    question: str = dspy.InputField(desc="pregunta (en español)")
    context: str = dspy.InputField(desc="contexto (en español)")
    reasoning: str = dspy.OutputField(desc="razonamiento (en español)")
    answer: str = dspy.OutputField(desc="respuesta (en español)")
    citations: str = dspy.OutputField(desc="citas (en español)")