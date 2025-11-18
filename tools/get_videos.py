import json
from typing import List, Dict, Optional
from datetime import timedelta

def format_time(seconds: float) -> str:
    """Convierte segundos a formato HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def get_segments_by_time_range(
    jsonl_file: str,
    start_time: float,
    end_time: float,
    video_id: Optional[str] = None
) -> Dict:
    """
    Extrae segmentos de subtítulos dentro de un rango de tiempo específico.
    
    Args:
        jsonl_file: Ruta al archivo JSONL
        start_time: Tiempo de inicio en segundos
        end_time: Tiempo final en segundos
        video_id: ID del video (opcional, si hay múltiples videos en el archivo)
    
    Returns:
        Dict con información del video y segmentos filtrados
    """
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Si se especifica video_id, buscar ese video específico
            if video_id and data.get('video_id') != video_id:
                continue
            
            # Filtrar segmentos en el rango de tiempo
            filtered_segments = [
                seg for seg in data.get('segments', [])
                if seg['time_start'] >= start_time and seg['time_end'] <= end_time
            ]
            
            # También incluir segmentos que se solapan parcialmente
            overlapping_segments = [
                seg for seg in data.get('segments', [])
                if (seg['time_start'] < start_time and seg['time_end'] > start_time) or
                   (seg['time_start'] < end_time and seg['time_end'] > end_time)
            ]
            
            all_segments = filtered_segments + overlapping_segments
            # Ordenar por tiempo de inicio y eliminar duplicados
            all_segments = sorted(
                {seg['time_start']: seg for seg in all_segments}.values(),
                key=lambda x: x['time_start']
            )
            
            return {
                'video_id': data.get('video_id'),
                'title': data.get('title'),
                'video_url': data.get('video_url'),
                'duration': data.get('duration'),
                'time_range': {
                    'start': format_time(start_time),
                    'end': format_time(end_time),
                    'start_seconds': start_time,
                    'end_seconds': end_time
                },
                'segments': all_segments,
                'full_text': ' '.join(seg['text'] for seg in all_segments)
            }
    
    return None

def get_video_info(jsonl_file: str, video_id: Optional[str] = None) -> Dict:
    """
    Obtiene información completa del video incluyendo duración.
    """
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            if video_id and data.get('video_id') != video_id:
                continue
            
            duration_seconds = data.get('duration', 0)
            segments = data.get('segments', [])
            
            info = {
                'video_id': data.get('video_id'),
                'title': data.get('title'),
                'url': data.get('video_url'),
                'duration_seconds': duration_seconds,
                'duration_formatted': format_time(duration_seconds),
                'duration_minutes': round(duration_seconds / 60, 2),
                'total_segments': len(segments),
                'channel': data.get('channel'),
                'view_count': data.get('view_count'),
                'like_count': data.get('like_count'),
            }
            
            if segments:
                info['first_segment'] = format_time(segments[0]['time_start'])
                info['last_segment'] = format_time(segments[-1]['time_end'])
            
            return info
    
    return None

def analyze_video_segments(jsonl_file: str, video_id: Optional[str] = None):
    """
    Muestra información general del video y sus segmentos.
    """
    info = get_video_info(jsonl_file, video_id)
    
    if info:
        print(f"Video ID: {info['video_id']}")
        print(f"Título: {info['title']}")
        print(f"Canal: {info['channel']}")
        print(f"\n📊 DURACIÓN DEL VIDEO:")
        print(f"  • Formato HH:MM:SS: {info['duration_formatted']}")
        print(f"  • En segundos: {info['duration_seconds']} seg")
        print(f"  • En minutos: {info['duration_minutes']} min")
        print(f"\n📝 Subtítulos:")
        print(f"  • Total de segmentos: {info['total_segments']}")
        if 'first_segment' in info:
            print(f"  • Primer segmento: {info['first_segment']}")
            print(f"  • Último segmento: {info['last_segment']}")
        print(f"\n👁️ Vistas: {info['view_count']:,}")
        print(f"👍 Likes: {info['like_count']}")
        print(f"\n🔗 URL: {info['url']}")
        
        return info
    
    return None

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    jsonl_file = "../json/linkata_videos_whisper.jsonl"  # Cambiar por tu archivo
    
    # Ejemplo 1: Analizar de minuto 3 a minuto 5
    print("=" * 80)
    print("ANÁLISIS DE RANGO DE TIEMPO: 3:00 - 5:00")
    print("=" * 80)
    
    result = get_segments_by_time_range(
        jsonl_file,
        start_time=180,  # 3 minutos
        end_time=300     # 5 minutos
    )
    
    if result:
        print(f"\nVideo: {result['title']}")
        print(f"Rango: {result['time_range']['start']} - {result['time_range']['end']}")
        print(f"\nSegmentos encontrados: {len(result['segments'])}")
        print("\n" + "-" * 80)
        
        for seg in result['segments']:
            print(f"[{format_time(seg['time_start'])} - {format_time(seg['time_end'])}]")
            print(f"  {seg['text']}")
            print()
        
        print("-" * 80)
        print("TEXTO COMPLETO:")
        print(result['full_text'])
    
    # Ejemplo 2: Analizar información general
    print("\n\n" + "=" * 80)
    print("INFORMACIÓN GENERAL DEL VIDEO")
    print("=" * 80)
    analyze_video_segments(jsonl_file)