import json

def merge_json_files(visual_path, whisper_path, output_path):
    # Load Whisper data
    whisper_data = {}
    with open(whisper_path, 'r', encoding='utf-8') as f:
        for line in f:
            video = json.loads(line)
            whisper_data[video['video_id']] = video

    merged_videos = []

    # Process Visual data
    with open(visual_path, 'r', encoding='utf-8') as f:
        for line in f:
            visual_video = json.loads(line)
            video_id = visual_video['video_id']

            if video_id in whisper_data:
                whisper_video = whisper_data[video_id]

                # Create a map by timestamp (robust match)
                whisper_segments_map = {
                    float(s['time_start']): s for s in whisper_video['segments']
                }

                for v_seg in visual_video['segments']:
                    start_time = float(v_seg['time_start'])

                    # Tolerancia por diferencias flotantes
                    w_seg = None
                    for w_start, seg in whisper_segments_map.items():
                        if abs(w_start - start_time) < 0.1:
                            w_seg = seg
                            break

                    v_seg['subtitle'] = w_seg['text'] if w_seg else ""

            merged_videos.append(visual_video)

    # Write merged output
    with open(output_path, 'w', encoding='utf-8') as f:
        for video in merged_videos:
            f.write(json.dumps(video, ensure_ascii=False) + '\n')

    return merged_videos[0] if merged_videos else None


# ----------- RUTAS QUE PEDISTE -------------
visual_path = "../json/linkata_videos_visual_completo.jsonl"
whisper_path = "../json/linkata_videos_whisper.jsonl"
output_path = "../json/linkata_videos_merged.jsonl"

# Ejecutar merge y mostrar el primer video unificado
first_merged_video = merge_json_files(visual_path, whisper_path, output_path)
print(first_merged_video)
