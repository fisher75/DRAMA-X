import json
import os
import argparse

def populate_paths_jsonl(annotations_file: str, integrated_file: str, output_file: str):
    # Load integrated_output_v2 entries (still a single JSON file)
    with open(integrated_file, 'r', encoding='utf-8') as f:
        integrated = json.load(f)

    # Build a lookup from key -> (img, vid)
    lookup = {}
    for item in integrated:
        img = item.get('s3_fileUrl', '')
        vid = item.get('s3_instructionReference', '')
        parts = img.rstrip('/').split('/')
        if len(parts) < 2:
            continue
        clip_id = parts[-2]                # e.g. 'clip_305_000786'
        frame_file = parts[-1]             # e.g. 'frame_000786.png'
        frame_id = os.path.splitext(frame_file)[0]
        key = f"{clip_id}_{frame_id}"
        lookup[key] = (img, vid)

    # Process each line of the JSONL annotations
    with open(annotations_file, 'r', encoding='utf-8') as src, \
         open(output_file,   'w', encoding='utf-8') as dst:
        for line in src:
            rec = json.loads(line)
            key = rec.get("id")
            if key in lookup:
                img, vid = lookup[key]
                rec['image_path'] = img
                rec['video_path'] = vid
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Updated JSONL written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate image_path/video_path in drama_x_annotations.jsonl"
    )
    parser.add_argument(
        "annotations",
        help="Path to drama_x_annotations.jsonl"
    )
    parser.add_argument(
        "integrated",
        help="Path to integrated_output_v2.json"
    )
    parser.add_argument(
        "-o", "--output",
        default="drama_x_annotations_populated.jsonl",
        help="Where to save the updated annotations"
    )
    args = parser.parse_args()
    populate_paths_jsonl(args.annotations, args.integrated, args.output)
