---
license: cc-by-4.0
---


1. Request access to the original DRAMA dataset at  
   https://usa.honda-ri.com/drama#Downloadthedataset

2. Download the ZIP and extract the file  
   `integrated_output_v2.json`

3. Ensure you have your existing  
   `drama_x_annotations.json` (with empty `image_path`/`video_path` fields) in the same folder.

4. Run the population script:  
   ```bash
   python populate_drama_x.py \
     drama_x_annotated.json \
     integrated_output_v2.json \
     -o drama_x_annotations_populated.json
