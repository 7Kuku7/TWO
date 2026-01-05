import pandas as pd
import json
import os

excel_path = r"K:\NVS-SQA\TWO\final_data（puls）.xlsx"
json_path = r"K:\NVS-SQA\TWO\mos_advanced.json"

try:
    df = pd.read_excel(excel_path)
    
    # Columns: ['序号', '场景', '方法', '路径', '失真类型', 'MOS', '不适', '模糊', '光照', '伪影']
    # Target Key: Scene+Method+Condition+Path
    
    data = {}
    for index, row in df.iterrows():
        scene = str(row['场景']).strip()
        method = str(row['方法']).strip()
        path = str(row['路径']).strip()
        condition = str(row['失真类型']).strip()
        
        # Construct key
        key = f"{scene}+{method}+{condition}+{path}"
        
        # Extract scores
        mos = float(row['MOS'])
        
        # Sub-scores (Handle potential missing values or non-numeric)
        sub_scores = {
            "discomfort": float(row['不适']),
            "blur": float(row['模糊']),
            "lighting": float(row['光照']),
            "artifacts": float(row['伪影'])
        }
        
        data[key] = {
            "mos": mos,
            "sub_scores": sub_scores
        }
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully converted {len(data)} entries to {json_path}")
    
except Exception as e:
    print(f"Error converting excel: {e}")
