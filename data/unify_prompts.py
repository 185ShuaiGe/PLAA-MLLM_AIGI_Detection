import json
import os

def unify_prompts(input_path, output_path):
    # 标准化的统一 Prompt (建议与你测试时使用的 default_prompt 保持完全一致)
    UNIFIED_PROMPT = "<image>\nAnalyze this image and determine if it is real or AI-generated. Please provide your reasoning."
    
    print(f"正在读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    count = 0
    if "samples" in data:
        for sample in data["samples"]:
            sample["text_query"] = UNIFIED_PROMPT
            count += 1
            
    # 覆盖保存或另存为新文件
    print(f"正在保存至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"完成！共替换了 {count} 条数据的 prompt。")

if __name__ == "__main__":
    # 请根据你的实际路径修改这里
    input_file = "val/annotations.json" 
    output_file = "val/annotations_unified.json" # 建议先存为新文件，确认无误后再重命名覆盖
    
    if os.path.exists(input_file):
        unify_prompts(input_file, output_file)
    else:
        print(f"找不到文件: {input_file}")