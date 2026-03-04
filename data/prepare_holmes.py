import os
import json
import random

def main():
    # 1. 配置路径 (假设在项目根目录下运行此脚本)
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root)
    
    holmes_dir = os.path.join(data_dir, "holmes_dataset")
    jsonl_file = os.path.join(holmes_dir, "SFTDATA.jsonl")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    samples = []
    
    print(f"开始解析 {jsonl_file} ...")
    
    # 2. 读取并解析原始 JSONL 文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # 获取原始字段
                query = data.get("query", "")
                response = data.get("response", "")
                image_paths = data.get("images", [])
                
                if not image_paths:
                    continue
                    
                # 处理图片路径映射
                # 原始路径形如: "./dataset/1_fake/progan_train_horse_1_fake_19373.png"
                # 提取出 "1_fake/..." 或 "0_real/..." 的部分
                raw_img_path = image_paths[0]
                path_parts = raw_img_path.replace("\\", "/").split("/")
                
                # 寻找关键的子文件夹名
                folder_name = "1_fake" if "1_fake" in path_parts else "0_real"
                file_name = path_parts[-1]
                
                # 判断标签 (0为真，1为AI生成)
                label = 1 if folder_name == "1_fake" else 0
                
                # 构建给 DataLoader 用的相对路径
                # DataLoader 的 self.data_dir 是 "data/train" 或 "data/val"
                # 所以相对路径应该往上跳一级，指向 holmes_dataset
                rel_image_path = os.path.join("..", "holmes_dataset", "dataset_huggingface", folder_name, file_name)
                # 统一使用正斜杠以防止路径在 JSON 中转义过于混乱
                rel_image_path = rel_image_path.replace("\\", "/")

                # 3. 按照 AIGIDataset 的期望格式构建样本字典
                # 注意：我们不硬编码 "stage"，因为 dataset_loader 会根据初始化参数自动处理
                sample = {
                    "image_path": rel_image_path,
                    "label": label,
                    "text_query": query,
                    "expert_explanation": response,
                    "mask": None  # Holmes数据集没有提供像素级mask，按你的loader逻辑设为None即可
                }
                
                samples.append(sample)
                
            except json.JSONDecodeError:
                print(f"警告: 无法解析第 {line_num + 1} 行")
                continue

    total_samples = len(samples)
    print(f"成功解析了 {total_samples} 个样本。")

    # 4. 划分训练集和验证集 (90% 训练, 10% 验证)
    random.seed(42) # 固定随机种子以保证可重复性
    random.shuffle(samples)
    
    split_idx = int(total_samples * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # 5. 写入 JSON 文件
    train_annotation_path = os.path.join(train_dir, "annotations_unified.json")
    val_annotation_path = os.path.join(val_dir, "annotations_unified.json")
    
    print(f"正在保存训练集 ({len(train_samples)} 个样本) 到 {train_annotation_path} ...")
    with open(train_annotation_path, 'w', encoding='utf-8') as f:
        json.dump({"samples": train_samples}, f, ensure_ascii=False, indent=2)

    print(f"正在保存验证集 ({len(val_samples)} 个样本) 到 {val_annotation_path} ...")
    with open(val_annotation_path, 'w', encoding='utf-8') as f:
        json.dump({"samples": val_samples}, f, ensure_ascii=False, indent=2)

    print("数据格式转换完成！你可以开始进行第一阶段和第二阶段的训练了。")

if __name__ == "__main__":
    main()