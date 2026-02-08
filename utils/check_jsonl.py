import json

TRAIN_DATA_FILES = [
    "data/final_data/单轮单指令_冒烟.jsonl",
    "data/final_data/单轮单指令_增强.jsonl",
    "data/final_data/多轮单指令_冒烟.jsonl",
    "data/final_data/多轮单指令_增强.jsonl",
    "data/final_data/单轮多指令_增强.jsonl",
    "data/final_data/单轮多指令_合成.jsonl",
    "data/final_data/多轮多指令_增强.jsonl",
    "data/final_data/决赛冒烟集.jsonl",
    "data/final_data/高质量多轮多.jsonl",
]

def check_jsonl_file(filepath):
    """检查一个 jsonl 文件的每一行是否是合法 JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"文件 {filepath} 第 {i} 行不是合法 JSON: {e}")
                return False
    return True

all_ok = True
for file in TRAIN_DATA_FILES:
    print(f"检查文件: {file} ...")
    if not check_jsonl_file(file):
        all_ok = False

if all_ok:
    print("所有文件都是合法的 JSONL 文件 ✅")
else:
    print("存在不合法的 JSONL 文件 ❌")
