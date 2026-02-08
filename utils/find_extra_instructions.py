import json
import os

def filter_new_intents(base_file_path, target_file_path, output_file_path):
    """
    找出 target_file 相对于 base_file 在 intent_name 上多出来的项，并保存。
    """
    # 1. 读取基础文件（单轮-冒烟测试集），提取所有已存在的 intent_name
    existing_intents = set()
    
    print(f"正在加载基础文件: {base_file_path} ...")
    if not os.path.exists(base_file_path):
        print(f"错误: 找不到文件 {base_file_path}")
        return

    with open(base_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "intent_name" in data:
                    existing_intents.add(data["intent_name"])
            except json.JSONDecodeError:
                print(f"警告: 基础文件第 {line_num} 行 JSON 格式错误，跳过。")

    print(f"基础文件中包含 {len(existing_intents)} 个唯一的 intent_name。")

    # 2. 读取目标文件（决赛指令集），对比并写入新文件
    new_items_count = 0
    seen_new_intents = set() # 用于统计新增了多少种 intent（可选）

    print(f"正在处理目标文件: {target_file_path} ...")
    if not os.path.exists(target_file_path):
        print(f"错误: 找不到文件 {target_file_path}")
        return
        
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(target_file_path, 'r', encoding='utf-8') as f_in, \
         open(output_file_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                current_intent = data.get("intent_name")
                
                # 核心逻辑：如果 intent_name 不在基础集合中
                if current_intent and current_intent not in existing_intents:
                    # 写入原行（或者 dump 后的标准 json）
                    # ensure_ascii=False 保证中文正常显示
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    new_items_count += 1
                    seen_new_intents.add(current_intent)
                    
            except json.JSONDecodeError:
                print(f"警告: 目标文件第 {line_num} 行 JSON 格式错误，跳过。")

    print("-" * 30)
    print(f"处理完成！")
    print(f"相较于基础文件，共筛选出 {new_items_count} 行数据。")
    print(f"包含 {len(seen_new_intents)} 个全新的 intent_name。")
    print(f"结果已保存至: {output_file_path}")

if __name__ == "__main__":
    # 定义文件路径
    base_file = "data/单轮-冒烟测试集.jsonl"
    target_file = "data/决赛指令集.jsonl"
    # 输出文件命名为 "新增intent数据.jsonl"，保存在 data 目录下
    output_file = "data/新增intent数据.jsonl"

    filter_new_intents(base_file, target_file, output_file)