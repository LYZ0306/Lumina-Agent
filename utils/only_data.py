import json

input_file = "data/单轮-冒烟测试集.jsonl"
output_file = "data/单轮-冒烟测试集_only_data.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for idx, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[JSON ERROR] line {idx}: {e}")
            print(f"  ➜ 内容: {line[:120]} ...")
            continue  # 跳过坏行，不中断

        # 只保留 data 字段
        only_data = {"data": obj.get("data", None)}
        fout.write(json.dumps(only_data, ensure_ascii=False) + "\n")

print(f"完成！输出文件：{output_file}")