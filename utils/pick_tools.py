import random
import json
import os

# --- 配置 ---
INPUT_FILE = "prompts/tools.txt"   # 输入文件
OUTPUT_FILE = "output.txt"         # 新增：输出文件
BATCH_COUNT = 200                  # 一次生成几组任务？

def pick_tools_for_prompt():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    # 1. 读取所有工具
    all_lines = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_lines.append(line.strip())

    if len(all_lines) < 6:
        print(f"⚠️ 警告：工具总数只有 {len(all_lines)} 个，不足以进行随机抽取。")

    print(f"✅ 成功加载 {len(all_lines)} 个工具。正在抽取...\n")
    print("=" * 40)

    # ---- 新增：打开输出文件 ----
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        def log(x):
            """同时写入文件和打印到屏幕"""
            print(x)
            out.write(x + "\n")

        # 2. 随机抽取并格式化输出
        for i in range(BATCH_COUNT):
            # k = random.randint(3, 6)
            k=10
            selected_lines = random.sample(all_lines, k)

            log(f"【 任务批次 {i+1} (包含 {k} 个工具) 】")

            formatted_tools = []
            for line in selected_lines:
                try:
                    json_obj = json.loads(line)
                    formatted_tools.append(json_obj)
                except:
                    pass

            if formatted_tools:
                pretty = json.dumps(formatted_tools, ensure_ascii=False, indent=4)
                log(pretty)
            else:
                raw = "\n" + ",\n".join(selected_lines) + "\n"
                log(raw)

            log("")

    print(f"\n🎉 已保存到 {OUTPUT_FILE}\n")

if __name__ == "__main__":
    pick_tools_for_prompt()
