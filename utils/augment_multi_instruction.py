import json
import random
import os
from tqdm import tqdm

# --- 配置 ---
INPUT_FILE = "data/final_data/单轮单指令_增强.jsonl"
OUTPUT_FILE = "data/final_data/单轮多指令_合成.jsonl"

# 连接词库：让拼接更自然
# 分为两类：通用连接词 和 句首连接词（用于长句子的后半部分）
CONNECTORS = [
    "，然后", "，并且", "，顺便", "。哦对了，还有", 
    "，同时", "。另外，", "，再帮我", "，接着",
    "；还有就是，", "，以及"
]

# --- 辅助函数 ---
def clean_trailing_punctuation(text):
    """去除句子末尾的标点符号，以便拼接"""
    return text.strip().rstrip("。！？.!?")

def generate_multi_instruction_data():
    # 1. 读取源数据
    source_data = []
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    print(f"正在读取源文件: {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # 确保数据格式正确：必须包含data且至少有一问一答
                    if "data" in item and len(item["data"]) >= 2:
                        # 只保留我们需要的部分，减小内存占用
                        clean_item = {
                            "user_content": item["data"][0]["content"],
                            "assistant_content": item["data"][1]["content"]
                        }
                        source_data.append(clean_item)
                except json.JSONDecodeError:
                    continue
    
    print(f"源数据加载完成，共 {len(source_data)} 条有效单指令样本。")
    
    if len(source_data) < 6:
        print("错误：源数据太少，无法进行多指令组合（至少需要6条）。")
        return

    # 2. 开始生成
    total_generated = 0
    ks = [2, 3, 4, 5, 6] # 遍历组合的数量
    SAMPLES_PER_K = 1000 # 每个k生成多少条

    print(f"开始生成多指令数据，K值范围: {ks}, 每个K生成: {SAMPLES_PER_K}条...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for k in ks:
            print(f"正在生成包含 {k} 条指令的样本...")
            
            for _ in tqdm(range(SAMPLES_PER_K), desc=f"K={k}"):
                # 随机抽取 k 个不重复的样本
                selected_samples = random.sample(source_data, k)
                
                # --- 拼接用户指令 ---
                combined_user_content = ""
                for i, sample in enumerate(selected_samples):
                    content = sample["user_content"]
                    
                    if i == 0:
                        # 第一句：去除末尾标点，保持原样
                        combined_user_content += clean_trailing_punctuation(content)
                    else:
                        # 后续句子：加连接词 + 去除末尾标点（除了最后一句）
                        connector = random.choice(CONNECTORS)
                        cleaned_content = clean_trailing_punctuation(content)
                        
                        # 如果是最后一句，保留标点（或者也可以不处理，直接拼）
                        if i == k - 1:
                            # 最后一句可以保留原句的语气词和标点，看起来更自然
                            # 这里我们选择直接拼上原句（带标点），如果原句没标点可以补一个
                            content_with_punct = content
                            if content_with_punct[-1] not in "。！？.!?":
                                content_with_punct += "。"
                            combined_user_content += connector + content_with_punct
                        else:
                            combined_user_content += connector + cleaned_content

                # --- 拼接 Assistant 回复 ---
                # 提取每个样本的回复，并用 | 连接
                # 注意：如果源数据里已经包含了 <tool> 标签，这里需要先去掉，最后统一加？
                # 假设源数据格式是 "TaskManagerOnOff(ActionType=True)" 这种纯函数字符串
                # 如果源数据带有 <tool> 标签，请取消下面这行的注释：
                # tool_contents = [s["assistant_content"].replace("<tool>", "").replace("</tool>", "") for s in selected_samples]
                
                tool_contents = [s["assistant_content"] for s in selected_samples]
                combined_assistant_content = "|".join(tool_contents)

                # --- 构建新样本 ---
                new_entry = {
                    "data": [
                        {"role": "user", "content": combined_user_content},
                        {"role": "assistant", "content": combined_assistant_content}
                    ],
                    # 添加一些元数据方便后续分析，训练时process_dataset会自动忽略这些
                    "difficulty": k, 
                    "subcategory": "单轮多指令_合成",
                    "source": "synthetic_concatenation"
                }
                
                f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                total_generated += 1

    print(f"\n✅ 生成完成！")
    print(f"共生成 {total_generated} 条数据。")
    print(f"文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_multi_instruction_data()