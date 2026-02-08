import json
import re

# --- 1. 配置 (保持不变) ---
ORIGINAL_FILE_PATH = "./data/多轮-冒烟测试集.jsonl" 
AUGMENTED_FILE_PATH = "./data/多轮-冒烟测试集增强.jsonl"
OUTPUT_FILE_PATH = "./data/多轮-冒烟测试集增强_formatted.jsonl"
EXPANSION_RATE = 50

# --- 新增的清洗函数 ---
def clean_and_load_json(line_str: str):
    """
    Cleans a string before attempting to parse it as JSON.
    Handles potential markdown artifacts and other non-standard characters.
    """
    # 1. 移除首尾可能存在的Markdown代码块标记和"json"标识符
    cleaned_str = re.sub(r"^\s*```json\s*", "", line_str)
    cleaned_str = re.sub(r"\s*```\s*$", "", cleaned_str)
    
    # 2. 移除首尾的空白字符，包括换行符、回车符等
    cleaned_str = cleaned_str.strip()
    
    # 3. 尝试移除行尾可能存在的逗号（以防万一）
    if cleaned_str.endswith(','):
        cleaned_str = cleaned_str[:-1]

    # 4. 尝试加载
    return json.loads(cleaned_str)


# --- 2. 主转换逻辑 (已修改) ---
def convert_augmented_data():
    print("--- 开始转换增强数据格式 ---")
    
    # --- 读取原始文件元数据 (不变) ---
    try:
        with open(ORIGINAL_FILE_PATH, 'r', encoding='utf-8') as f:
            original_metadata = [json.loads(line) for line in f if line.strip()]
        for item in original_metadata:
            item.pop('data', None)
        print(f"从 '{ORIGINAL_FILE_PATH}' 中成功加载 {len(original_metadata)} 条元数据。")
    except Exception as e:
        print(f"读取或解析原始文件时出错: {e}")
        return

    # --- 读取增强文件 (使用新的清洗函数) ---
    augmented_data_only = []
    try:
        print(f"正在读取并清洗增强文件: '{AUGMENTED_FILE_PATH}'")
        with open(AUGMENTED_FILE_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        # 使用我们新的、更稳健的加载函数
                        json_obj = clean_and_load_json(line)
                        augmented_data_only.append(json_obj['data'])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"❌ 错误: 解析第 {i} 行失败: {e}")
                        print(f"   问题行内容: {line.strip()}")
                        print("   请手动检查并修正该行，或从增强数据文件中删除该行。")
                        return # 遇到错误即停止，以便修复
        print(f"从 '{AUGMENTED_FILE_PATH}' 中成功加载 {len(augmented_data_only)} 条增强对话。")

    except FileNotFoundError:
        print(f"错误：找不到增强文件 '{AUGMENTED_FILE_PATH}'。请检查路径。")
        return
    
    # --- 后续的验证和写入逻辑 (不变) ---
    # ... (这部分代码与之前完全相同) ...
    if len(augmented_data_only) != len(original_metadata) * EXPANSION_RATE:
        print("\n--- ⚠️ 警告：数据量不匹配！---")

    written_count = 0
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as outfile:
        # ... (写入逻辑不变) ...
        for i, metadata in enumerate(original_metadata):
            start_index = i * EXPANSION_RATE
            end_index = start_index + EXPANSION_RATE
            augmented_chunk = augmented_data_only[start_index:end_index]
            for aug_data in augmented_chunk:
                new_record = metadata.copy()
                new_record['data'] = aug_data
                outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                written_count += 1
                
    print("\n--- 转换完成 ---")
    print(f"成功生成 {written_count} 条格式完整的增强数据。")
    print(f"结果已保存到: '{OUTPUT_FILE_PATH}'")


if __name__ == "__main__":
    convert_augmented_data()

# import json
# import re

# # --- 1. 配置 ---
# INPUT_FILE_PATH = "./data/多轮-冒烟测试集增强.jsonl"
# OUTPUT_FILE_PATH = "./data/多轮-冒烟测试集增强_fixed.jsonl"


# # --- 2. 核心修复函数 (全新逻辑) ---
# def fix_line_safely(line_str: str) -> (str, bool):
#     """
#     Safely finds and fixes unescaped quotes within a JSON string line.
#     Returns the fixed line and a boolean indicating if a fix was applied.
#     """
#     original_line = line_str.strip()
#     if not original_line:
#         return "", False # 空行

#     # 步骤1: 尝试直接加载，如果成功，说明无需修复
#     try:
#         json.loads(original_line)
#         return original_line, False
#     except json.JSONDecodeError:
#         # 如果失败，进入修复流程
#         pass

#     # 步骤2: 定义一个精确的正则表达式，只匹配 Parameter="Value" 的形式
#     # (\w+)       - 捕获组1: 任何单词字符 (参数名, e.g., DeviceType)
#     # =           - 匹配等号
#     # (")         - 捕获组2: 一个双引号
#     # (.*?)       - 捕获组3: 任何字符, 非贪婪模式 (参数值, e.g., 音箱)
#     # (")         - 捕获组4: 另一个双引号
#     #
#     # 这个正则后面跟一个'零宽度负向先行断言' (?![,\s]*\})，
#     # 确保这个引号后面不是逗号或空格然后跟着一个花括号 `}`。
#     # 这可以避免错误地匹配JSON本身的键值对，比如 "role":"user"。
#     pattern = re.compile(r'(\w+)=(")(.*?)(")(?![,\s]*\})')

#     def replacer(match):
#         # 重组时，对捕获的引号进行转义: " -> \"
#         return f'{match.group(1)}=\\"{match.group(3)}\\"'

#     # 步骤3: 在整行字符串上执行替换
#     fixed_line = pattern.sub(replacer, original_line)

#     # 步骤4: 再次尝试加载修复后的字符串，进行验证
#     try:
#         json.loads(fixed_line)
#         # 如果这次加载成功，说明修复是有效的
#         return fixed_line, True
#     except json.JSONDecodeError:
#         # 如果仍然失败，说明问题更复杂，我们放弃修复并发出警告
#         print(f"\n警告: 自动修复失败，该行可能存在更复杂的语法错误: {original_line}")
#         return original_line, False


# # --- 3. 主处理逻辑 (更新) ---
# def process_file():
#     print(f"--- 开始修复文件: {INPUT_FILE_PATH} ---")
    
#     lines_processed = 0
#     lines_fixed = 0
    
#     try:
#         with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as infile, \
#              open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as outfile:
            
#             for line in infile:
#                 lines_processed += 1
#                 fixed_line, was_fixed = fix_line_safely(line)
                
#                 if was_fixed:
#                     lines_fixed += 1
                
#                 if fixed_line:
#                     outfile.write(fixed_line + "\n")

#         print("\n--- 修复完成 ---")
#         print(f"总共处理了 {lines_processed} 行。")
#         print(f"修复了 {lines_fixed} 行存在嵌套引号问题的代码。")
#         print(f"结果已保存到: {OUTPUT_FILE_PATH}")
#         if lines_processed != lines_fixed:
#              print(f"有 {lines_processed - lines_fixed} 行无需修复。")
#         print(f"\n您现在应该可以使用 '{OUTPUT_FILE_PATH}' 文件进行下一步的数据格式转换。")

#     except FileNotFoundError:
#         print(f"错误: 找不到输入文件 '{INPUT_FILE_PATH}'。")
#     except Exception as e:
#         print(f"处理过程中发生未知错误: {e}")


# if __name__ == "__main__":
#     process_file()