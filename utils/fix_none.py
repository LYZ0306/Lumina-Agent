# import json
# import re
# import os

# # 定义需要处理的文件列表
# TRAIN_DATA_FILES = [
#     "data/final_data/单轮单指令_冒烟.jsonl",
#     "data/final_data/单轮单指令_增强.jsonl",
#     "data/final_data/多轮单指令_冒烟.jsonl",
#     "data/final_data/多轮单指令_增强.jsonl",
#     "data/final_data/单轮多指令_增强.jsonl",
#     "data/final_data/单轮多指令_合成.jsonl",
#     "data/final_data/多轮多指令_增强.jsonl",
#     "data/final_data/决赛冒烟集.jsonl",
# ]

# def clean_empty_params(content):
#     """
#     清洗content字符串中的空参数。
#     兼容两种空值格式：
#     1. 标准格式: Key=""
#     2. 转义格式: Key=\"\" (针对嵌套引用数据)
#     """
#     if not content or not isinstance(content, str):
#         return content

#     # 定义基础正则组件
#     # param_key: 匹配参数名 (字母数字下划线)
#     param_key = r'[a-zA-Z0-9_]+'
    
#     # empty_val: 核心修改点
#     # (?: ... ) 是非捕获组
#     # ""       匹配标准的空引号
#     # |        或
#     # \\"\\"   匹配转义后的空引号 (即文本里的 \"\")
#     empty_val = r'(?:""|\\"\\")'

#     # --- 组合正则模式 ---

#     # Pattern 1: 位于开头或中间的参数 (Key="" 或 Key=\"\", )
#     # 匹配: 参数名 + 等号 + 空值 + 逗号
#     pattern_start_or_mid = f'{param_key}\\s*=\\s*{empty_val}\\s*,\\s*'
    
#     # Pattern 2: 位于末尾的参数 (, Key="" 或 , Key=\"\")
#     # 匹配: 逗号 + 参数名 + 等号 + 空值
#     pattern_end = f',\\s*{param_key}\\s*=\\s*{empty_val}'
    
#     # Pattern 3: 唯一的参数 ((Key="") 或 (Key=\"\"))
#     # 匹配: 左括号后 + 参数名 + 等号 + 空值 + 右括号前
#     pattern_sole = f'(?<=\\()\\s*{param_key}\\s*=\\s*{empty_val}\\s*(?=\\))'

#     # --- 执行替换 ---
    
#     # 1. 先删掉 "Key="", " 这种形式
#     cleaned_content = re.sub(pattern_start_or_mid, '', content)
    
#     # 2. 再删掉 ", Key=""" 这种形式
#     cleaned_content = re.sub(pattern_end, '', cleaned_content)
    
#     # 3. 最后处理剩下的唯一参数情况
#     cleaned_content = re.sub(pattern_sole, '', cleaned_content)

#     return cleaned_content

# def process_files():
#     total_files = 0
#     total_modified_lines = 0

#     for file_path in TRAIN_DATA_FILES:
#         if not os.path.exists(file_path):
#             print(f"[跳过] 文件不存在: {file_path}")
#             continue

#         print(f"正在处理: {file_path} ...")
        
#         # 输出文件名，防止覆盖原文件，确认无误后可自行重命名
#         output_path = file_path.replace(".jsonl", "_cleaned.jsonl")
        
#         file_modified_count = 0
        
#         with open(file_path, 'r', encoding='utf-8') as f_in, \
#              open(output_path, 'w', encoding='utf-8') as f_out:
            
#             for line_idx, line in enumerate(f_in):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     data_obj = json.loads(line)
#                     has_change = False
                    
#                     if "data" in data_obj and isinstance(data_obj["data"], list):
#                         for message in data_obj["data"]:
#                             # 仅处理 assistant 回复
#                             if message.get("role") == "assistant" and "content" in message:
#                                 original_content = message["content"]
#                                 new_content = clean_empty_params(original_content)
                                
#                                 if new_content != original_content:
#                                     message["content"] = new_content
#                                     has_change = True
                    
#                     # 写入处理后的数据
#                     f_out.write(json.dumps(data_obj, ensure_ascii=False) + "\n")
                    
#                     if has_change:
#                         file_modified_count += 1
                        
#                 except json.JSONDecodeError:
#                     print(f"警告: 第 {line_idx+1} 行 JSON 解析失败，已跳过。")
#                     continue
        
#         print(f"完成. 修改行数: {file_modified_count}. 保存至: {output_path}")
#         total_files += 1
#         total_modified_lines += file_modified_count

#     print(f"\n全部完成! 处理文件数: {total_files}, 总修改行数: {total_modified_lines}")

# if __name__ == "__main__":
#     process_files()


import json
import re
import os

# 定义需要处理的文件列表
TRAIN_DATA_FILES = [
    "data/final_data/单轮单指令_冒烟.jsonl",
    "data/final_data/单轮单指令_增强.jsonl",
    "data/final_data/多轮单指令_冒烟.jsonl",
    "data/final_data/多轮单指令_增强.jsonl",
    "data/final_data/单轮多指令_增强.jsonl",
    "data/final_data/单轮多指令_合成.jsonl",
    "data/final_data/多轮多指令_增强.jsonl",
    "data/final_data/决赛冒烟集.jsonl",
]

def clean_empty_params(content):
    """
    清洗content字符串中的空参数。
    兼容三种空值格式：
    1. 标准格式: Key=""
    2. 转义格式: Key=\"\" (针对嵌套引用数据)
    3. 空列表:   Key=[]
    """
    if not content or not isinstance(content, str):
        return content

    # 定义基础正则组件
    # param_key: 匹配参数名 (字母数字下划线)
    param_key = r'[a-zA-Z0-9_]+'
    
    # empty_val: 核心修改点
    # (?: ... ) 是非捕获组
    # ""       匹配标准的空引号
    # |        或
    # \\"\\"   匹配转义后的空引号 (即文本里的 \"\")
    # |        或
    # \[\]     匹配空列表 (需要转义中括号)
    empty_val = r'(?:""|\\"\\"|\[\])'

    # --- 组合正则模式 ---

    # Pattern 1: 位于开头或中间的参数 (Key="", 或 Key=[], )
    # 匹配: 参数名 + 等号 + 空值 + 逗号
    pattern_start_or_mid = f'{param_key}\\s*=\\s*{empty_val}\\s*,\\s*'
    
    # Pattern 2: 位于末尾的参数 (, Key="" 或 , Key=[])
    # 匹配: 逗号 + 参数名 + 等号 + 空值
    pattern_end = f',\\s*{param_key}\\s*=\\s*{empty_val}'
    
    # Pattern 3: 唯一的参数 ((Key="") 或 (Key=[]))
    # 匹配: 左括号后 + 参数名 + 等号 + 空值 + 右括号前
    pattern_sole = f'(?<=\\()\\s*{param_key}\\s*=\\s*{empty_val}\\s*(?=\\))'

    # --- 执行替换 ---
    
    # 1. 先删掉 "Key="", " 或 "Key=[], " 这种形式
    cleaned_content = re.sub(pattern_start_or_mid, '', content)
    
    # 2. 再删掉 ", Key=""" 或 ", Key=[]" 这种形式
    cleaned_content = re.sub(pattern_end, '', cleaned_content)
    
    # 3. 最后处理剩下的唯一参数情况
    cleaned_content = re.sub(pattern_sole, '', cleaned_content)

    return cleaned_content

def process_files():
    total_files = 0
    total_modified_lines = 0

    for file_path in TRAIN_DATA_FILES:
        if not os.path.exists(file_path):
            print(f"[跳过] 文件不存在: {file_path}")
            continue

        print(f"正在处理: {file_path} ...")
        
        # 输出文件名，防止覆盖原文件，确认无误后可自行重命名
        output_path = file_path.replace(".jsonl", "_cleaned.jsonl")
        
        file_modified_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line_idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                try:
                    data_obj = json.loads(line)
                    has_change = False
                    
                    if "data" in data_obj and isinstance(data_obj["data"], list):
                        for message in data_obj["data"]:
                            # 仅处理 assistant 回复
                            if message.get("role") == "assistant" and "content" in message:
                                original_content = message["content"]
                                new_content = clean_empty_params(original_content)
                                
                                if new_content != original_content:
                                    message["content"] = new_content
                                    has_change = True
                    
                    # 写入处理后的数据
                    f_out.write(json.dumps(data_obj, ensure_ascii=False) + "\n")
                    
                    if has_change:
                        file_modified_count += 1
                        
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_idx+1} 行 JSON 解析失败，已跳过。")
                    continue
        
        print(f"完成. 修改行数: {file_modified_count}. 保存至: {output_path}")
        total_files += 1
        total_modified_lines += file_modified_count

    print(f"\n全部完成! 处理文件数: {total_files}, 总修改行数: {total_modified_lines}")

if __name__ == "__main__":
    process_files()