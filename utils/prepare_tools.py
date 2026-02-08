import json

# --- 配置 ---
# 输入：您的冒-烟测试集JSONL文件路径
JSONL_FILE_PATH = "/data1/lyz/Agent_HW/demo/data/决赛指令集.jsonl" 

# 输出：将结果保存到的文件名
OUTPUT_FILE_PATH = "hardcoded_tools_single_line_2.txt"

# --- 脚本主逻辑 ---
def extract_and_save_tool_definitions_single_line(input_path, output_path):
    """
    读取jsonl文件，提取唯一的、精简后的工具定义，
    并将它们作为每项占一行的Python列表字符串写入到输出文件中。
    """
    unique_tools = {}

    print(f"Reading from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    intent_name = data.get("intent_name")

                    if intent_name and intent_name not in unique_tools:
                        tool_definition = {
                            "intent_name": data.get("intent_name"),
                            "intent_description": data.get("intent_description"),
                            "slots": data.get("slots"),
                        }
                        tool_definition = {k: v for k, v in tool_definition.items() if v is not None}
                        unique_tools[intent_name] = tool_definition

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line {line_num}: {line.strip()}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        print("Please ensure the path is correct.")
        return

    tool_definitions_list = list(unique_tools.values())
    print(f"Found {len(tool_definitions_list)} unique tool definitions.")

    # --- 将格式化好的列表写入文件 ---
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write("[\n")
            for i, tool in enumerate(tool_definitions_list):
                # --- 核心改动：移除所有换行符和不必要的空格，生成单行字符串 ---
                # separators=(',', ':') 移除了逗号和冒号后的空格，进一步压缩
                formatted_entry = json.dumps(tool, ensure_ascii=False, separators=(',', ':'))
                
                # 写入带缩进的单行条目
                outfile.write(f"    {formatted_entry}")
                
                if i < len(tool_definitions_list) - 1:
                    outfile.write(",\n")
                else:
                    outfile.write("\n")
            outfile.write("]")
        
        print(f"Successfully processed the file.")
        print(f"==> The single-line formatted Python list has been saved to: {output_path}")

    except IOError as e:
        print(f"Error: Could not write to the file '{output_path}'. Reason: {e}")


if __name__ == "__main__":
    extract_and_save_tool_definitions_single_line(JSONL_FILE_PATH, OUTPUT_FILE_PATH)    