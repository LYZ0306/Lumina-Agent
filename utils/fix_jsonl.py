import re
import sys
import os

def fix_line(line):
    """
    修复单行数据：查找 "content": "..." 结构，
    并转义其内部除结尾引号外的所有引号。
    """
    # 使用正则分割行，保留分隔符
    # 模式：("content"\s*:\s*") 匹配 "content": " (允许空格)
    parts = re.split(r'("content"\s*:\s*")', line)
    
    # 如果没有找到 content 字段，原样返回
    if len(parts) < 3:
        return line

    fixed_parts = [parts[0]] # 放入第一段（content之前的内容）

    # re.split 分割后，列表结构为：[前缀, 分隔符, 后缀(含content值...), 分隔符, 后缀...]
    # 我们从索引 1 开始遍历，每次处理一对 (分隔符, 后缀)
    for i in range(1, len(parts), 2):
        delimiter = parts[i]       # 例如: "content": "
        rest = parts[i+1]          # 例如: 小艺...ActionType="True")"}], "difficulty"...
        
        # 寻找 content 值的结束位置
        # 逻辑：在 content 值内部，引号是杂乱的，但在 content 结束时，引号后面必定紧跟 JSON 结构符号
        # 在你的数据结构中，content 通常是 {"role":..., "content":...} 的最后一项
        # 所以结束引号后面通常紧跟 '}' (结束对象)
        # 正则解释：寻找一个双引号，它后面紧跟着可能的空白字符，然后是 '}'
        match = re.search(r'"\s*}', rest)
        
        if match:
            end_quote_index = match.start()
            
            # 提取 content 内部的文本
            content_body = rest[:end_quote_index]
            
            # 提取 content 结束后的剩余部分（包括结束引号和后面的 }...）
            suffix = rest[end_quote_index:]
            
            # 【核心修复】：将 content_body 中所有未被转义的引号 " 替换为 \"
            # (?<!\\) 是反向否定预查，确保不重复转义已经转义过的 \"
            fixed_body = re.sub(r'(?<!\\)"', r'\"', content_body)
            
            fixed_parts.append(delimiter)
            fixed_parts.append(fixed_body + suffix)
        else:
            # 如果找不到符合结构的结束引号（极罕见情况），保留原样以防破坏
            fixed_parts.append(delimiter)
            fixed_parts.append(rest)
            
    return "".join(fixed_parts)

def main():
    # 配置输入输出文件
    input_file = "data/final_data/多轮多指令_增强.jsonl"  # 将此处改为你的文件名
    output_file = "data/final_data/多轮多指令_增强_fixed.jsonl"

    # 如果命令行传参，则优先使用命令行参数
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        print("用法: python fix_content_quotes.py source.jsonl target.jsonl")
        return

    print(f"正在修复 {input_file} ...")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            fixed_line = fix_line(line)
            f_out.write(fixed_line + '\n')
            
            if line != fixed_line:
                count += 1

    print(f"完成！已修复 {count} 行，保存至 {output_file}")

if __name__ == "__main__":
    main()