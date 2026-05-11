"""
Session JSONL 解析 + 消息过滤
- parse_session_file: 解析整个文件，返回有效消息列表
- read_new_messages: 从指定位置读取新增消息（实时监控用）
- is_valid_conversation_message: 判断单条消息是否有效
- get_active_sessions: 获取当前活跃session文件列表
"""
import glob
import json
import os
import re

from config import SESSIONS_DIR


# ============ 过滤规则 ============

FILTER_PATTERNS = [
    re.compile(r"^\[CONTEXT COMPACTION"),
    re.compile(r"^\[System note:"),        # 所有System note消息
    re.compile(r"^［"),                      # 全角括号开头的消息
    re.compile(r"^📦 Preflight compression:"),
    re.compile(r"^⚡ Interrupting current task"),
    re.compile(r"^\[Command interrupted\]"),
]


def is_valid_conversation_message(obj: dict) -> tuple[bool, str]:
    """
    判断消息是否有效（用于存入向量库）
    返回 (是否有效, 跳过原因)
    """
    role = obj.get("role", "")

    # 只接受 user 和 assistant
    if role not in ("user", "assistant"):
        return False, f"role={role}"

    content = obj.get("content", "")
    if not content or not content.strip():
        return False, "empty_content"

    # 过滤系统级内容
    for pattern in FILTER_PATTERNS:
        if pattern.match(content.strip()):
            return False, f"filtered_pattern:{pattern.pattern}"

    return True, ""


def clean_text_for_vector(text: str) -> str:
    """
    清洗文本用于向量转换
    - 去除 <tool_call>...</tool_call> 块
    - 去除 <reasoning>...</reasoning> 块
    - 去除 finish_reason 等元信息
    """
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text)
    text = re.sub(r"finish_reason:[^\n]*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_session_file(filepath: str) -> tuple[list[dict], dict]:
    """
    解析 session JSONL 文件
    返回 (有效消息列表, 跳过原因统计)
    """
    messages = []
    skip_reasons = {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    is_valid, reason = is_valid_conversation_message(obj)
                    if not is_valid:
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        continue
                    messages.append({
                        "role": obj.get("role"),
                        "content": obj.get("content", ""),
                        "timestamp": obj.get("timestamp", ""),
                    })
                except json.JSONDecodeError:
                    continue

        if skip_reasons:
            print(f"  -> 跳过: {skip_reasons}")

    except Exception as e:
        print(f"  [文件读取错误] {filepath}: {e}")

    return messages, skip_reasons


def read_new_messages(filepath: str, last_pos: int, last_inode: int | None = None) -> tuple[list[dict], int, int]:
    """
    从文件中 last_pos 位置之后读取新消息
    检测到文件被截断/轮转时自动重置读取位置
    返回 (消息列表, 新的文件位置, 当前inode)
        - 若文件被截断(last_pos > 文件大小)：从0重新读，返回新位置=0
        - 若inode变化：说明文件轮转，从0重新读
    """
    messages = []
    try:
        st = os.stat(filepath)
        current_inode = st.st_ino
        file_size = st.st_size

        # 检测文件被截断或轮转：seek位置超出文件大小，或inode变化
        if last_pos > file_size:
            print(f"  [警告] 文件被截断 {os.path.basename(filepath)}: 记录位置={last_pos} > 实际大小={file_size}，重置到0")
            last_pos = 0
        elif last_inode is not None and last_inode != current_inode:
            print(f"  [警告] 文件轮转 {os.path.basename(filepath)}: inode {last_inode} -> {current_inode}，从头开始")
            last_pos = 0

        with open(filepath, "r", encoding="utf-8") as f:
            f.seek(last_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    is_valid, _ = is_valid_conversation_message(obj)
                    if not is_valid:
                        continue
                    messages.append({
                        "role": obj.get("role"),
                        "content": obj.get("content", ""),
                        "timestamp": obj.get("timestamp", ""),
                    })
                except json.JSONDecodeError:
                    continue
            new_pos = f.tell()
            return messages, new_pos, current_inode
    except Exception as e:
        print(f"  [读取错误] {filepath}: {e}")
    return [], last_pos, 0


def get_active_sessions(limit: int = 3) -> list[str]:
    """
    获取最近活跃的session文件（按修改时间倒序）
    """
    files = glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl"))
    if not files:
        return []
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[:limit]
