"""
长文本切片模块
- split_long_text: 按段落/句子切分，每chunk不超过max_len字符
"""
import re

from config import CHUNK_SIZE


def split_long_text(text: str, max_len: int = CHUNK_SIZE) -> list[str]:
    """
    将长文本切分成多个chunk，每chunk不超过max_len字符
    优先在段落（\\n\\n）处切，再在句号处切，最后兜底强制截断
    """
    if len(text) <= max_len:
        return [text]

    chunks = []

    # 按段落分割
    paragraphs = re.split(r"\n\n+", text)

    for para in paragraphs:
        if not para.strip():
            continue

        # 段落能直接放入，直接追加
        if len(para) <= max_len:
            chunks.append(para)
            continue

        # 段落超长，按句子切
        sentences = re.split(r"(?<=[。！？.!?])", para)
        current = ""

        for sent in sentences:
            if not sent.strip():
                continue

            # 单句不超过max_len，可以累积
            if len(sent) <= max_len:
                if len(current) + len(sent) <= max_len:
                    current += sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            else:
                # 单句超长，先保存current
                if current:
                    chunks.append(current)
                    current = ""
                # 强制按max_len截断
                for i in range(0, len(sent), max_len):
                    chunks.append(sent[i:i+max_len])

        # 剩余current
        if current:
            chunks.append(current)

    return chunks
