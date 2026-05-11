"""
llama-embedding 调用封装
- base64编码原始文本后传给llama-embedding
- 解析输出格式: embedding 0: <1024个浮点数>
- 向量验证（重新生成并比对前10维）
"""
import base64
import json
import os
import re
import subprocess
import time

from config import (
    LLAMA_EMBEDDING_BIN,
    BGE_M3_MODEL,
    LLAMA_LD_LIBRARY_PATH,
    VECTOR_DIM,
)


def _get_env():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_LD_LIBRARY_PATH
    return env


def get_embedding(text: str) -> list[float] | None:
    """
    将文本base64编码后传给llama-embedding，返回向量列表或None
    输出格式: embedding 0: <1024个浮点数>
    """
    if not text or not text.strip():
        return None

    b64_text = base64.b64encode(text.encode("utf-8")).decode("ascii")

    try:
        result = subprocess.run(
            [
                LLAMA_EMBEDDING_BIN,
                "-m", BGE_M3_MODEL,
                "-b", "512",      # context batch
            ],
            input=b64_text.encode("ascii"),
            capture_output=True,
            timeout=60,
            env=_get_env(),
        )
        output = result.stdout.decode("utf-8", errors="ignore")
        stderr = result.stderr.decode("utf-8", errors="ignore")

        if result.returncode != 0:
            print(f"  [llama错误] returncode={result.returncode}, stderr={stderr[:200]}")
            return None

        # 解析 "embedding 0: -0.0123, -0.0456, ..."
        match = re.search(r"embedding 0:\s*(.+?)(?:\n|$)", output, re.DOTALL)
        if not match:
            print(f"  [llama解析错误] 输出不含'embedding 0:', output前200: {output[:200]}")
            return None

        parts = re.split(r'[,\s]+', match.group(1).strip())
        if len(parts) < VECTOR_DIM:
            print(f"  [llama解析错误] 向量维度不足: {len(parts)} < {VECTOR_DIM}")
            return None

        vector = [float(x.strip()) for x in parts[:VECTOR_DIM]]
        return vector

    except subprocess.TimeoutExpired:
        print(f"  [llama超时] text长度={len(text)}")
        return None
    except Exception as e:
        print(f"  [llama异常] {e}")
        return None


def verify_embedding(text: str, vector: list[float]) -> bool:
    """
    验证向量：再次转换并比对前10维，偏差<0.01则通过
    """
    v2 = get_embedding(text)
    if v2 is None:
        return False
    return all(abs(a - b) < 0.01 for a, b in zip(vector[:10], v2[:10]))
