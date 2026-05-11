"""
Qdrant 客户端封装
- upsert: 单条/批量写入，wait=true等待WAL落盘
- verify_written_point: 从磁盘读取point验证
- get_collection_info / scroll / delete_collection
"""
import json
import subprocess
import time

from config import QDRANT_URL, COLLECTION_NAME


def _curl(method: str, path: str, data: dict | None = None, timeout: int = 60) -> dict:
    """通用curl封装"""
    cmd = ["curl", "-s", "-X", method, f"{QDRANT_URL}{path}"]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return json.loads(result.stdout)


def upsert_points(points: list[dict], wait: bool = True) -> bool:
    """
    写入points到Qdrant
    wait=True: 等待WAL落盘后才返回
    """
    if not points:
        return True
    for attempt in range(3):
        try:
            resp = _curl("PUT", f"/collections/{COLLECTION_NAME}/points", {
                "points": points,
                "wait": wait,
            }, timeout=60)
            if resp.get("status") == "ok":
                return True
            print(f"  [Qdrant错误] {resp}")
        except Exception as e:
            print(f"  [Qdrant错误] attempt {attempt+1}: {e}")
        time.sleep(1)
    return False


def verify_written_point(point_id: int, expected_vector: list[float], max_retries: int = 3) -> tuple[bool, str]:
    """
    从Qdrant读取已落盘的point，验证向量前10维匹配
    验证通过返回(True, "ok")，失败返回(False, 错误原因)
    """
    for attempt in range(max_retries):
        try:
            resp = _curl("GET", f"/collections/{COLLECTION_NAME}/points/{point_id}", timeout=30)
            if resp.get("status") == "ok" and "result" in resp:
                point = resp["result"]
                stored_vector = point.get("vector", [])
                if stored_vector and all(
                    abs(a - b) < 0.01
                    for a, b in zip(expected_vector[:10], stored_vector[:10])
                ):
                    return True, "ok"
        except Exception as e:
            print(f"  [验证错误] attempt {attempt+1}: {e}")
        time.sleep(1)
    return False, f"point_id={point_id} 验证失败"


def get_collection_info() -> dict:
    """获取Collection信息"""
    try:
        return _curl("GET", f"/collections/{COLLECTION_NAME}")
    except Exception as e:
        return {"error": str(e)}


def get_points_count() -> int:
    """获取Collection中的points数量"""
    try:
        info = get_collection_info()
        return info.get("result", {}).get("points_count", 0)
    except Exception:
        return 0


def scroll(page_size: int = 100, offset: int | None = None) -> tuple[list[dict], bool]:
    """
    滚动获取points，用于查找最后一条
    自动翻页，直到取完所有记录

    返回: (points列表, 是否完整获取)
    - (points, True) = 正常翻页完成
    - (points, False) = 中途出错，返回已获取的部分
    """
    all_points = []
    current_offset = offset
    while True:
        payload = {"limit": page_size}
        if current_offset is not None:
            payload["offset"] = current_offset
        try:
            resp = _curl("POST", f"/collections/{COLLECTION_NAME}/points/scroll", payload)
            points = resp.get("result", {}).get("points", [])
            all_points.extend(points)
            next_offset = resp.get("result", {}).get("next_page_offset")
            if next_offset is None or len(points) < page_size:
                return all_points, True  # 翻页完成
            current_offset = next_offset
        except Exception as e:
            print(f"  [错误] scroll翻页异常: {e}")
            return all_points, False  # 中途出错，返回已获取的


def delete_collection(name: str | None = None) -> bool:
    """删除Collection"""
    target = name or COLLECTION_NAME
    try:
        resp = _curl("DELETE", f"/collections/{target}")
        return resp.get("result", False)
    except Exception:
        return False


def create_collection(
    name: str | None = None,
    vector_size: int = 1024,
    distance: str = "Cosine",
) -> bool:
    """创建Collection（若已存在则跳过）"""
    target = name or COLLECTION_NAME
    try:
        resp = _curl("PUT", f"/collections/{target}", {
            "vectors": {
                "size": vector_size,
                "distance": distance,
            },
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100,
            },
            "on_disk_payload": True,
        })
        return resp.get("result", False)
    except Exception:
        return False


def search_points(query_vector: list[float], top_k: int = 5, filter_cond: dict | None = None) -> list[dict]:
    """向量检索"""
    payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
    }
    if filter_cond:
        payload["filter"] = filter_cond
    try:
        resp = _curl("POST", f"/collections/{COLLECTION_NAME}/points/search", payload)
        return resp.get("result", [])
    except Exception:
        return []
