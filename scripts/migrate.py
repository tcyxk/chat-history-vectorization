#!/usr/bin/env python3
"""
聊天记录历史迁移脚本
用法: python3 migrate.py

流程:
  1. 查询Qdrant最大point_id，作为全局递增起点
  2. 在session文件中定位起始位置（跳过已处理的消息）
  3. 逐条处理: 文本清洗 → 向量生成 → 写入Qdrant(wait=true) → 验证 → 下一条
  4. 每条消息处理完保存checkpoint

存储: Qdrant WAL模式，Linux ext4真实fsync，断电最多丢1条
"""
import fcntl
import glob
import json
import os
import signal
import sys
import time

# ============ 单例锁 ============
LOCK_FILE = os.path.expanduser("~/.hermes/migrate.lock")


def acquire_lock():
    """获取进程锁，确保只有一个migrate实例运行"""
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd
    except BlockingIOError:
        print("[错误] migrate.py已在运行，请先停止旧进程")
        print("  执行: pkill -f 'python3 migrate.py'")
        lock_fd.close()
        sys.exit(1)


# ============ 导入项目内模块 ============
from config import (
    AGENT_NAME,
    CHUNK_SIZE,
    COLLECTION_NAME,
    HERMES_HOME,
    MAX_TEXT_LEN,
    MIGRATE_CHECKPOINT,
    SESSIONS_DIR,
)
from embedding import get_embedding, verify_embedding
from session_parser import clean_text_for_vector, parse_session_file
from text_splitter import split_long_text
from qdrant_client import (
    upsert_points,
    verify_written_point,
    scroll,
    get_points_count,
    get_collection_info,
    create_collection,
)


g_stop = False


def signal_handler(signum, frame):
    global g_stop
    print("\n[信号] 收到停止信号，等待当前条处理完...")
    g_stop = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============ Checkpoint ============

def load_checkpoint() -> dict:
    if os.path.exists(MIGRATE_CHECKPOINT):
        with open(MIGRATE_CHECKPOINT, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict):
    with open(MIGRATE_CHECKPOINT, "w") as f:
        json.dump(data, f)


# ============ 定位逻辑 ============

def get_qdrant_last_record() -> tuple[int, str | None, str | None]:
    """
    查询Qdrant中最大point_id，返回 (next_point_id, last_session_file, last_timestamp)
    next_point_id = max_point_id + 1 = 全局递增起点
    """
    count = get_points_count()
    if count == 0:
        return 0, None, None

    # scroll 取所有记录找最大
    all_points, ok = scroll(page_size=1000)
    if not all_points:
        return 0, None, None
    if not ok:
        print(f"  [警告] Qdrant查询不完整，可能影响定位")

    max_point = max(all_points, key=lambda p: p["id"])
    max_id = max_point["id"]
    payload = max_point.get("payload", {})
    return max_id + 1, payload.get("session_file"), payload.get("timestamp")


def find_start_position(next_msg_idx: int, last_file: str | None) -> tuple[str, int, int, int]:
    """
    根据next_msg_idx在session文件中找到起始位置

    返回: (起始文件路径, 文件索引, 文件内消息索引, current_msg_idx)
    """
    session_files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")))
    if not session_files:
        return "", 0, 0, 0

    if next_msg_idx == 0:
        return session_files[0], 0, 0, 0

    # 定位到last_file，加速查找
    start_file_idx = 0
    if last_file:
        for i, f in enumerate(session_files):
            if last_file in f:
                start_file_idx = i
                break

    # 逐文件累积消息数，找到next_msg_idx所在的文件
    global_count = 0
    for file_idx in range(start_file_idx, len(session_files)):
        messages, _ = parse_session_file(session_files[file_idx])
        file_msg_count = len(messages)
        if global_count + file_msg_count > next_msg_idx:
            msg_i = next_msg_idx - global_count
            return session_files[file_idx], file_idx, msg_i, next_msg_idx
        global_count += file_msg_count

    # 超出总消息数，从最后一个文件末尾继续
    if session_files:
        last_file_idx = len(session_files) - 1
        messages, _ = parse_session_file(session_files[last_file_idx])
        return session_files[last_file_idx], last_file_idx, len(messages), next_msg_idx
    return "", 0, 0, next_msg_idx


# ============ 主迁移 ============

def migrate():
    # 获取单例锁
    lock_fd = acquire_lock()

    print("=" * 60)
    print(f"聊天记录迁移 [{AGENT_NAME}]")
    print("=" * 60)
    print(f"Session目录: {SESSIONS_DIR}")
    print(f"向量库: Qdrant Cloud")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    try:
        # 确保Collection存在
        print("[初始化] 检查Collection...")
        info = get_collection_info()
        if info.get("result") is None:
            print("  -> Collection不存在，创建中...")
            create_collection()
            print("  -> 创建完成")
        else:
            print(f"  -> Collection状态: {info['result'].get('status')}")
        print()

        # 步骤1: 查询Qdrant最大point_id，作为全局递增起点
        print("[步骤1] 查询Qdrant最后记录...")
        next_point_id, last_file, last_ts = get_qdrant_last_record()
        print(f"  -> Qdrant最大point_id={next_point_id - 1}，全局递增起点={next_point_id}")

        # 步骤2: 定位起始位置
        print("[步骤2] 定位起始位置...")
        start_file, start_file_idx, resume_msg_i, resume_msg_idx = find_start_position(
            next_point_id, last_file
        )
        if start_file:
            print(f"  -> 从 {os.path.basename(start_file)} 消息索引={resume_msg_i} 开始")
            print(f"  -> 全局消息索引={resume_msg_idx}")
        else:
            print("  -> 无session文件")
        print()

        checkpoint = load_checkpoint()
        total_vectors = 0
        total_errors = 0
        start_time = time.time()
        current_msg_idx = resume_msg_idx
        session_files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")))
        found_resume_point = False

        for file_idx in range(start_file_idx, len(session_files)):
            if g_stop:
                break

            filepath = session_files[file_idx]
            filename = os.path.basename(filepath)
            print(f"[文件 {file_idx+1}/{len(session_files)}] {filename}")

            messages, _ = parse_session_file(filepath)
            if not messages:
                print("  -> 无有效消息，跳过")
                continue

            # 确定起始消息索引
            if not found_resume_point:
                if file_idx == start_file_idx:
                    start_msg_i = resume_msg_i
                else:
                    start_msg_i = 0
                if start_msg_i >= len(messages):
                    found_resume_point = True
                    continue
            else:
                start_msg_i = 0

            print(f"  -> {len(messages)} 条消息，从第 {start_msg_i} 条开始")

            for msg_i in range(start_msg_i, len(messages)):
                if g_stop:
                    break

                msg = messages[msg_i]
                original_content = msg["content"]
                timestamp = msg["timestamp"]

                # 清洗 + 截断 + 切片
                clean_content = clean_text_for_vector(original_content)
                if len(clean_content) > MAX_TEXT_LEN:
                    clean_content = clean_content[:MAX_TEXT_LEN]
                chunks = split_long_text(clean_content, CHUNK_SIZE)
                print(f"  [{current_msg_idx}] role={msg['role']}, chunks={len(chunks)}, len={len(clean_content)}")

                for chunk_idx, chunk_text in enumerate(chunks):
                    # 生成向量
                    vector = get_embedding(chunk_text)
                    if vector is None:
                        print(f"    [!] 向量生成失败")
                        total_errors += 1
                        continue

                    # 验证: 重新生成并比对
                    if not verify_embedding(chunk_text, vector):
                        print(f"    [!] 验证失败，重试...")
                        vector = get_embedding(chunk_text)
                        if vector is None or not verify_embedding(chunk_text, vector):
                            print(f"    [!] 验证再次失败，跳过")
                            total_errors += 1
                            continue

                    # 全局递增ID，每个chunk独立point，不重复
                    point_id = next_point_id
                    next_point_id += 1

                    points = [{
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "session_file": filename,
                            "timestamp": timestamp,
                            "role": msg["role"],
                            "content": original_content,
                            "msg_index": current_msg_idx,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                        }
                    }]
                    ok = upsert_points(points, wait=True)
                    if not ok:
                        print(f"    [!] Qdrant写入失败")
                        total_errors += 1
                        continue

                    # 写后验证: 从磁盘读取确认
                    ok_verify, err = verify_written_point(point_id, vector)
                    if not ok_verify:
                        print(f"    [!] 验证未通过: {err}")
                        total_errors += 1
                        continue

                    total_vectors += 1

                # 一条消息的所有chunk都处理完，保存checkpoint
                checkpoint["last_msg_index"] = current_msg_idx
                checkpoint["last_session_file"] = filename
                checkpoint["last_timestamp"] = timestamp
                checkpoint["next_point_id"] = next_point_id
                save_checkpoint(checkpoint)

                # 写入后休息，给内存释放时间
                time.sleep(1.5)

                current_msg_idx += 1

            found_resume_point = True

        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("迁移完成!")
        print(f"  处理消息: {current_msg_idx - resume_msg_idx}")
        print(f"  成功向量化: {total_vectors}")
        print(f"  生成错误: {total_errors}")
        print(f"  耗时: {elapsed:.1f} 秒")
        print("=" * 60)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        print("[锁] 已释放")


if __name__ == "__main__":
    migrate()
