#!/usr/bin/env python3
"""
聊天记录实时监控脚本
用法: python3 realtime.py

流程:
  1. 启动时从Qdrant最后一条定位起始位置
  2. 每隔POLL_INTERVAL秒轮询活跃session文件
  3. 逐条处理: 读取 → base64编码 → 向量 → 写入Qdrant → 磁盘验证 → checkpoint
  4. 处理完进入下一轮休眠，不累积
"""
import fcntl
import json
import os
import signal
import sys
import time

# ============ 单例锁 ============
LOCK_FILE = os.path.expanduser("~/.hermes/realtime.lock")

def acquire_lock():
    """获取进程锁，确保只有一个realtime实例运行"""
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd
    except BlockingIOError:
        print("[错误] realtime.py已在运行，请先停止旧进程")
        print("  执行: pkill -f 'python3 realtime.py'")
        lock_fd.close()
        sys.exit(1)

from config import (
    AGENT_NAME,
    CHUNK_SIZE,
    COLLECTION_NAME,
    HERMES_HOME,
    MAX_TEXT_LEN,
    POLL_INTERVAL,
    QDRANT_URL,
    REALTIME_CHECKPOINT,
    SESSIONS_DIR,
)
from embedding import get_embedding, verify_embedding
from qdrant_client import (
    create_collection,
    get_collection_info,
    scroll,
    upsert_points,
    verify_written_point,
)
from session_parser import (
    clean_text_for_vector,
    get_active_sessions,
    read_new_messages,
)
from text_splitter import split_long_text


g_running = True
g_stop = False


def signal_handler(signum, frame):
    global g_running, g_stop
    print("\n[信号] 收到停止信号，等待当前条处理完...")
    g_running = False
    g_stop = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============ Checkpoint ============

def load_checkpoint() -> dict:
    if os.path.exists(REALTIME_CHECKPOINT):
        with open(REALTIME_CHECKPOINT, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict):
    with open(REALTIME_CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)


# ============ 定位逻辑 ============

def get_qdrant_last_record():
    """查询Qdrant中point_id最大的一条记录"""
    try:
        all_points, ok = scroll(page_size=1000)
        if not all_points:
            return None, None, None, None
        if not ok:
            print(f"  [警告] Qdrant查询不完整，可能影响定位")
        max_pt = max(all_points, key=lambda p: p.get("id", -1))
        max_id = max_pt.get("id", -1)
        payload = max_pt.get("payload", {})
        return (
            payload.get("timestamp"),
            payload.get("msg_index"),
            payload.get("session_file"),
            max_id,
        )
    except Exception as e:
        print(f"  [Qdrant查询错误] {e}")
        return None, None, None, None


def find_start_position(last_ts, last_idx, last_file, last_point_id):
    """返回(文件路径, 文件索引, 起始msg_idx, next_point_id)"""
    import glob as _glob
    all_files = sorted(_glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")))

    if not all_files:
        return None, 0, 0, last_point_id if last_point_id is not None else 0

    # 如果有checkpoint记录的文件，在所有文件中查找
    if last_file:
        matching = [f for f in all_files if last_file in os.path.basename(f)]
        if matching:
            file_idx = all_files.index(matching[0])
            return matching[0], file_idx, (last_idx or 0), (last_point_id if last_point_id is not None else 0)
        # checkpoint记录的文件不在当前文件列表中，报警
        print(f"  [警告] checkpoint文件未找到: {last_file}，重新定位...")

    # 否则取最近活跃的
    files = get_active_sessions(limit=10)
    if not files:
        return all_files[0], 0, 0, last_point_id if last_point_id is not None else 0

    file_idx = next((i for i, f in enumerate(files) if last_file in f), 0)
    return files[file_idx], file_idx, (last_idx or 0), (last_point_id if last_point_id is not None else 0)


# ============ 主监控 ============

def main():
    global g_running, g_stop

    # 获取单例锁
    lock_fd = acquire_lock()

    print("=" * 60)
    print(f"聊天记录实时监控 [{AGENT_NAME}]")
    print("=" * 60)
    print(f"监控目录: {SESSIONS_DIR}")
    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    try:
        # 确保Collection存在
        info = get_collection_info()
        if info.get("result") is None:
            print("[初始化] 创建Collection...")
            create_collection()
            print("  -> 完成")
        else:
            print(f"[Collection] 状态: {info['result'].get('status')}")

        # 启动时定位
        print("[初始化] 查询Qdrant最后记录...")
        last_ts, last_idx, last_file, last_point_id = get_qdrant_last_record()
        if last_ts:
            print(f"  -> 库中最后: ts={last_ts}, idx={last_idx}, point_id={last_point_id}")
        else:
            print("  -> Qdrant为空")

        start_file, start_file_idx, start_msg_idx, next_point_id = find_start_position(
            last_ts, last_idx, last_file, last_point_id
        )
        print(f"  -> 起始: {os.path.basename(start_file) if start_file else 'None'}, msg_idx={start_msg_idx}, next_point_id={next_point_id}")
        print()

        checkpoint = load_checkpoint()
        current_msg_idx = start_msg_idx if start_file else 0
        total_errors = 0

        # checkpoint中的next_point_id可能被清空或不准确，以Qdrant真实最大ID为准
        if last_point_id is not None:
            next_point_id = last_point_id + 1
        elif "next_point_id" in checkpoint:
            next_point_id = checkpoint["next_point_id"]
        # 否则用 find_start_position 返回的初始值

        # 文件位置+inode追踪
        file_positions = checkpoint.get("file_positions", {})
        file_inodes = checkpoint.get("file_inodes", {})  # filename -> inode

        while g_running and not g_stop:
            try:
                session_files = get_active_sessions(limit=10)
                if not session_files:
                    time.sleep(POLL_INTERVAL)
                    continue
                total_success = 0

                for file_idx, filepath in enumerate(session_files):
                    if not g_running:
                        break

                    filename = os.path.basename(filepath)
                    last_pos = file_positions.get(filename, 0)
                    last_inode = file_inodes.get(filename)

                    messages, new_pos, current_inode = read_new_messages(filepath, last_pos, last_inode)
                    if not messages:
                        # 文件可能刚轮转（inode变化），新文件从头开始 -> 重置position
                        # 如果position没变且inode没变，说明是正常EOF，无需更新
                        file_inodes[filename] = current_inode
                        if new_pos > 0:
                            file_positions[filename] = new_pos
                        continue

                    # 更新追踪信息
                    file_positions[filename] = new_pos
                    file_inodes[filename] = current_inode

                    for i, msg in enumerate(messages):
                        if not g_running:
                            break

                        msg_global_idx = current_msg_idx + i
                        if file_idx == start_file_idx and msg_global_idx < start_msg_idx:
                            continue

                        original_content = msg["content"]
                        clean_content = clean_text_for_vector(original_content)
                        if len(clean_content) > MAX_TEXT_LEN:
                            clean_content = clean_content[:MAX_TEXT_LEN]

                        chunks = split_long_text(clean_content, CHUNK_SIZE)

                        for chunk_idx, chunk_text in enumerate(chunks):
                            if not g_running:
                                break

                            # 生成向量
                            vector = get_embedding(chunk_text)
                            if vector is None:
                                print(f"  [!] 向量失败: idx={msg_global_idx}")
                                continue

                            # 验证: 重新生成比对
                            if not verify_embedding(chunk_text, vector):
                                vector = get_embedding(chunk_text)
                                if vector is None or not verify_embedding(chunk_text, vector):
                                    print(f"  [!] 验证失败: idx={msg_global_idx}")
                                    total_errors += 1
                                    continue

                            # 简单递增ID（不再用 msg_idx*1000）
                            point_id = next_point_id
                            next_point_id += 1
                            point = {
                                "id": point_id,
                                "vector": vector,
                                "payload": {
                                    "role": msg["role"],
                                    "content": original_content,
                                    "timestamp": msg["timestamp"],
                                    "session_file": filename,
                                    "msg_index": msg_global_idx,
                                    "chunk_index": chunk_idx,
                                    "total_chunks": len(chunks),
                                    "agent": AGENT_NAME,
                                },
                            }

                            # 写入 Qdrant（wait=true）
                            if not upsert_points([point], wait=True):
                                print(f"  [!] Qdrant写入失败: point_id={point_id}")
                                continue

                            # 从磁盘读取验证
                            ok, _ = verify_written_point(point_id, vector)
                            if not ok:
                                print(f"  [!] 磁盘验证失败: point_id={point_id}")
                                continue

                            total_success += 1

                    # 一条消息的所有chunk都处理完，保存checkpoint（移到这里，per-message而非per-chunk）
                    checkpoint["last_timestamp"] = msg["timestamp"]
                    checkpoint["last_msg_idx"] = msg_global_idx
                    checkpoint["last_session_file"] = filename
                    checkpoint["next_point_id"] = next_point_id
                    checkpoint["file_positions"] = file_positions
                    checkpoint["file_inodes"] = file_inodes
                    save_checkpoint(checkpoint)

                    # checkpoint保存后直接处理下一条，轮询间隔由主循环控制

                    current_msg_idx += 1

                if total_success > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] 写入 {total_success} 条")

                time.sleep(POLL_INTERVAL)

            except Exception as e:
                print(f"[错误] 主循环异常: {e}")
                time.sleep(POLL_INTERVAL)

        print()
        print("监控已停止")
        save_checkpoint(checkpoint)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        print("[锁] 已释放")


if __name__ == "__main__":
    main()
