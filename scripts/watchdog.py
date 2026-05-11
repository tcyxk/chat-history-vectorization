#!/usr/bin/env python3
"""
服务看门狗 + 任务堆积检测
功能：
  1. 启动检测：等待 Qdrant/llama-embedding 就绪后再放行
  2. 服务保活：定期检测云端 Qdrant（无法远程重启，仅报警）
  3. 任务堆积检测：检测 checkpoint 是否有任务卡住
  4. 自动恢复：服务异常时自动重启相关脚本（realtime.py）

用法：
  python3 watchdog.py              # 前台运行（调试）
  nohup python3 watchdog.py &     # 后台运行
"""
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

# ============ 单例锁 ============
LOCK_FILE = os.path.expanduser("~/.hermes/watchdog.lock")

def acquire_lock():
    """获取进程锁，确保只有一个watchdog实例运行"""
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd
    except BlockingIOError:
        print("[错误] watchdog.py已在运行，请先停止旧进程")
        print("  执行: pkill -f 'python3 watchdog.py'")
        lock_fd.close()
        sys.exit(1)

from config import (
    AGENT_NAME,
    COLLECTION_NAME,
    HERMES_HOME,
    LLAMA_EMBEDDING_BIN,
    MIGRATE_CHECKPOINT,
    POLL_INTERVAL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    REALTIME_CHECKPOINT,
    WATCHDOG_INTERVAL,
)

# ============ 路径常量 ============
SCRIPTS_DIR = os.path.join(HERMES_HOME, "scripts")


# ============ 全局状态 ============

g_running = True
g_log_file = None


def log(msg: str):
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if g_log_file:
        with open(g_log_file, "a") as f:
            f.write(line + "\n")


# ============ 网络检测 ============

def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# ============ 服务健康检测 ============

def check_qdrant_alive() -> tuple[bool, str]:
    """检测 Qdrant 是否存活"""
    # 1. 端口检测
    if not is_port_open(QDRANT_HOST, QDRANT_PORT):
        return False, "端口不可访问"

    # 2. API 检测
    try:
        result = subprocess.run(
            ["curl", "-s", f"{QDRANT_URL}/collections"],
            capture_output=True, timeout=5
        )
        data = json.loads(result.stdout)
        if data.get("result") is not None:
            return True, "正常"
    except Exception as e:
        return False, f"API错误: {e}"

    return False, "未知错误"


def check_llama_embedding_alive() -> tuple[bool, str]:
    """检测 llama-embedding 是否可用"""
    if not os.path.exists(LLAMA_EMBEDDING_BIN):
        return False, "二进制文件不存在"

    if not os.access(LLAMA_EMBEDDING_BIN, os.X_OK):
        return False, "无执行权限"

    # 简单调用测试
    try:
        result = subprocess.run(
            [LLAMA_EMBEDDING_BIN, "--help"],
            capture_output=True, timeout=5,
            env={**os.environ, "LD_LIBRARY_PATH": os.path.join(HERMES_HOME, "bin")}
        )
        return True, "正常"
    except Exception as e:
        return False, f"执行失败: {e}"


def check_qdrant_collection_status() -> tuple[bool, str]:
    """检测 Collection 是否存在且健康"""
    try:
        result = subprocess.run(
            ["curl", "-s", f"{QDRANT_URL}/collections/{COLLECTION_NAME}"],
            capture_output=True, timeout=5
        )
        data = json.loads(result.stdout)
        result_obj = data.get("result")
        if result_obj is None:
            return False, f"Collection不存在: {COLLECTION_NAME}"
        status = result_obj.get("status", "unknown")
        points = result_obj.get("points_count", 0)
        return True, f"正常 (points={points}, status={status})"
    except Exception as e:
        return False, f"API错误: {e}"


def check_service_health() -> dict:
    """综合健康检测"""
    qdrant_ok, qdrant_msg = check_qdrant_alive()
    llama_ok, llama_msg = check_llama_embedding_alive()
    collection_ok, collection_msg = check_qdrant_collection_status() if qdrant_ok else (False, "Qdrant未就绪")

    return {
        "qdrant": (qdrant_ok, qdrant_msg),
        "llama_embedding": (llama_ok, llama_msg),
        "collection": (collection_ok, collection_msg),
        "all_ok": qdrant_ok and llama_ok and collection_ok,
    }


# ============ 任务堆积检测 ============

def get_checkpoint_age(path: str) -> tuple[float | None, str]:
    """获取 checkpoint 文件的年龄（秒），返回 (秒数, 修改时间字符串)"""
    if not os.path.exists(path):
        return None, "文件不存在"

    try:
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        return age, ts
    except Exception as e:
        return None, str(e)


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def check_task_backlog() -> dict:
    """
    检测是否有任务堆积/卡住
    """
    results = {}

    # realtime checkpoint
    rt_age, rt_ts = get_checkpoint_age(REALTIME_CHECKPOINT)
    rt_data = load_checkpoint(REALTIME_CHECKPOINT)
    rt_last_idx = rt_data.get("last_msg_idx", 0)

    results["realtime"] = {
        "exists": rt_age is not None,
        "age_seconds": rt_age,
        "last_modified": rt_ts,
        "last_msg_idx": rt_last_idx,
        "stale": rt_age is not None and rt_age > 300 and rt_last_idx == 0,  # 5分钟无数据
    }

    # migrate checkpoint
    mg_age, mg_ts = get_checkpoint_age(MIGRATE_CHECKPOINT)
    mg_data = load_checkpoint(MIGRATE_CHECKPOINT)
    mg_last_idx = mg_data.get("last_msg_index", 0)

    results["migrate"] = {
        "exists": mg_age is not None,
        "age_seconds": mg_age,
        "last_modified": mg_ts,
        "last_msg_idx": mg_last_idx,
    }

    return results


def is_realtime_running() -> bool:
    """检测 realtime 脚本是否在运行"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "realtime\\.py"],
            capture_output=True, text=True
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


# ============ 自动恢复 ============

def auto_fix_issues(health: dict, backlog: dict) -> list[str]:
    """自动处理异常，返回处理描述列表"""
    actions = []

    # 1. Qdrant 异常（云端 Qdrant 无法远程重启，仅报警）
    if not health["qdrant"][0]:
        actions.append(f"Qdrant异常: {health['qdrant'][1]}，请检查云端服务")

    # 2. realtime 脚本卡住（5分钟无进展）
    rt = backlog.get("realtime", {})
    if rt.get("stale") and not is_realtime_running():
        log("[Realtime] 检测到任务堆积且进程未运行，尝试重启...")
        realtime_log = os.path.join(HERMES_HOME, "logs", "realtime.log")
        try:
            subprocess.Popen(
                ["python3", os.path.join(SCRIPTS_DIR, "realtime.py")],
                stdout=open(realtime_log, "a"),
                stderr=subprocess.STDOUT,
                cwd=SCRIPTS_DIR,
            )
            actions.append("realtime.py已重启")
            log("[Realtime] 进程已启动")
        except Exception as e:
            actions.append(f"realtime.py重启失败: {e}")
            log(f"[Realtime] 重启失败: {e}")

    return actions


# ============ 主循环 ============

def main():
    global g_running, g_log_file

    # 获取单例锁
    lock_fd = acquire_lock()

    print("=" * 60)
    print(f"服务看门狗 [{AGENT_NAME}]")
    print("=" * 60)

    # 日志文件
    log_dir = os.path.join(HERMES_HOME, "logs")
    os.makedirs(log_dir, exist_ok=True)
    g_log_file = os.path.join(log_dir, f"watchdog_{AGENT_NAME}.log")

    log(f"看门狗启动")
    log(f"Qdrant: {QDRANT_URL}")
    log(f"Collection: {COLLECTION_NAME}")

    # 信号处理
    def signal_handler(signum, frame):
        global g_running
        log("收到停止信号")
        g_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # ============ 启动阶段：等待服务就绪 ============
        log("[启动] 等待服务就绪...")

        wait_ok = False
        for attempt in range(1, 31):
            health = check_service_health()
            if health["all_ok"]:
                log(f"[启动] 服务就绪 (尝试{attempt}次)")
                wait_ok = True
                break
            if attempt <= 10:
                log(f"[启动] 等待... ({attempt}/30) - {health['qdrant'][1]}")
            time.sleep(2)

        if not wait_ok:
            log("[启动] 服务未就绪，继续监控（将持续重试）")

        consecutive_failures = 0
        last_health_ok = True

        # ============ 主循环 ============
        while g_running:
            try:
                # 健康检测
                health = check_service_health()

                if health["all_ok"]:
                    consecutive_failures = 0
                    last_health_ok = True
                else:
                    consecutive_failures += 1
                    if last_health_ok:
                        log(f"[警告] Qdrant/llama_embedding/Collection 出现异常")
                    last_health_ok = False
                    issues = []
                    if not health["qdrant"][0]:
                        issues.append(f"Qdrant:{health['qdrant'][1]}")
                    if not health["llama_embedding"][0]:
                        issues.append(f"llama:{health['llama_embedding'][1]}")
                    if not health["collection"][0]:
                        issues.append(f"Collection:{health['collection'][1]}")
                    log(f"  #{consecutive_failures} {', '.join(issues)}")

                    # 自动恢复
                    if consecutive_failures >= 2:
                        backlog = check_task_backlog()
                        actions = auto_fix_issues(health, backlog)
                        for a in actions:
                            log(f"[自动恢复] {a}")

                # 任务堆积检测
                backlog = check_task_backlog()
                rt = backlog.get("realtime", {})
                if rt.get("exists") and rt.get("age_seconds") and rt["age_seconds"] > 300:
                    log(f"[任务] realtime checkpoint 老旧 ({rt['age_seconds']:.0f}s未更新), msg_idx={rt.get('last_msg_idx')}")

                mg = backlog.get("migrate", {})
                if mg.get("exists") and mg.get("age_seconds"):
                    log(f"[任务] migrate checkpoint age={mg['age_seconds']:.0f}s, msg_idx={mg.get('last_msg_idx')}")

            except Exception as e:
                log(f"[错误] 主循环异常: {e}")

            # 休眠
            for _ in range(WATCHDOG_INTERVAL // 5):
                if not g_running:
                    break
                time.sleep(5)

        log("看门狗已停止")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        print("[锁] 已释放")


if __name__ == "__main__":
    main()
