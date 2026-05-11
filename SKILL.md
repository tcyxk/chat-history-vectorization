---
name: chat-history-vectorization
description: >
  聊天记录向量化系统：把Hermes历史聊天记录转成向量存Qdrant，
  实现语义搜索和RAG上下文注入。支持实时增量写入、历史批量迁移、
  语义检索_hook三种模式，全流程单进程保证数据一致性。
version: 1.0.0
author: xiaobao
tags: [qdrant, vector, rag, chat-history, hermes-hook]
---

# 聊天记录向量化系统

## 一、系统架构

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  pre_gateway_dispatch hook (rag_context/handler.py)  │
│  1. get_embedding(用户消息) → 1024维向量            │
│  2. search_points(向量, top_k=5) → Qdrant检索      │
│  3. 格式化相关对话 → rewrite用户消息                  │
└─────────────────────────────────────────────────────┘
    │
    ▼  rewrite后的消息（含相关对话上下文）
Gateway ──→ Agent（小小宝）──→ 回复
```

**三层架构：**
1. **存储层**：Qdrant向量库（云端 60.205.223.136:6333）
2. **写入层**：migrate.py（历史迁移）+ realtime.py（实时增量）
3. **检索层**：RAG hook（pre_gateway_dispatch，消息进入Gateway前拦截）

**关键设计原则：**
- 单进程铁律：全程只能有一条数据在执行，任何一步出现两条就说明逻辑有问题
- 断电最多丢1秒数据：Qdrant WAL + flush_interval_sec=1
- 多账号隔离：每个agent独立Collection（chat_xiaobao）、独立session目录、独立checkpoint
- 静默运行：正常情况下RAG hook不打日志，不打扰用户

---

## 二、核心文件

| 文件 | 作用 | 入口命令 |
|------|------|----------|
| `scripts/config.py` | 全局配置（Qdrant地址、路径、参数） | 被所有脚本import |
| `scripts/embedding.py` | llama-embedding调用封装 | 被所有脚本import |
| `scripts/qdrant_client.py` | Qdrant HTTP API封装（upsert/verify/scroll/search） | 被所有脚本import |
| `scripts/session_parser.py` | 解析Hermes session文件，提取消息 | migrate.py调用 |
| `scripts/text_splitter.py` | 长文本按句子切片（chunk_size=2000字符） | migrate.py调用 |
| `scripts/migrate.py` | 批量迁移历史聊天记录到Qdrant | `python migrate.py` |
| `scripts/realtime.py` | 实时监控session文件增量写入Qdrant | `python realtime.py` |
| `scripts/watchdog.py` | 健康检查（Qdrant/llama-embedding进程监控） | 定时cron |
| `hooks/rag_context/handler.py` | RAG hook，消息进入前语义检索注入上下文 | Gateway自动加载 |

---

## 三、快速启动

### 1. 启动历史迁移（首次或断档后）
```bash
cd ~/.hermes/scripts
python migrate.py
# 输出示例：Processing: sessions/20260510_151522_7f3a2e.msgpack [59/847]
# 失败重跑：从checkpoint恢复，不重复写入
```

### 2. 启动实时增量写入
```bash
cd ~/.hermes/scripts
python realtime.py
# 输出示例：[06:02:15] 检测到新消息，ID=3988
# 失败重跑：从Qdrant最后一条恢复，不漏记
```

### 3. 启动健康检查（定时cron）
```bash
# 每60秒检查一次Qdrant和llama-embedding是否存活
python watchdog.py
```

### 4. RAG Hook（自动生效）
- Gateway启动时自动加载 `hooks/rag_context/HOOK.yaml`
- **无需手动启动**，pre_gateway_dispatch事件自动触发
- 查看加载状态：`tail -f ~/.hermes/logs/agent.log | grep hook`

---

## 四、RAG Hook 工作流程

```
用户发消息 "MBTI测试怎么做"
           │
           ▼
pre_gateway_dispatch 事件触发
           │
           ▼
handler.py handle():
  1. get_embedding("MBTI测试怎么做") → 1024维向量
  2. search_points(向量, top_k=5) → Qdrant返回5条最相关记录
  3. 格式化:
      [相关对话回忆]
      ───────────────────────────────────────
      [2026-05-07T19:10][assistant] 谢谢兴哥认可！
      [2026-05-07T19:12][assistant] 好的兴哥，我先了解...
      ───────────────────────────────────────
      你当前的问题是：MBTI测试怎么做
  4. return {"action": "rewrite", "text": "..."}
           │
           ▼
Gateway 拿rewrite后的消息发给Agent
           │
           ▼
小小宝"想起"了相关对话上下文，回复更准确
```

**Payload结构（Qdrant每条记录）：**
```json
{
  "id": 1234000,
  "vector": [0.123, -0.456, ...],  // 1024维
  "payload": {
    "timestamp": "2026-05-07T19:10:23",
    "role": "user",           // user / assistant
    "content": "在吗？",
    "session_file": "20260507_191023_xxx.msgpack",
    "msg_index": 0,
    "chunk_index": 0,
    "text_length": 6
  }
}
```

**ID设计：`point_id = msg_index * 1000 + chunk_index`**
- msg_index：消息在文件内的序号
- chunk_index：长消息分片序号
- 优势：重启后从Qdrant最大ID继续，不重复不漏记

---

## 五、配置说明（config.py）

```python
# Qdrant（云端）
QDRANT_HOST = "60.205.223.136"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME = "chat_xiaobao"   # Collection名称

# 向量参数
VECTOR_DIM = 1024                  # BGE-M3向量维度
CHUNK_SIZE = 2000                  # 每chunk最大字符数（句子边界）
MAX_TEXT_LEN = 8000                # 转换最大长度（超过截断）

# llama-embedding
LLAMA_EMBEDDING_BIN = "~/.hermes/bin/llama-embedding"
BGE_M3_MODEL = "/storage/emulated/0/Download/OnePlus Share/bge-m3-q8_0.gguf"
```

---

## 六、运维命令

### 查看Qdrant数据量
```bash
curl -s http://60.205.223.136:6333/collections/chat_xiaobao | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['points_count'])"
```

### 查看最新写入的记录
```bash
curl -s -X POST http://60.205.223.136:6333/collections/chat_xiaobao/points/scroll -H "Content-Type: application/json" -d '{"limit":3,"with_payload":true}' | python3 -c "import sys,json; d=json.load(sys.stdin); [print(p['id'], p['payload'].get('timestamp',''), p['payload'].get('content','')[:50]) for p in d['result']['points']]"
```

### 查看迁移进度（checkpoint）
```bash
cat ~/.hermes/migrate_xiaobao_checkpoint.json
# 输出：{"last_session": "sessions/20260511_234501_abc123.msgpack", "last_msg_index": 847, "next_point_id": 847000}
```

### 清空重来（慎用）
```bash
# 1. 停掉所有脚本
pkill -f "python.*migrate.py" && pkill -f "python.*realtime.py"
# 2. 清Qdrant collection（通过临时collection中转，彻底清storage）
# 3. 清checkpoint和lock
rm ~/.hermes/*_xiaobao_checkpoint.json ~/.hermes/*.lock
# 4. 重启迁移
python migrate.py
```

### 重启RAG Hook（修改hook代码后）
```bash
# hook在Gateway启动时加载，改动后需重启Gateway
hermes gateway run --replace
```

---

## 七、常见问题

### Q：为什么有时候小小宝"想不起"相关对话？
A：RAG只取top_k=5条最相似记录，如果用户消息与历史记录语义关联弱，检索结果就差。可以提高TOP_K或优化content过滤逻辑。

### Q：realtime.py漏记了新消息怎么办？
A：从checkpoint恢复，realtime.py启动时会自动从Qdrant最后一条开始续传。

### Q：断电了会丢多少数据？
A：最多丢1秒。Qdrant WAL机制 + wait=true + flush_interval_sec=1，保证断电最多丢1秒。

### Q：为什么不用本地Qdrant？
A：Termux文件系统不执行真正的fsync，Qdrant WAL机制失效，断电数据会损坏。已改用云端Qdrant（阿里云服务器）。

### Q：清空Qdrant collection后数据"又回来了"？
A：这是Qdrant async删除导致的后台storage残留。`delete_collection` API返回成功，但storage目录异步清理未完成，此时重建同名collection，旧数据还在。
**正确清空流程（三步走）：**
```bash
# 1. 删除原collection
curl -X DELETE http://60.205.223.136:6333/collections/chat_xiaobao
# 2. 等5秒让storage GC完成
sleep 5
# 3. 创建临时collection，scroll确认为0，再删临时collection
curl -X PUT http://60.205.223.136:6333/collections/temp_verify
curl -X POST http://60.205.223.136:6333/collections/temp_verify/points/scroll -d '{"limit":10}'
curl -X DELETE http://60.205.223.136:6333/collections/temp_verify
```
核心Bug：Qdrant删除collection是异步的，重建同名collection时storage尚未清空，导致旧数据"复活"。

### Q：RAG hook里`get_embedding()`怎么调用才正确？
A：handler.py通过 `sys.path.insert` 引入scripts目录的embedding模块，直接调用 `get_embedding(text)`。**关键**：llama-embedding要求纯base64 ASCII字符串输入（不加JSON/其他包装），timeout=60秒，输出格式 `embedding 0: <1024个浮点数>`。

---

## 八、开发日志
完整Bug修复记录见：`/storage/emulated/0/我的备份/chat_vectorization_DEVLOG.md`
