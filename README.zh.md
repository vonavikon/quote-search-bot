# quote-search-bot

[RU](README.md) · [EN](README.en.md) · **中文** · [Español](README.es.md) · [Français](README.fr.md)

Telegram 机器人，对自己的文本语料做语义搜索。数据来源是 Telegram 频道的 JSON 导出：帖子被转成向量索引，搜索按语义进行，而不是按子串匹配。结果卡片是带作者署名的文本，配上一张来自公有领域画作池（Wikimedia Commons）的画。

项目为语录频道（哲学、诗歌、格言）而写，但语料不必是语录：任何短文本频道都合适——笔记、诗、书摘。

## 功能

- 按主题（"孤独"）、作者（"帕斯捷尔纳克谈爱情"）和引号内的精确短语（"自怜"）搜索
- 混合搜索：qwen3-embedding-4b 稠密向量，覆盖罕见姓名和精确短语的 BM25 词法兜底，通过 RRF 融合
- LLM 语料标注：kind（语录 / 诗 / 作者随想）、作者、来源。信任规则：只有帖子文本里明确署名才填作者；禁止按文风猜测，代码里另有校验
- 配图挑选不经过 LLM：查询主题、"文本—画作"语义映射、语录文本主题、post_id 稳定哈希（卡片始终配同一幅画）
- 卡片下方的按钮：再来一条、相似语录、阅读全文、Sic erat in fatis（随机分支）
- 来自维基百科的作者和画作简介，以流式方式拼出
- 实时入库：把机器人加为频道管理员后，新帖和编辑都会被看到，索引即时补充
- 不重复：已给用户展示过的语录和画作不再出现，直到没有可展示的为止
- 每用户每日 100 次请求限额（管理员不限），`/stats` 里统计 LLM 开销

## 环境要求

- Python 3.12+
- 从 [@BotFather](https://t.me/BotFather) 获取的机器人 token
- [bothub](https://bothub.place) 的 API key——OpenAI 兼容模型的网关（embedding 和 LLM）
- 频道导出：Telegram Desktop，打开频道，菜单 (⋮)，Export chat history，格式选 Machine-readable JSON

## 快速开始

1. 克隆仓库并安装依赖：

   ```
   pip install -r requirements.txt
   ```

2. 把 `.env.example` 复制为 `.env`，填入 `TG_BOT_TOKEN`、`ADMIN_TG_ID`、`BOTHUB_API_KEY`。

3. 把 `result.json`（频道导出）放到仓库根目录。检查解析：

   ```
   python -X utf8 load_corpus.py
   ```

   会打印有文本的帖子数量和前三条。

4. 标注语料（一遍 LLM，每批 15 条，缓存 `normalization.json`；重跑只补新帖）：

   ```
   python -X utf8 normalize.py
   ```

5. 下载画作池（约 900 张 jpg，来自 Wikimedia Commons，请求之间有礼貌性停顿，耗时约一小时）。此步必需：没有文件就无法发送卡片。

   ```
   python -X utf8 fetch_paintings.py
   ```

6. 启动机器人：

   ```
   python -X utf8 bot.py
   ```

   启动时计算所有帖子的向量（每批 32 条；600 帖约几分钟，一遍 embedding 调用），并构建画作语义映射，之后进入长轮询：不需要 webhook、域名和 TLS。

## 更新语料

- **实时入库。** 把机器人加为频道管理员。每条新帖和每次编辑都会自动入库：每帖一次 embedding 和一次 LLM 标注调用。
- **补同步。** 在与机器人的对话里发 `/resync`，然后把新的 `result.json` 作为文件发来。机器人把缺的补进索引——操作按帖子 id 和文本幂等。
- **本地、不开机器人。** `python -X utf8 update.py`——放好新的 `result.json` 后运行：一条命令增量补齐索引和标注。
- **统计。** `/stats`——帖子数和向量数、按 kind 的分布、当月 LLM 开销。

## Docker

画作和首次导出在宿主机上准备，状态保存在 volume 里：

```
python -X utf8 fetch_paintings.py
cp .env.example .env      # заполнить ключи
# result.json положить рядом с docker-compose.yml
docker compose up -d --build
```

启动机器人前 entrypoint 会跑一遍标注（幂等，缓存有效时瞬间完成）。首次索引完成后，可以从 `docker-compose.yml` 里去掉 `result.json` 的挂载：更新通过 `/resync` 直接写入 volume。

## 环境变量

| 变量 | 用途 |
|---|---|
| `TG_BOT_TOKEN` | BotFather 发的 token |
| `ADMIN_TG_ID` | 管理员的数字 Telegram ID：可用 `/stats`、`/resync`，接收导出，漏帖告警；不限请求数 |
| `BOTHUB_API_KEY` | 模型网关的 key |
| `BOT_DATA` | 状态目录（索引、标注、SQLite、缓存）。默认是代码目录；Docker 里是 `/data` |

## 成本

首次运行是为所有帖子生成 embedding，加一遍标注；600 帖的语料大约几十美分。运行期每次搜索花费一次查询 embedding，长查询再加一次 LLM 改写，帖子入库是一次 embedding 加一次标注。`/stats` 显示实际开销。

## 工作原理

| 文件 | 职责 |
|---|---|
| `bot.py` | aiogram 机器人：处理函数、卡片、结果牌组 |
| `router.py` | 消息路由：管理员命令、开场白、搜索；长查询改写 |
| `hybrid.py` | 混合搜索：dense + BM25，RRF 融合，兜底策略 |
| `index.py` | 向量索引 `corpus.npz`，增量补入 |
| `normalize.py` | kind / author / source 的 LLM 标注 |
| `images.py` | 画作挑选和 Telegram file_id 缓存 |
| `reference.py` | 来自维基百科的作者和画作简介 |
| `ingest.py` | 实时入库和 `/resync` |
| `store.py` | SQLite：用户、已展示内容、限额、事件 |
| `spend.py` | LLM 开销统计 |
| `cards.py` | 卡片组装：文本、作者、来源、排版 |
| `text_clean.py` | 展示排版 |
| `load_corpus.py` | 解析 result.json，可选的合并与 OCR 清单 |
| `fetch_paintings.py` | 从 Wikimedia Commons 下载画作池 |
| `evalset_draft.py` | 评测集草稿：为自己的查询标注相关帖子 |
| `lab/` | 质量测量：embedding 模型对比、重排效果 |

embedding 模型和标注模型是在基准语料上测 recall@5 选出的（见 `lab/`）；两者都可通过 bothub 访问，改 `embed.py` / `normalize.py` 里的 `MODEL` 常量即可更换。

## 测试

```
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

160+ 个测试，全部离线：LLM 和搜索都跑在 mock 下，不需要外部网络。

## 许可证

MIT。画作池属于公有领域（Wikimedia Commons），带作者署名的清单在 `paintings/paintings.json`。
