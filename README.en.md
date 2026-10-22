# quote-search-bot

[RU](README.md) · **EN** · [中文](README.zh.md) · [Español](README.es.md) · [Français](README.fr.md)

A Telegram bot for semantic search over your own text corpus. The data source is a Telegram channel export in JSON: posts are turned into a vector index, and search works by meaning rather than by substring. A result card is the text with the author's signature and a painting from a public-domain pool (Wikimedia Commons).

The project was written for a quotes channel (philosophy, poetry, aphorisms), but the corpus doesn't have to be quotes: any channel of short texts will do — notes, poems, book excerpts.

## Features

- Search by topic ("loneliness"), by author ("pasternak about love"), and by exact phrase in quotes ("self-pity")
- Hybrid search: dense vectors from qwen3-embedding-4b, a lexical BM25 fallback for rare names and exact phrases, fusion via RRF
- LLM labeling of the corpus: kind (quote / poem / author's reflection), author, source. Trust rule: an author is only assigned from an explicit signature in the post text; guessing by style is forbidden and is additionally checked in code
- Painting selection for the card without an LLM: query topic, a text-to-painting semantic map, the quote's text topic, a stable hash of post_id (a card always gets the same painting)
- Buttons under the card: another quote, similar quote, read in full, Sic erat in fatis (a random branch)
- Reference notes about the author and the painting from Wikipedia, assembled with streaming
- Live ingest: a bot added as channel admin sees new posts and edits, and the index grows on the fly
- No repeats: quotes and paintings already shown to a user don't come up again while there is still something left to show
- Daily limit of 100 requests per user (no limit for the admin), LLM spend tracking in `/stats`

## Requirements

- Python 3.12+
- A bot token from [@BotFather](https://t.me/BotFather)
- An [bothub](https://bothub.place) API key — a gateway for OpenAI-compatible models (embeddings and LLM)
- A channel export: Telegram Desktop, the channel, menu (⋮), Export chat history, Machine-readable JSON format

## Getting started

1. Clone the repository and install the dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in `TG_BOT_TOKEN`, `ADMIN_TG_ID`, `BOTHUB_API_KEY`.

3. Put `result.json` (the channel export) in the repository root. Check the parsing:

   ```
   python -X utf8 load_corpus.py
   ```

   It prints the number of posts with text and the first three.

4. Label the corpus (one LLM pass, batches of 15, the `normalization.json` cache; a re-run only picks up new posts):

   ```
   python -X utf8 normalize.py
   ```

5. Download the painting pool (~900 jpgs from Wikimedia Commons, polite pauses between requests, takes about an hour). This step is required: cards can't be sent without the files.

   ```
   python -X utf8 fetch_paintings.py
   ```

6. Start the bot:

   ```
   python -X utf8 bot.py
   ```

   On startup the bot embeds all posts (batches of 32; for 600 posts that's a few minutes and a single embedding pass) and builds the semantic map of paintings, then switches to long polling: no webhook, domain, or TLS needed.

## Updating the corpus

- **Live ingest.** Add the bot as a channel admin. Every new post and every edit is ingested automatically: one embedding and one labeling LLM call per post.
- **Catch-up sync.** Send `/resync` in a chat with the bot, then send a fresh `result.json` as a file. The bot stitches in whatever is missing — the operation is idempotent by post id and text.
- **Locally, without the bot.** `python -X utf8 update.py` — put a fresh `result.json` in place and run it: incrementally stitches the index and the labels in one command.
- **Stats.** `/stats` — the number of posts and vectors, a breakdown by kind, LLM spend for the month.

## Docker

Paintings and the initial export are prepared on the host; state lives in a volume:

```
python -X utf8 fetch_paintings.py
cp .env.example .env      # заполнить ключи
# result.json положить рядом с docker-compose.yml
docker compose up -d --build
```

The entrypoint runs labeling before starting the bot (idempotent; instant when the cache is up to date). After the initial indexing, the `result.json` mount can be removed from `docker-compose.yml`: updates go through `/resync` straight into the volume.

## Environment variables

| Variable | Purpose |
|---|---|
| `TG_BOT_TOKEN` | token from BotFather |
| `ADMIN_TG_ID` | the admin's numeric Telegram ID: grants `/stats`, `/resync`, export intake, alerts about skipped posts; no request limit |
| `BOTHUB_API_KEY` | model gateway API key |
| `BOT_DATA` | state directory (index, labels, SQLite, caches). Defaults to the code directory; `/data` in Docker |

## Cost

The initial run is embeddings for all posts plus one labeling pass; on a 600-post corpus that's tens of cents. At runtime, each search costs one query embedding, long queries add one more LLM reformulation, and ingesting a post costs an embedding plus labeling. `/stats` shows the actual spend.

## How it works

| File | Role |
|---|---|
| `bot.py` | aiogram bot: handlers, cards, result decks |
| `router.py` | message routing: admin commands, intro, search; reformulation of long queries |
| `hybrid.py` | hybrid search: dense + BM25, RRF fusion, fallback policy |
| `index.py` | vector index `corpus.npz`, incremental stitching |
| `normalize.py` | LLM labeling of kind / author / source |
| `images.py` | painting selection and the Telegram file_id cache |
| `reference.py` | reference notes about the author and the painting from Wikipedia |
| `ingest.py` | live ingest and `/resync` |
| `store.py` | SQLite: users, shown items, limits, events |
| `spend.py` | LLM spend tracking |
| `cards.py` | card assembly: text, author, source, typography |
| `text_clean.py` | display typography |
| `load_corpus.py` | parses result.json, optional merge and OCR manifests |
| `fetch_paintings.py` | downloads the painting pool from Wikimedia Commons |
| `evalset_draft.py` | eval-set draft: you label the posts relevant to your own queries |
| `lab/` | quality measurements: embedding model comparison, reranking effect |

The embedding model and the labeling model were chosen by measuring recall@5 on a reference corpus (see `lab/`); both are available through bothub and can be swapped via the `MODEL` constant in `embed.py` / `normalize.py`.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

160+ tests, all offline: the LLM and search run under mocks, no external network needed.

## License

MIT. The painting pool is public domain (Wikimedia Commons); the list with author attributions is in `paintings/paintings.json`.
