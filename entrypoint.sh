#!/bin/sh
# Corpus normalization is idempotent: it only picks up unlabeled posts,
# and with an up-to-date cache it exits without any LLM calls. A failure
# (network, key) must not block the bot's startup: the next start picks
# the labeling up.
python -X utf8 normalize.py || echo "!! normalize не отработал, продолжаю"
exec python -X utf8 bot.py
