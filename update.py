"""Обновление пула после свежего экспорта result.json — одна команда.

Делает: index.build() инкрементом (дошьёт только новые/изменённые посты,
старые векторы из кэша) -> normalize.run() инкрементом (метаданные только
для неразмеченных) -> сводка.

Не запускать параллельно с normalize.py: оба пишут normalization.json.
"""
import json

from index import build
from load_corpus import load_posts
import normalize


def main():
    before = len(load_posts())
    print(f"== корпус: {before} постов в свежем экспорте")

    print("\n== 1/2 эмбеддинги")
    V, posts = build()

    print("\n== 2/2 нормализация")
    cache, stats, cost = normalize.run()

    print("\n== сводка")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"потрачено на нормализацию: ${cost:.5f}")
    print(f"векторов: {V.shape[0]}, размечено: {stats['n']}")


if __name__ == "__main__":
    main()
