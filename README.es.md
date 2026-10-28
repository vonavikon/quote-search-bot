# quote-search-bot

[RU](README.md) · [EN](README.en.md) · [中文](README.zh.md) · **Español** · [Français](README.fr.md)
[![VirusTotal](https://img.shields.io/badge/VirusTotal-0%2F75_clean-brightgreen)](https://www.virustotal.com/gui/file/e479ad7919419e88c38fe93f005203b00ea4e9e946b42b55c899c1184777773c)


Bot de Telegram para búsqueda semántica sobre un corpus de textos propio. Los datos salen de la exportación de un canal de Telegram a JSON: los posts se convierten en un índice vectorial y la búsqueda va por sentido, no por subcadena. La tarjeta de resultado es el texto con la firma del autor y un cuadro de un fondo de dominio público (Wikimedia Commons).

El proyecto se escribió para un canal de citas (filosofía, poesía, aforismos), pero el corpus no tiene por qué ser de citas: sirve cualquier canal de textos cortos — notas, poemas, fragmentos de libros.

## Funciones

- Búsqueda por tema («soledad»), por autor («Pasternak sobre el amor») y por frase exacta entre comillas («lástima de uno mismo»)
- Búsqueda híbrida: vectores densos de qwen3-embedding-4b, BM25 léxico como respaldo para nombres raros y frases exactas, fusión por RRF
- Etiquetado del corpus con LLM: kind (cita / poema / reflexión del autor), autor, fuente. Regla de confianza: el autor solo se asigna a partir de una firma explícita en el texto del post; adivinar por el estilo está prohibido y además se comprueba por código
- Elección del cuadro para la tarjeta sin LLM: tema de la consulta, mapa semántico «texto — cuadro», tema del texto de la cita, hash estable de post_id (la tarjeta lleva siempre el mismo cuadro)
- Botones bajo la tarjeta: otra cita, cita parecida, leer completa, Sic erat in fatis (rama aleatoria)
- Fichas sobre el autor y el cuadro desde Wikipedia, se montan con streaming
- Ingesta en vivo: el bot añadido como administrador del canal ve los posts nuevos y las ediciones, y el índice se amplía al momento
- Sin repeticiones: las citas y cuadros ya mostrados a un usuario no vuelven mientras quede algo por mostrar
- Límite diario de 100 consultas por usuario (el administrador, sin límite), registro del gasto de LLM en `/stats`

## Requisitos

- Python 3.12+
- Token del bot de [@BotFather](https://t.me/BotFather)
- Clave de API de [bothub](https://bothub.place) — pasarela de modelos compatibles con OpenAI (embeddings y LLM)
- Exportación del canal: Telegram Desktop, el canal, menú (⋮), Export chat history, formato Machine-readable JSON

## Puesta en marcha

1. Clona el repositorio e instala las dependencias:

   ```
   pip install -r requirements.txt
   ```

2. Copia `.env.example` a `.env` y rellena `TG_BOT_TOKEN`, `ADMIN_TG_ID`, `BOTHUB_API_KEY`.

3. Pon `result.json` (la exportación del canal) en la raíz del repositorio. Comprueba el parseo:

   ```
   python -X utf8 load_corpus.py
   ```

   Imprime el número de posts con texto y los tres primeros.

4. Etiqueta el corpus (una pasada de LLM, lotes de 15, caché `normalization.json`; una repetición solo incorpora los posts nuevos):

   ```
   python -X utf8 normalize.py
   ```

5. Descarga el fondo de cuadros (~900 jpg de Wikimedia Commons, pausas de cortesía entre peticiones, tarda alrededor de una hora). El paso es obligatorio: sin los archivos no se envían las tarjetas.

   ```
   python -X utf8 fetch_paintings.py
   ```

6. Arranca el bot:

   ```
   python -X utf8 bot.py
   ```

   Al arrancar se calculan los vectores de todos los posts (lotes de 32; para 600 posts son unos minutos y una pasada de embeddings) y el mapa semántico de los cuadros; después, long polling: no hacen falta webhook, dominio ni TLS.

## Actualizar el corpus

- **Ingesta en vivo.** Añade el bot como administrador del canal. Cada post nuevo y cada edición se ingieren automáticamente: un embedding y una llamada de etiquetado LLM por post.
- **Resincronización.** Comando `/resync` en el chat con el bot, luego envía un `result.json` fresco como archivo. El bot cose lo que falta — la operación es idempotente por id y texto del post.
- **En local, sin el bot.** `python -X utf8 update.py` — pon un `result.json` fresco y ejecútalo: completa el índice y el etiquetado de forma incremental con un solo comando.
- **Estadísticas.** `/stats` — número de posts y vectores, desglose por kind, gasto de LLM del mes.

## Docker

Los cuadros y la exportación inicial se preparan en el host; el estado vive en un volume:

```
python -X utf8 fetch_paintings.py
cp .env.example .env      # заполнить ключи
# result.json положить рядом с docker-compose.yml
docker compose up -d --build
```

El entrypoint pasa el etiquetado antes de arrancar el bot (idempotente; al instante si el caché está al día). Tras la indexación inicial se puede quitar el montaje de `result.json` de `docker-compose.yml`: las actualizaciones van por `/resync` directamente al volume.

## Variables de entorno

| Variable | Función |
|---|---|
| `TG_BOT_TOKEN` | token de BotFather |
| `ADMIN_TG_ID` | Telegram-ID numérico del administrador: acceso a `/stats`, `/resync`, recepción de exportaciones, avisos de posts saltados; sin límite de consultas |
| `BOTHUB_API_KEY` | clave de la pasarela de modelos |
| `BOT_DATA` | carpeta de estado (índice, etiquetado, SQLite, cachés). Por defecto, la carpeta del código; en Docker, `/data` |

## Coste

La pasada inicial son los embeddings de todos los posts y una pasada de etiquetado; en un corpus de 600 posts son decenas de céntimos. En ejecución, cada búsqueda cuesta un embedding de la consulta, las consultas largas añaden una reformulación LLM, e ingerir un post, un embedding más el etiquetado. `/stats` muestra el gasto real.

## Cómo funciona

| Archivo | Papel |
|---|---|
| `bot.py` | bot de aiogram: handlers, tarjetas, mazos de resultados |
| `router.py` | enrutado del mensaje: comandos de administrador, intro, búsqueda; reformulación de consultas largas |
| `hybrid.py` | búsqueda híbrida: denso + BM25, fusión RRF, política de respaldos |
| `index.py` | índice vectorial `corpus.npz`, costura incremental |
| `normalize.py` | etiquetado LLM de kind / author / source |
| `images.py` | elección del cuadro y caché de file_id de Telegram |
| `reference.py` | fichas del autor y del cuadro desde Wikipedia |
| `ingest.py` | ingesta en vivo y `/resync` |
| `store.py` | SQLite: usuarios, lo mostrado, límites, eventos |
| `spend.py` | registro del gasto de LLM |
| `cards.py` | montaje de la tarjeta: texto, autor, fuente, tipografía |
| `text_clean.py` | tipografía de visualización |
| `load_corpus.py` | parseo de result.json, manifiestos opcionales de uniones y OCR |
| `fetch_paintings.py` | descarga del fondo de cuadros de Wikimedia Commons |
| `evalset_draft.py` | borrador de eval-set: marcas los posts relevantes para tus consultas |
| `lab/` | medidas de calidad: comparación de modelos de embeddings, efecto del reranking |

El modelo de embeddings y el de etiquetado se eligieron midiendo recall@5 en un corpus de referencia (ver `lab/`); ambos están disponibles vía bothub y se cambian por la constante `MODEL` en `embed.py` / `normalize.py`.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Más de 160 tests, todos offline: el LLM y la búsqueda van con mocks, no hace falta red externa.

## Licencia

MIT. El fondo de cuadros es de dominio público (Wikimedia Commons); la lista con la atribución de autores está en `paintings/paintings.json`.
