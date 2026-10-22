# quote-search-bot

[RU](README.md) · [EN](README.en.md) · [中文](README.zh.md) · [Español](README.es.md) · **Français**

Bot Telegram de recherche sémantique sur un corpus de textes à soi. Les données viennent d'un export de canal Telegram en JSON : les posts deviennent un index vectoriel, la recherche se fait par le sens et non par sous-chaîne. La carte de résultat est le texte avec la signature de l'auteur et un tableau issu d'une réserve du domaine public (Wikimedia Commons).

Le projet a été écrit pour un canal de citations (philosophie, poésie, aphorismes), mais le corpus n'a pas besoin d'être fait de citations : n'importe quel canal de textes courts convient — notes, poèmes, extraits de livres.

## Fonctionnalités

- Recherche par thème (« solitude »), par auteur (« Pasternak sur l'amour ») et par phrase exacte entre guillemets (« pitié de soi »)
- Recherche hybride : vecteurs denses de qwen3-embedding-4b, repli lexical BM25 pour les noms rares et les phrases exactes, fusion par RRF
- Annotation du corpus par LLM : kind (citation / poème / réflexion de l'auteur), auteur, source. Règle de confiance : l'auteur n'est renseigné qu'à partir d'une signature explicite dans le texte du post ; deviner au style est interdit et est en outre vérifié par le code
- Choix du tableau pour la carte sans LLM : thème de la requête, carte sémantique « texte — tableau », thème du texte de la citation, hachage stable de post_id (une carte a toujours le même tableau)
- Boutons sous la carte : une autre citation, citation proche, lire en entier, Sic erat in fatis (branche aléatoire)
- Fiches sur l'auteur et le tableau depuis Wikipédia, assemblées en streaming
- Ingestion en direct : le bot nommé administrateur du canal voit les nouveaux posts et les éditions, l'index se complète à la volée
- Sans répétitions : les citations et tableaux déjà montrés à un utilisateur ne reviennent pas tant qu'il reste quelque chose à montrer
- Limite quotidienne de 100 requêtes par utilisateur (pas de limite pour l'administrateur), suivi des dépenses LLM dans `/stats`

## Prérequis

- Python 3.12+
- Un token de bot de [@BotFather](https://t.me/BotFather)
- Une clé API [bothub](https://bothub.place) — passerelle de modèles compatibles OpenAI (embeddings et LLM)
- Un export du canal : Telegram Desktop, le canal, menu (⋮), Export chat history, format Machine-readable JSON

## Démarrage

1. Clonez le dépôt et installez les dépendances :

   ```
   pip install -r requirements.txt
   ```

2. Copiez `.env.example` en `.env` et remplissez `TG_BOT_TOKEN`, `ADMIN_TG_ID`, `BOTHUB_API_KEY`.

3. Placez `result.json` (l'export du canal) à la racine du dépôt. Vérifiez le parsing :

   ```
   python -X utf8 load_corpus.py
   ```

   Il affiche le nombre de posts avec du texte et les trois premiers.

4. Annotez le corpus (une passe de LLM, lots de 15, cache `normalization.json` ; une nouvelle exécution ne récupère que les posts inédits) :

   ```
   python -X utf8 normalize.py
   ```

5. Téléchargez la réserve de tableaux (~900 jpg depuis Wikimedia Commons, pauses courtoises entre les requêtes, environ une heure). Étape obligatoire : sans les fichiers, les cartes ne partent pas.

   ```
   python -X utf8 fetch_paintings.py
   ```

6. Lancez le bot :

   ```
   python -X utf8 bot.py
   ```

   Au démarrage, les vecteurs de tous les posts sont calculés (lots de 32 ; pour 600 posts, quelques minutes et une seule passe d'embedding) ainsi que la carte sémantique des tableaux, puis place au long polling : pas besoin de webhook, de domaine ni de TLS.

## Mise à jour du corpus

- **Ingestion en direct.** Ajoutez le bot comme administrateur du canal. Chaque nouveau post et chaque édition est ingéré automatiquement : un embedding et un appel LLM d'annotation par post.
- **Resynchronisation.** Commande `/resync` dans le chat avec le bot, puis envoyez un `result.json` frais en fichier. Le bot recoud ce qui manque — l'opération est idempotente par id et par texte du post.
- **En local, sans le bot.** `python -X utf8 update.py` — posez un `result.json` frais et lancez : recoud incrémentalement l'index et l'annotation en une seule commande.
- **Statistiques.** `/stats` — nombre de posts et de vecteurs, répartition par kind, dépenses LLM du mois.

## Docker

Les tableaux et l'export initial se préparent sur l'hôte ; l'état vit dans un volume :

```
python -X utf8 fetch_paintings.py
cp .env.example .env      # заполнить ключи
# result.json положить рядом с docker-compose.yml
docker compose up -d --build
```

L'entrypoint déroule l'annotation avant de lancer le bot (idempotent ; instantané quand le cache est à jour). Après l'indexation initiale, le montage de `result.json` peut être retiré de `docker-compose.yml` : les mises à jour passent par `/resync` directement dans le volume.

## Variables d'environnement

| Variable | Rôle |
|---|---|
| `TG_BOT_TOKEN` | token de BotFather |
| `ADMIN_TG_ID` | Telegram-ID numérique de l'administrateur : accès à `/stats`, `/resync`, réception des exports, alertes sur les posts manqués ; sans limite de requêtes |
| `BOTHUB_API_KEY` | clé de la passerelle de modèles |
| `BOT_DATA` | dossier d'état (index, annotation, SQLite, caches). Par défaut, le dossier du code ; `/data` en Docker |

## Coût

La passe initiale, ce sont les embeddings de tous les posts et une passe d'annotation ; sur un corpus de 600 posts, cela fait des dizaines de centimes. En fonctionnement, chaque recherche coûte un embedding de requête, les requêtes longues ajoutent une reformulation LLM, et l'ingestion d'un post, un embedding plus l'annotation. `/stats` montre les dépenses réelles.

## Fonctionnement

| Fichier | Rôle |
|---|---|
| `bot.py` | bot aiogram : handlers, cartes, paquets de résultats |
| `router.py` | routage du message : commandes admin, intro, recherche ; reformulation des requêtes longues |
| `hybrid.py` | recherche hybride : dense + BM25, fusion RRF, politique de replis |
| `index.py` | index vectoriel `corpus.npz`, recousage incrémental |
| `normalize.py` | annotation LLM de kind / author / source |
| `images.py` | choix du tableau et cache des file_id Telegram |
| `reference.py` | fiches sur l'auteur et le tableau depuis Wikipédia |
| `ingest.py` | ingestion en direct et `/resync` |
| `store.py` | SQLite : utilisateurs, contenu montré, limites, événements |
| `spend.py` | suivi des dépenses LLM |
| `cards.py` | assemblage de la carte : texte, auteur, source, typographie |
| `text_clean.py` | typographie d'affichage |
| `load_corpus.py` | parsing de result.json, manifestes optionnels de recollages et d'OCR |
| `fetch_paintings.py` | téléchargement de la réserve de tableaux depuis Wikimedia Commons |
| `evalset_draft.py` | ébauche d'eval-set : vous annotez les posts pertinents pour vos requêtes |
| `lab/` | mesures de qualité : comparaison de modèles d'embedding, effet du reranking |

Le modèle d'embedding et le modèle d'annotation ont été choisis en mesurant le recall@5 sur un corpus de référence (voir `lab/`) ; tous deux passent par bothub et se remplacent via la constante `MODEL` dans `embed.py` / `normalize.py`.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

160+ tests, tous hors ligne : le LLM et la recherche tournent sous mocks, aucun réseau externe nécessaire.

## Licence

MIT. La réserve de tableaux est du domaine public (Wikimedia Commons) ; la liste avec l'attribution des auteurs est dans `paintings/paintings.json`.
