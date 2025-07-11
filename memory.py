import sqlite3, pathlib, faiss, numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM            = 384
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def _embed(text):
    # Accepts str or list[str]
    if isinstance(text, str):
        v = _model.encode([text])
    else:
        v = _model.encode(text)
    return _normalize(v.astype("float32"))

def _blob_to_vec(blob: bytes) -> np.ndarray:
    v = np.frombuffer(blob, dtype="float32")
    return v.reshape(1, -1)  # always shape (1, EMBED_DIM)

def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.flatten().astype("float32").tobytes()

class AgentMemory:
    def __init__(self, db_path: pathlib.Path):
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memories("
            "id INTEGER PRIMARY KEY, "
            "text  TEXT UNIQUE, "
            "vec   BLOB, "
            "score REAL, "
            "tag   TEXT)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_tag ON memories(tag);")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_score ON memories(score);")
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # cosine similarity
        self._id_map = {}
        self._load_all()

    def add(self, text: str, score: float, tag: str):
        vec = _embed(text)[0]  # always 1D
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO memories(text, vec, score, tag) VALUES(?,?,?,?)",
                (text, _vec_to_blob(vec), score, tag),
            )
            if cur.rowcount:
                row_id = cur.lastrowid
                self.index.add(vec.reshape(1, -1))
                self._id_map[self.index.ntotal - 1] = row_id
                self._maybe_prune()

    def retrieve(
        self, query: str, top_k: int = 8, want_tag: str | None = None, min_sim: float = 0.5
    ) -> list[str]:
        """Return up to top_k relevant memories, optionally filtering by tag and similarity threshold."""
        if self.index.ntotal == 0:
            return []

        qvec = _embed(query)
        D, I = self.index.search(qvec, min(self.index.ntotal, top_k*2))
        memories = []
        for idx, sim in zip(I[0], D[0]):
            if idx == -1 or sim < min_sim:
                continue
            text = self._id_to_text(idx)
            tag_ = self._id_to_tag(idx)
            if want_tag and tag_ != want_tag:
                continue
            if text and text not in memories:
                memories.append(text)
            if len(memories) >= top_k:
                break
        return memories

    def retrieve_with_scores(
        self, query: str, top_k: int = 8, want_tag: str | None = None, min_sim: float = 0.5
    ) -> list[tuple[str, float]]:
        """Return (text, similarity) tuples for debug or advanced filtering."""
        if self.index.ntotal == 0:
            return []
        qvec = _embed(query)
        D, I = self.index.search(qvec, min(self.index.ntotal, top_k*2))
        out = []
        for idx, sim in zip(I[0], D[0]):
            if idx == -1 or sim < min_sim:
                continue
            text = self._id_to_text(idx)
            tag_ = self._id_to_tag(idx)
            if want_tag and tag_ != want_tag:
                continue
            if text and text not in [t for t, _ in out]:
                out.append((text, float(sim)))
            if len(out) >= top_k:
                break
        return out

    def _load_all(self):
        self.index.reset()
        self._id_map = {}
        rows = self.db.execute("SELECT id, vec FROM memories ORDER BY id ASC").fetchall()
        if rows:
            mat = np.vstack([_blob_to_vec(r[1]) for r in rows])
            mat = _normalize(mat)
            self.index.add(mat)
            self._id_map = {i: row_id for i, (row_id, _) in enumerate(rows)}

    def _id_to_text(self, faiss_idx: int) -> str:
        row_id = self._id_map.get(faiss_idx)
        if row_id is None:
            return ""
        res = self.db.execute("SELECT text FROM memories WHERE id=?", (row_id,)).fetchone()
        return res[0] if res else ""

    def _id_to_tag(self, faiss_idx: int) -> str:
        row_id = self._id_map.get(faiss_idx)
        if row_id is None:
            return ""
        res = self.db.execute("SELECT tag FROM memories WHERE id=?", (row_id,)).fetchone()
        return res[0] if res else ""

    def _maybe_prune(self, max_rows: int = 500, min_score: float = 0.2):
        count = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count <= max_rows:
            return
        excess = count - max_rows
        self.db.execute(
            "DELETE FROM memories WHERE id IN ("
            "SELECT id FROM memories WHERE score < ? ORDER BY id ASC LIMIT ?)",
            (min_score, excess),
        )
        self.db.commit()
        self._load_all()

    def clear_all(self):
        with self.db:
            self.db.execute("DELETE FROM memories;")
        self._load_all()

