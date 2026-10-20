import numpy as np

try:
    import hnswlib

    HNSW_AVAILABLE = True
except Exception:  # optional dependency, never required at startup
    HNSW_AVAILABLE = False
    hnswlib = None


VECTOR_DIM = 256


def _hash_vector(text: str) -> np.ndarray:
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    for term in str(text or "").lower().split():
        idx = (hash(term) & 0x7fffffff) % VECTOR_DIM
        vec[idx] += 1.0
    for index in range(max(len(str(text)) - 3, 0)):
        gram = str(text).lower()[index : index + 4]
        idx = (hash(gram) & 0x7fffffff) % VECTOR_DIM
        vec[idx] += 0.65
    norm = np.linalg.norm(vec)
    if norm and vec is not None and not np.isnan(norm):
        vec = vec / norm
    return vec.astype(np.float32)


class HnswIndex:
    """Small, file-free optional vector index built on hnswlib when available."""

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self.dim = dim
        self.chunks = []
        self.index = None

    def build(self, chunks) -> bool:
        self.chunks = list(chunks)
        self.index = None
        if not HNSW_AVAILABLE or not self.chunks:
            return False
        try:
            self.index = hnswlib.Index(space="cosine", dim=self.dim)
            self.index.init_index(max(1, len(self.chunks)), ef_construction=200, M=16)
            vectors = np.stack([_hash_vector(getattr(c, "text", str(c))) for c in self.chunks])
            self.index.add_items(vectors, np.arange(len(self.chunks)))
            self.index.set_ef(64)
            return True
        except Exception:
            self.index = None
            return False

    def search(self, query: str, top_k: int = 4) -> list:
        from .rag import RetrievalHit

        hits = []
        if not self.chunks or not self.index or top_k < 1:
            return hits
        try:
            labels, distances = self.index.knn_query(_hash_vector(query).reshape(1, -1), k=min(top_k, len(self.chunks)))
        except Exception:
            return hits
        pairs = list(zip(labels[0].tolist(), distances[0].tolist()))
        pairs.sort(key=lambda item: item[1])
        for rank, (idx, dist) in enumerate(pairs[:top_k], start=1):
            try:
                chunk = self.chunks[int(idx)]
            except (IndexError, TypeError):
                continue
            score = max(0.0, 1.0 - float(dist))
            hits.append(RetrievalHit(source_id=f"H{rank}", chunk=chunk, score=score))
        return hits