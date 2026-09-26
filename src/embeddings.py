"""Local embeddings. Default is BAAI/bge-small-en-v1.5 - free, runs on CPU.

A deterministic hashing embedder (EMBED_BACKEND=hash) keeps the pipeline
runnable with no model download, which is what the unit tests use.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

from .config import Settings

#: BGE retrieval models expect this instruction on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    name: str = "embedder"
    dim: int = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """sentence-transformers wrapper; the model is loaded on first use."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._use_query_prefix = "bge" in model_name.lower() and "-en" in model_name.lower()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self._device)
            # renamed in sentence-transformers 6.x
            getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            self.dim = int(getter())
        return self._model

    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit vectors -> cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts, show_progress=len(texts) > 256)

    def embed_query(self, text: str) -> list[float]:
        payload = BGE_QUERY_PREFIX + text if self._use_query_prefix else text
        return self._encode([payload])[0]


class OnnxEmbedder(Embedder):
    """Same bge weights, run through onnxruntime instead of torch.

    Roughly 120 MB of dependencies instead of ~1 GB, and about 200 MB of RAM,
    which is what makes a small always-on server viable. Vectors match the
    torch path to ~1e-3 cosine.
    """

    MAX_LENGTH = 512

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._use_query_prefix = "bge" in model_name.lower() and "-en" in model_name.lower()

    def _load(self) -> None:
        if self._session is not None:
            return
        import numpy as np  # noqa: F401  (imported here to keep module import cheap)
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        weights = hf_hub_download(self.name, "onnx/model.onnx")
        vocab = hf_hub_download(self.name, "tokenizer.json")

        tokenizer = Tokenizer.from_file(vocab)
        tokenizer.enable_truncation(max_length=self.MAX_LENGTH)
        tokenizer.enable_padding(length=None)
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 0  # let onnxruntime pick
        self._session = ort.InferenceSession(
            weights, options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.dim = int(self._session.get_outputs()[0].shape[-1])

    def _encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        encodings = self._tokenizer.encode_batch(list(texts))
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        hidden = self._session.run(None, feeds)[0]
        pooled = hidden[:, 0, :]  # BGE pools the CLS token, not the mean
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype("float32").tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._encode_batch(texts[start : start + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        self._load()
        payload = BGE_QUERY_PREFIX + text if self._use_query_prefix else text
        return self._encode_batch([payload])[0]


class HashEmbedder(Embedder):
    """Offline bag-of-ngrams hashing. Lexical, but deterministic and instant."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _WORD_RE.findall(text.lower())
        grams: Iterable[str] = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(settings: Settings) -> Embedder:
    backend = settings.embed_backend
    if backend == "hash":
        return HashEmbedder()
    if backend == "onnx":
        return OnnxEmbedder(settings.embed_model, batch_size=settings.embed_batch_size)
    return SentenceTransformerEmbedder(settings.embed_model, batch_size=settings.embed_batch_size)
