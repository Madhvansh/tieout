"""Dense encoder for the v2 retriever: BAAI/bge-small-en-v1.5 on CPU, onnxruntime and tokenizers only.

Model     BAAI/bge-small-en-v1.5 (MIT licence): 33M parameters, 384 dimensions, a widely used small
          English retrieval model. Its own Hugging Face repo publishes an ONNX export
          (onnx/model.onnx, 133 MB fp32); that file and tokenizer.json are downloaded once, pinned
          to one revision and checked by size and SHA-256, into baseline/cache/bge-small-en-v1.5/.
Usage     as the model card says: a query gets the instruction prefix below, a passage gets none;
          the vector is the [CLS] hidden state, L2-normalised, so a dot product is the cosine.
          Inputs are cut at 512 tokens, the model's limit.
Compute   CPU only, one process, at most 6 intra-op threads. Texts are sorted by length before
          batching so a batch pads to similar lengths; results come back in input order.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_DIR = ROOT / "baseline" / "cache" / "bge-small-en-v1.5"
FILES = {"onnx/model.onnx": (133_093_490, "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"),
         "tokenizer.json": (711_396, None)}
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 512
DIM = 384
THREADS = 6
BATCH = 16

_ENCODER = None


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(name: str, size: int, sha: str) -> None:
    """Download one model file from huggingface.co at the pinned revision."""
    import requests
    path = MODEL_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "tieout-baseline/2"}) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    if part.stat().st_size != size or (sha and _sha256(part) != sha):
        part.unlink()
        raise RuntimeError(f"{name}: downloaded file does not match the pinned size or SHA-256")
    part.replace(path)


def download() -> pathlib.Path:
    """Make sure the model files are in baseline/cache; fetch the missing ones."""
    for name, (size, sha) in FILES.items():
        path = MODEL_DIR / name
        if not path.exists() or path.stat().st_size != size:
            print(f"dense  downloading {REPO}/{name} ({size / 1e6:.0f} MB)", flush=True)
            _fetch(name, size, sha)
    return MODEL_DIR


class Encoder:
    """Text to unit vectors with the bge-small ONNX model."""

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        download()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def _batch(self, texts: list) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        cls = self.session.run(None, feed)[0][:, 0, :]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)

    def encode(self, texts: list, prefix: str = "") -> np.ndarray:
        """Unit vectors (len(texts), 384), float32, in input order."""
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            out[idx] = self._batch([prefix + texts[i] for i in idx])
        return out

    def count_tokens(self, text: str) -> int:
        """Tokens the model sees for a text (at most 512)."""
        return len(self.tokenizer.encode(text).ids)


def encoder() -> Encoder:
    """The process-wide encoder, loaded on first use."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Encoder()
    return _ENCODER


def embed_query(question: str) -> np.ndarray:
    """Unit vector for a question, with the model's query instruction."""
    return encoder().encode([question], prefix=QUERY_PREFIX)[0]
