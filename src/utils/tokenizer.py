"""Minimal whitespace tokenizer used only by the `tiny` text_backend for CPU smoke
tests (Section 30). On Kaggle with `text_backend: hf`, use the pretrained VLM's own
tokenizer (transformers.AutoTokenizer) instead — this is NOT meant to be a real
language tokenizer.
"""
from __future__ import annotations

from typing import Dict, List

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class SimpleTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab[PAD]
        self.bos_id = vocab[BOS]
        self.eos_id = vocab[EOS]
        self.unk_id = vocab[UNK]

    @classmethod
    def build_from_texts(cls, texts: List[str]) -> "SimpleTokenizer":
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for text in texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int = 32) -> List[int]:
        ids = [self.bos_id]
        for word in text.lower().split():
            ids.append(self.vocab.get(word, self.unk_id))
        ids.append(self.eos_id)
        ids = ids[:max_len]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = [self.inv_vocab.get(i, UNK) for i in ids]
        words = [w for w in words if w not in (PAD, BOS, EOS)]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.vocab)
