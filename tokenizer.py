
import json
import re
from collections import Counter


class CaptionTokenizer:
    def __init__(self):
        self.word2idx = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def tokenize(self, text):
        return re.findall(r"[a-z]+|[0-9]+", text.lower())

    def fit(self, captions, min_freq=2):
        counter = Counter()

        for caption in captions:
            counter.update(self.tokenize(caption))

        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }

    def encode(self, text):
        tokens = self.tokenize(text)

        return (
            [self.word2idx["<BOS>"]]
            + [self.word2idx.get(t, 3) for t in tokens]
            + [self.word2idx["<EOS>"]]
        )

    def decode(self, ids):
        words = []

        for idx in ids:
            word = self.idx2word.get(int(idx), "<UNK>")

            if word == "<EOS>":
                break

            if word not in ("<PAD>", "<BOS>"):
                words.append(word)

        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        tokenizer = cls()

        with open(path, "r", encoding="utf-8") as f:
            tokenizer.word2idx = json.load(f)

        tokenizer.idx2word = {
            idx: word for word, idx in tokenizer.word2idx.items()
        }

        return tokenizer
