from collections import Counter, defaultdict


class BpeExample:
    def __init__(self, special_tokens: list[str]) -> None:
        self.special_tokens = special_tokens
        self.decoder: dict[int, bytes] = dict()

        for i, token in enumerate(special_tokens):
            self.decoder[i] = token.encode("utf-8")
        offset = len(special_tokens)
        for i in range(256):
            self.decoder[offset + i] = bytes([i])

        self.encoder: dict[bytes, int] = {token: idx for idx, token in self.decoder.items()}
        self.merges: list[tuple[bytes, ...]] = []

    def _pretokenize(self, text: str) -> dict[tuple[bytes, ...], int]:
        # remove special tokens
        for sptoken in self.special_tokens:
            text = text.replace(sptoken, "")
        word_counts = Counter(text.split())
        items: dict[tuple[bytes, ...], int] = {
            tuple(bytes([b]) for b in word.encode("utf-8")): count for word, count in word_counts.items()
        }
        return items

    def _merge(self, vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
        pairs: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for word, freq in vocab.items():
            for bigram in zip(word, word[1:]):
                pairs[bigram] += freq
        if not pairs:
            raise ValueError("Empty vocab provided")

        to_merge = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]  # (b's', b't')
        self.merges.append(to_merge)

        new_vocab: dict[tuple[bytes, ...], int] = {}
        for word, freq in vocab.items():
            i = 0
            new_word: list[bytes] = []
            while i < len(word):
                if word[i : i + 2] == to_merge:
                    new_word.append(to_merge[0] + to_merge[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        next_id = len(self.decoder)
        new_token = to_merge[0] + to_merge[1]
        self.decoder[next_id] = new_token
        self.encoder[new_token] = next_id

        return new_vocab

    def _merge_check(self, vocab: dict[tuple[bytes, ...], int], num_merges: int = 6) -> dict[tuple[bytes, ...], int]:
        for _ in range(num_merges):
            vocab = self._merge(vocab)
        return vocab

    def train(self, text: str, num_merges: int) -> dict[tuple[bytes, ...], int]:
        pretokens = self._pretokenize(text)
        return self._merge_check(pretokens, num_merges=num_merges)


if __name__ == "__main__":
    text = """low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest"""

    special_tokens = ["<|endoftext|>"]

    tokenizer = BpeExample(special_tokens=special_tokens)
    pretokens = tokenizer._pretokenize(text)
    merges = tokenizer._merge_check(pretokens, 6)
    print(merges)
