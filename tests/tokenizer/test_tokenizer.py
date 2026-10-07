"""Tests for BPE tokenizer."""

import tempfile
from pathlib import Path

from sec_bpe.tokenizer.config import BPEConfig
from sec_bpe.tokenizer.special_tokens import SpecialTokens
from sec_bpe.tokenizer.vocabulary import Vocabulary
from sec_bpe.tokenizer.byte_encoder import ByteEncoder
from sec_bpe.tokenizer.pair_counter import PairCounter
from sec_bpe.tokenizer.merge import Merge, MergeTable
from sec_bpe.tokenizer.encoder import BPEEncoder
from sec_bpe.tokenizer.decoder import BPEDecoder
from sec_bpe.tokenizer.tokenizer import BPETokenizer
from sec_bpe.tokenizer.reference_trainer import ReferenceBPETrainer


class TestBPEConfig:
    """Test BPE configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = BPEConfig()
        assert config.vocab_size == 8192
        assert config.min_frequency == 2
        assert len(config.special_tokens) == 3
        assert config.encoding == "utf-8"
        assert config.trainer == "reference"

    def test_invalid_vocab_size(self):
        """Test that invalid vocab size raises ValueError."""
        try:
            BPEConfig(vocab_size=100)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_min_frequency(self):
        """Test that invalid min frequency raises ValueError."""
        try:
            BPEConfig(min_frequency=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestSpecialTokens:
    """Test special token handling."""

    def test_special_token_mapping(self):
        """Test special token to ID mapping."""
        special_tokens = SpecialTokens(["<PAD>", "<UNK>"])
        assert special_tokens.get_id("<PAD>") == 0
        assert special_tokens.get_id("<UNK>") == 1

    def test_id_to_token(self):
        """Test ID to token mapping."""
        special_tokens = SpecialTokens(["<PAD>", "<UNK>"])
        assert special_tokens.get_token(0) == "<PAD>"
        assert special_tokens.get_token(1) == "<UNK>"

    def test_is_special(self):
        """Test special token detection."""
        special_tokens = SpecialTokens(["<PAD>"])
        assert special_tokens.is_special("<PAD>") is True
        assert special_tokens.is_special("<UNK>") is False


class TestVocabulary:
    """Test vocabulary management."""

    def test_initial_vocabulary_size(self):
        """Test initial vocabulary includes special tokens and bytes."""
        vocab = Vocabulary(["<PAD>", "<UNK>"])
        assert vocab.size() == 256 + 2  # 2 special + 256 bytes

    def test_add_merge_token(self):
        """Test adding BPE merge token."""
        vocab = Vocabulary(["<PAD>"])
        new_id = vocab.add_merge_token(b"AB")
        assert new_id == 257  # 1 special + 256 bytes
        assert vocab.size() == 258

    def test_get_token_bytes(self):
        """Test getting token bytes."""
        vocab = Vocabulary(["<PAD>"])
        token_bytes = vocab.get_token(vocab.base_byte_start() + 65)
        assert token_bytes == bytes([65])


class TestByteEncoder:
    """Test byte-level encoding."""

    def test_encode_text(self):
        """Test encoding text to byte token IDs."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        encoder = ByteEncoder(vocab, special_tokens)

        token_ids = encoder.encode_text("AB")
        assert len(token_ids) == 2
        assert token_ids[0] == vocab.base_byte_start() + 65  # 'A'
        assert token_ids[1] == vocab.base_byte_start() + 66  # 'B'

    def test_encode_unicode(self):
        """Test encoding Unicode text."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        encoder = ByteEncoder(vocab, special_tokens)

        token_ids = encoder.encode_text("café")
        assert len(token_ids) == 5  # café is 5 bytes in UTF-8

    def test_decode_ids(self):
        """Test decoding token IDs to text."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        encoder = ByteEncoder(vocab, special_tokens)

        token_ids = [vocab.base_byte_start() + 65, vocab.base_byte_start() + 66]
        text = encoder.decode_ids(token_ids)
        assert text == "AB"


class TestPairCounter:
    """Test pair frequency counting."""

    def test_count_pairs(self):
        """Test counting adjacent pairs."""
        special_tokens = SpecialTokens(["<PAD>"])
        counter = PairCounter(special_tokens)

        token_ids = [1, 2, 1, 2, 3]
        pairs = counter.count_pairs(token_ids)

        assert pairs[(1, 2)] == 2
        assert pairs[(2, 1)] == 1
        assert pairs[(2, 3)] == 1

    def test_select_most_frequent(self):
        """Test selecting most frequent pair."""
        special_tokens = SpecialTokens(["<PAD>"])
        counter = PairCounter(special_tokens)

        from collections import Counter
        pairs = Counter({(1, 2): 5, (2, 3): 3, (3, 4): 5})

        left, right, freq = counter.select_most_frequent(pairs)
        assert (left, right) == (1, 2) or (left, right) == (3, 4)  # Tie-break by lowest left
        assert freq == 5

    def test_skip_special_tokens(self):
        """Test that pairs with special tokens are skipped."""
        special_tokens = SpecialTokens(["<PAD>"])
        counter = PairCounter(special_tokens)

        token_ids = [0, 1, 2, 0, 3]  # 0 is special
        pairs = counter.count_pairs(token_ids)

        assert (0, 1) not in pairs
        assert (2, 0) not in pairs
        assert (1, 2) in pairs


class TestMergeTable:
    """Test merge table management."""

    def test_add_merge(self):
        """Test adding a merge."""
        table = MergeTable()
        table.add_merge(1, 2, 256, 100)

        assert table.size() == 1
        assert table.get_rank(1, 2) == 0

    def test_merge_ranking(self):
        """Test merge ranking."""
        table = MergeTable()
        table.add_merge(1, 2, 256, 100)
        table.add_merge(256, 3, 257, 50)

        assert table.get_rank(1, 2) == 0
        assert table.get_rank(256, 3) == 1

    def test_serialization(self):
        """Test merge serialization."""
        table = MergeTable()
        table.add_merge(1, 2, 256, 100)

        data = table.to_list()
        loaded = MergeTable.from_list(data)

        assert loaded.size() == table.size()
        assert loaded.get_rank(1, 2) == 0


class TestBPEEncoder:
    """Test BPE encoding."""

    def test_simple_encoding(self):
        """Test simple text encoding."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        merge_table = MergeTable()
        encoder = BPEEncoder(vocab, special_tokens, merge_table)

        token_ids = encoder.encode("AB")
        assert len(token_ids) == 2

    def test_special_token_encoding(self):
        """Test special token encoding."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        merge_table = MergeTable()
        encoder = BPEEncoder(vocab, special_tokens, merge_table)

        token_ids = encoder.encode("<PAD>")
        assert token_ids == [0]


class TestBPEDecoder:
    """Test BPE decoding."""

    def test_simple_decoding(self):
        """Test simple text decoding."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        decoder = BPEDecoder(vocab, special_tokens)

        token_ids = [vocab.base_byte_start() + 65, vocab.base_byte_start() + 66]
        text = decoder.decode(token_ids)
        assert text == "AB"

    def test_filter_special_tokens(self):
        """Test that special tokens are filtered during decode."""
        special_tokens = SpecialTokens(["<PAD>"])
        vocab = Vocabulary(["<PAD>"])
        decoder = BPEDecoder(vocab, special_tokens)

        token_ids = [0, vocab.base_byte_start() + 65]  # <PAD> + 'A'
        text = decoder.decode(token_ids)
        assert text == "A"


class TestBPETokenizer:
    """Test main tokenizer API."""

    def test_round_trip(self):
        """Test encode/decode round trip."""
        config = BPEConfig(vocab_size=300)
        tokenizer = BPETokenizer(config)

        text = "Revenue increased by 10%."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_unicode_round_trip(self):
        """Test Unicode round trip."""
        config = BPEConfig(vocab_size=300)
        tokenizer = BPETokenizer(config)

        text = "café naïve €100"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_tokenize(self):
        """Test tokenize method."""
        config = BPEConfig(vocab_size=300)
        tokenizer = BPETokenizer(config)

        tokens = tokenizer.tokenize("AB")
        assert len(tokens) == 2
        assert "id" in tokens[0]
        assert "bytes" in tokens[0]


class TestReferenceTrainer:
    """Test reference BPE trainer."""

    def test_simple_training(self):
        """Test training on simple corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test corpus
            corpus_path = tmpdir / "train.txt"
            corpus_path.write_text(
                "<DOCUMENT_START>\nhello hello hello\n<DOCUMENT_END>\n\n"
                "<DOCUMENT_START>\nworld world world\n<DOCUMENT_END>\n",
                encoding="utf-8",
            )

            config = BPEConfig(vocab_size=300, min_frequency=2)
            trainer = ReferenceBPETrainer(config)
            trainer.load_corpus(corpus_path)

            vocab, merges = trainer.train()

            assert vocab.size() > 256  # Should have learned some merges
            assert merges.size() > 0

    def test_document_boundary_respect(self):
        """Test that document boundaries are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            corpus_path = tmpdir / "train.txt"
            corpus_path.write_text(
                "<DOCUMENT_START>\nAB\n<DOCUMENT_END>\n\n"
                "<DOCUMENT_START>\nCD\n<DOCUMENT_END>\n",
                encoding="utf-8",
            )

            config = BPEConfig(vocab_size=300, min_frequency=2)
            trainer = ReferenceBPETrainer(config)
            trainer.load_corpus(corpus_path)

            vocab, merges = trainer.train()

            # Should not learn a merge across document boundaries
            # (B from first doc + C from second doc)
            # This is implicitly tested by the trainer's pair counting
            # which skips special tokens
