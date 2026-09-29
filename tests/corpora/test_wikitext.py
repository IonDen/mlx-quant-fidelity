"""load_wikitext2 with the Hub fetch and the parquet read replaced (no network, no model)."""

import huggingface_hub
import pytest

from mlx_quant_fidelity.corpora import wikitext


class _BosTokenizer:
    """Llama-3 style: encode() prepends token 100 unless add_special_tokens=False."""

    name_or_path = "fake/llama-like"

    def encode(self, text, add_special_tokens=True):
        body = [ord(ch) % 50 + 1 for ch in text]
        return ([100] if add_special_tokens else []) + body


class _NoBosTokenizer:
    """Qwen style: no BOS either way."""

    name_or_path = "fake/qwen-like"

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) % 50 + 1 for ch in text]


class _EosAppendingTokenizer:
    """T5 style: encode() APPENDS token 2 (EOS); nothing leads the sequence."""

    name_or_path = "fake/eos-appender"

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) % 50 + 1 for ch in text] + ([2] if add_special_tokens else [])


class _NoKwargTokenizer:
    """encode() does not accept add_special_tokens at all."""

    name_or_path = "fake/plain"

    def encode(self, text):
        return [ord(ch) % 50 + 1 for ch in text]


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "unused.parquet")
    monkeypatch.setattr(wikitext, "_read_parquet_text", lambda path: "abcdefghijklmnopqrstuvwxyz")


def test_wikitext_records_bos_when_tokenizer_adds_one(offline):
    """Reds if provenance says bos_policy='none' while chunk 0 starts with the BOS token."""
    corpus = wikitext.load_wikitext2(_BosTokenizer(), chunk_length=8)
    assert int(corpus.chunks[0][0]) == 100  # what is scored: BOS leads chunk 0
    assert corpus.provenance.bos_policy == "first-chunk"


def test_wikitext_records_none_for_qwen_style_tokenizer(offline):
    corpus = wikitext.load_wikitext2(_NoBosTokenizer(), chunk_length=8)
    assert corpus.provenance.bos_policy == "none"


def test_wikitext_records_none_when_the_tokenizer_appends_eos(offline):
    """Bug: comparing lengths alone records a trailing EOS as a leading BOS ('first-chunk'), so
    the report claims chunk 0 starts with a special token that is not there."""
    corpus = wikitext.load_wikitext2(_EosAppendingTokenizer(), chunk_length=8)
    assert corpus.provenance.bos_policy == "none"


def test_wikitext_records_none_when_encode_rejects_the_kwarg(offline):
    """Reds if a tokenizer without add_special_tokens crashes the loader instead of 'none'."""
    corpus = wikitext.load_wikitext2(_NoKwargTokenizer(), chunk_length=8)
    assert corpus.provenance.bos_policy == "none"


def test_load_wikitext2_caps_and_records_post_cap_tokens(offline):
    """Reds if n_tokens counts the uncapped corpus: 26 chars, chunk 8, max 2 -> 16 tokens."""
    corpus = wikitext.load_wikitext2(_NoBosTokenizer(), chunk_length=8, max_chunks=2)
    assert len(corpus.chunks) == 2
    assert corpus.provenance.n_tokens == 16


def test_wikitext_fetch_is_revision_pinned(monkeypatch):
    """Reds if the dataset is fetched from a moving branch: the corpus (hence every number)
    could change under an unchanged report, and the report would not say which revision."""
    seen: dict[str, object] = {}

    def fake_download(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "unused.parquet"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(wikitext, "_read_parquet_text", lambda path: "abcdefghijklmnop")
    corpus = wikitext.load_wikitext2(_NoBosTokenizer(), chunk_length=8)
    pinned = "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert seen["kwargs"]["revision"] == pinned  # type: ignore[index]
    assert corpus.provenance.dataset_revision == pinned


class _EmptyPlainTokenizer:
    """Pathological: the plain encoding of the probe text is empty, the default adds a BOS."""

    name_or_path = "fake/empty-plain"

    def encode(self, text, add_special_tokens=True):
        return [100] if add_special_tokens else []


def test_bos_policy_handles_an_empty_plain_encoding():
    """Bug: `without_special[0]` raised IndexError while loading the corpus when the plain
    encoding came back empty; a leading special token must still be recorded."""
    assert wikitext._bos_policy(_EmptyPlainTokenizer()) == "first-chunk"
