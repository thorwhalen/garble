"""Tests for TextVectorizer and the word-cloud helpers (wordcloud replaced by a fake)."""

import sys
import types

import numpy as np
import pytest

from garble import base
from garble.base import TextVectorizer, url_to_name

DOCS = {
    "a": "the cat sat on the mat",
    "b": "the dog sat on the log",
    "c": "quantum chromodynamics describes quarks",
}


@pytest.fixture
def model():
    return TextVectorizer().fit(DOCS.values())


def test_word_of_idx_inverts_vocabulary(model):
    for word, idx in model.vocabulary_.items():
        assert model.word_of_idx[idx] == word
    assert len(model.word_of_idx) == 1 + max(model.vocabulary_.values())


def test_call_returns_sparse_row(model):
    row = model("cat mat")
    assert row.shape[-1] == len(model.vocabulary_)
    assert row.nnz == 2


def test_weight_of_word_sorted_by_idf_descending(model):
    w = model.weight_of_word("the cat quarks")
    assert set(w) == {"the", "cat", "quarks"}
    weights = list(w.values())
    assert weights == sorted(weights, reverse=True)
    assert w["the"] == min(weights)  # appears in most documents


def test_unknown_words_are_ignored(model):
    assert model.weight_of_word("zzzzz") == {}


def test_url_to_name():
    assert url_to_name("http://example.com/path") == "example.com"
    assert url_to_name("http://example.com") == "example.com"


class _FakeWordCloud:
    instances = []

    def __init__(self, **kw):
        self.kw, self.words, self.saved = kw, None, None
        _FakeWordCloud.instances.append(self)

    def fit_words(self, weights):
        self.words = weights

    def to_file(self, path):
        self.saved = path
        open(path, "w").write("fake")


@pytest.fixture
def fake_wordcloud(monkeypatch):
    mod = types.ModuleType("wordcloud")
    mod.WordCloud = _FakeWordCloud
    monkeypatch.setitem(sys.modules, "wordcloud", mod)
    _FakeWordCloud.instances = []


def test_word_cloud_gen(fake_wordcloud, model):
    clouds = list(base.word_cloud_gen(DOCS, model=model))
    assert len(clouds) == 3
    assert clouds[0].kw == dict(width=1000, height=1000) and clouds[0].words
    # model=None fits one itself
    assert len(list(base.word_cloud_gen(DOCS))) == 3


def test_compute_and_save_word_clouds(fake_wordcloud, model, tmp_path):
    out = tmp_path / "clouds"
    base.compute_and_save_word_clouds(DOCS, str(out), model=model, ext=".png")
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png", "c.png"]
    n_before = len(_FakeWordCloud.instances)
    base.compute_and_save_word_clouds(DOCS, str(out), model=model, ext=".png")
    assert len(_FakeWordCloud.instances) == n_before  # existing files are skipped
    base.compute_and_save_word_clouds(
        DOCS, str(out), model=model, key_to_filename=lambda k: f"doc_{k}", ext=".png"
    )
    assert (out / "doc_a.png").exists()


def test_errors_are_reported_not_raised(fake_wordcloud, model, capsys, tmp_path):
    class Boom(dict):
        def __getitem__(self, k):
            raise RuntimeError("no text")

    base.compute_and_save_word_clouds(Boom(a="x"), str(tmp_path / "out"), model=model)
    assert "Error on 0 - a: no text" in capsys.readouterr().out
