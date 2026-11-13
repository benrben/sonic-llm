"""The sample library and its answer key: the generator reproduces the committed files byte for byte, and every key matches
its document."""

import json
import sys
from pathlib import Path

import pytest

from sonic_playground import documents, grading
from sonic_playground.settings import PLAYGROUND_DIR

sys.path.insert(0, str(PLAYGROUND_DIR / "tools"))
import make_samples  # noqa: E402

SAMPLES = PLAYGROUND_DIR / "samples"


@pytest.fixture(scope="module")
def key() -> dict:
    return json.loads((SAMPLES / "answers.json").read_text(encoding="utf-8"))


def test_the_generator_reproduces_the_committed_samples_byte_for_byte(tmp_path):
    out = tmp_path / "out"
    make_samples.write(out)
    made = {p.name: p.read_bytes() for p in out.iterdir()}
    committed = {p.name: p.read_bytes() for p in SAMPLES.iterdir()}
    assert sorted(made) == sorted(committed)
    assert [n for n in made if made[n] != committed[n]] == []


def test_the_generator_only_replaces_its_own_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.md").write_text("mine")
    (out / "oldname-x.md").write_text("old")  # not NNN-kind: not the generator's
    (out / "999-gone.md").write_text("old")  # a name like those it makes: replaced
    make_samples.write(out)
    assert (out / "notes.md").exists() and (out / "oldname-x.md").exists() and not (out / "999-gone.md").exists()


def test_the_library_is_256_documents_of_ten_kinds_under_1_2_mb(key):
    docs = sorted(SAMPLES.glob("*.md"))
    sizes = [p.stat().st_size for p in docs]
    assert len(docs) == 256 and all(2500 <= n <= 4000 for n in sizes), (min(sizes), max(sizes))
    assert sum(p.stat().st_size for p in SAMPLES.iterdir()) < 1_200_000
    assert set(key["documents"]) == {p.name for p in docs} and len(set(key["labels"])) == 10
    assert {k["kind"] for k in key["documents"].values()} == set(key["labels"])
    assert len({p.read_text() for p in docs}) == 256  # every document differs


def test_the_answer_key_is_in_the_text_of_each_document(key):
    for name, k in key["documents"].items():
        text = grading.tokens((SAMPLES / name).read_text(encoding="utf-8"))
        assert 4 <= len(k["fields"]) <= 6 and 4 <= len(k["questions"]) <= 6, name
        for q in k["questions"]:
            assert any(grading._has(text, grading._form(a)) for a in q["accept"]), (name, q)
        for field, value in k["fields"].items():
            assert grading._has(text, grading._form(value)), (name, field, value)


def test_a_question_has_one_answer_and_its_text_would_not_pass_for_another_question(key):
    """No two questions of a document accept the same form, and reading the right answer back is graded right."""
    for name, k in key["documents"].items():
        firsts = [q["accept"][0] for q in k["questions"]]
        assert len(set(firsts)) == len(firsts), name
        for q in k["questions"]:
            assert grading.grade_answer(q["accept"][0], q["accept"]) == grading.CORRECT, (name, q)


def test_the_key_is_not_a_document_and_documents_know_their_key(cfg):
    cfg.samples_dir.mkdir(exist_ok=True)
    make_samples.write(cfg.samples_dir)
    paths = [f["path"] for f in documents.list_files()]
    assert len(paths) == 256 and "samples/answers.json" not in paths
    k = documents.key_of(paths[0])
    assert k["kind"] == "meeting notes" and documents.key_of("uploads/" + paths[0].split("/")[1]) is None
    assert all(f["keyed"] for f in documents.list_files())
    (cfg.documents_dir).mkdir(parents=True)
    documents.save_upload("mine.md", b"my text")
    assert [f["keyed"] for f in documents.list_files()][-1] is False
