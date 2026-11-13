"""The document library: the bundled samples, the user's uploads, and nothing outside them."""

import pytest

from sonic_playground import documents
from sonic_playground.documents import DocumentError


@pytest.fixture
def library(cfg):
    cfg.samples_dir.mkdir()
    (cfg.samples_dir / "a.md").write_text("# A\n" + "word " * 400)
    (cfg.samples_dir / "notes").mkdir()
    (cfg.samples_dir / "notes" / "b.txt").write_text("short")
    (cfg.samples_dir / "image.png").write_bytes(b"\x89PNG")
    return cfg


def test_the_library_lists_text_files_of_both_folders_with_token_estimates(library):
    documents.save_upload("mine.md", b"my own text")
    files = documents.list_files()
    assert [f["path"] for f in files] == ["samples/a.md", "samples/notes/b.txt", "uploads/mine.md"]
    assert files[0]["est_tokens"] == files[0]["bytes"] // 4


def test_read_file_reads_inside_the_two_folders_only(library):
    documents.save_upload("mine.md", b"my own text")
    assert documents.read_file("samples/notes/b.txt") == "short"
    assert documents.read_file("uploads/mine.md", 5) == "my ow"
    for bad in ("../runs.jsonl", "samples/../../x", "samples/missing.md", "other/a.md", "samples", "samples/image.png/.."):
        with pytest.raises(DocumentError):
            documents.read_file(bad)


def test_an_upload_keeps_a_safe_name_and_replaces_one_of_the_same_name(library):
    assert documents.save_upload("../../etc/pass wd?.md", b"x") == "uploads/pass wd_.md"
    documents.save_upload("a.txt", b"one")
    documents.save_upload("a.txt", b"two")
    assert documents.read_file("uploads/a.txt") == "two"


@pytest.mark.parametrize("name,data,why", [("x.exe", b"x", "only text files"), ("a.md", b"", "empty"), ("a.md", b"\xff\xfe\x00", "UTF-8"),
                                           ("a.md", b"x" * (documents.MAX_UPLOAD_BYTES + 1), "larger")])
def test_an_upload_that_is_not_a_small_text_file_is_refused(library, name, data, why):
    with pytest.raises(DocumentError, match=why):
        documents.save_upload(name, data)
    assert not (library.documents_dir).exists() or not list(library.documents_dir.iterdir())


def test_an_uploaded_document_can_be_removed_but_a_sample_and_other_paths_cannot(library):
    documents.save_upload("mine.md", b"my own text")
    assert documents.delete_upload("mine.md") == "uploads/mine.md"
    assert [f["path"] for f in documents.list_files()] == ["samples/a.md", "samples/notes/b.txt"]
    for bad in ("mine.md", "a.md", "../samples/a.md", "samples/a.md", "..", ""):
        with pytest.raises(DocumentError):
            documents.delete_upload(bad)
    assert (library.samples_dir / "a.md").is_file()
