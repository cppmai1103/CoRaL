import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.train_gpt2_from_scratch import extract_corpus as ec


def doc(doc_id, text, score=0.99):
    return SimpleNamespace(id=doc_id, text=text, metadata={"language_score": score, "url": "u", "dump": "d", "date": "t"})


def body(tag, n=600):
    return (f"{tag} " * n)[:n]


def collect(stream, target=3, exclude_ids=(), exclude_hashes=(), banned=frozenset({"badword"})):
    seen_ids, seen_hashes = set(), set()
    return ec.collect_documents(
        iter(stream), target, language="xx", config="cfg", source="clean", exclude_ids=set(exclude_ids),
        exclude_hashes=set(exclude_hashes), seen_ids=seen_ids, seen_hashes=seen_hashes, banned=set(banned),
        log=lambda m: None)


class CollectDocuments(unittest.TestCase):
    def test_skips_each_reason_and_stops_at_target(self):
        annotated = body("annotated")
        stream = [
            doc("<a>", body("short", 100)),                      # length
            doc("<b>", body("toolong", 5000)),                   # length
            doc("<annotated-id>", body("x1")),                   # overlap_id
            doc("<c>", annotated),                               # overlap_text (same text, new id)
            doc("<d>", body("badword ")),                        # adult
            doc("<e>", body("keep1")),
            doc("<e>", body("keep2")),                           # duplicate_id
            doc("<f>", body("keep1")),                           # duplicate_text
            doc("<g>", body("keep3")),
            doc("<h>", body("keep4")),
            doc("<i>", body("never reached")),
        ]
        rows, stats = collect(stream, target=3, exclude_ids={"<annotated-id>"}, exclude_hashes={ec.text_hash(annotated)})
        self.assertEqual([r["doc_id"] for r in rows], ["<e>", "<g>", "<h>"])
        self.assertEqual(stats["length"], 2)
        for reason in ("overlap_id", "overlap_text", "adult", "duplicate_id", "duplicate_text"):
            self.assertEqual(stats[reason], 1, reason)
        self.assertFalse(stats["stream_exhausted"])
        self.assertEqual(rows[0]["url"], "u")

    def test_short_stream_is_reported_as_exhausted(self):
        rows, stats = collect([doc("<a>", body("one"))], target=5)
        self.assertEqual(len(rows), 1)
        self.assertTrue(stats["stream_exhausted"])

    def test_text_is_stripped_before_hashing_and_measuring(self):
        text = body("pad").strip()
        rows, _ = collect([doc("<a>", "  \n" + text + "\n ")], target=1)
        self.assertEqual(rows[0]["text"], text)
        self.assertEqual(rows[0]["text_hash"], ec.text_hash(text))


class AnnotatedExclusions(unittest.TestCase):
    def test_reads_clean_and_flag_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "vie"
            folder.mkdir()
            for name, doc_id, text in (("vie_clean.csv", "<one>", "text one"), ("vie_flag.csv", "<two>", "text two")):
                with (folder / name).open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["id", "text"])
                    writer.writeheader()
                    writer.writerow({"id": doc_id, "text": text})
            ids, hashes = ec.load_annotated(Path(tmp), "vie")
        self.assertEqual(ids, {"<one>", "<two>"})
        self.assertEqual(hashes, {ec.text_hash("text one"), ec.text_hash("text two")})

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                ec.load_annotated(Path(tmp), "vie")


class OverlapCheck(unittest.TestCase):
    def test_assertion_fires_on_overlap(self):
        rows, _ = collect([doc("<a>", body("one"))], target=1)
        ec.check_no_overlap(rows, set(), set())
        with self.assertRaises(AssertionError):
            ec.check_no_overlap(rows, {"<a>"}, set())
        with self.assertRaises(AssertionError):
            ec.check_no_overlap(rows, set(), {rows[0]["text_hash"]})


if __name__ == "__main__":
    unittest.main()
