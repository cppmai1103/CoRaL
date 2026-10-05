import json
import tempfile
import unittest
from pathlib import Path

from src import extract_llm_scores as e

DIMS = e.DIMENSIONS


def row(doc_id, language, text, scores, split="train"):
    r = {"doc_id": doc_id, "language": language, "text": text, "text_hash": e.text_hash(text)}
    r.update({d: str(s) for d, s in zip(DIMS, scores)})
    r.update({f"split_{d}": split for d in DIMS})
    return r


def llm(doc_id, text, scores, status="ok"):
    rec = {"id": doc_id, "parse_status": status, "error": None, "justification": "x", "text": text}
    rec.update({d: s for d, s in zip(DIMS, scores)})
    return rec


class Match(unittest.TestCase):
    def run_match(self, table, files):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, recs in files.items():
                p = Path(d) / name
                p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
                paths.append(p)
            return e.match_records(table, sorted(paths))

    def test_match_by_id_and_language_from_table(self):
        table = [row("a", "vie", "xin chào", [1, 2, 3, 4, 5]), row("b", "thai", "สวัสดี", [0, 0, 0, 0, 0])]
        docs, files, langs = self.run_match(table, {"vie_Latn.jsonl": [llm("a", "xin chào", [2, 2, 3, 5, 4])],
                                                    "mya_Mymr.jsonl": [llm("z", "other", [1] * 5)]})
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["language"], "vie")
        self.assertEqual(docs[0]["llm"]["cleanliness"], 5)
        self.assertEqual(docs[0]["human"]["cleanliness"], 4.0)
        self.assertEqual(files["mya_Mymr.jsonl"]["matched"], 0)
        self.assertEqual(langs["thai"]["missing_ids"], ["b"])

    def test_reused_placeholder_id_resolved_by_text(self):
        table = [row("<urn:pdid:1>", "fil", "first text", [1] * 5), row("<urn:pdid:1>", "fil", "second text", [3] * 5)]
        docs, _, _ = self.run_match(table, {"fil_Latn.jsonl": [llm("<urn:pdid:1>", "second text", [4] * 5),
                                                               llm("<urn:pdid:1>", "unknown text", [0] * 5)]})
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["human"]["reasoning"], 3.0)

    def test_failed_parse_and_text_mismatch_are_kept_and_counted(self):
        table = [row("a", "indo", "teks asli", [2] * 5)]
        docs, _, langs = self.run_match(table, {"indo_Latn.jsonl": [llm("a", "teks lain", [None] * 5, "failed")]})
        self.assertEqual(len(docs), 1)
        self.assertIsNone(docs[0]["llm"]["reasoning"])
        self.assertEqual(langs["indo"]["parse_failed"], 1)
        self.assertEqual(langs["indo"]["text_mismatch"], 1)

    def test_unknown_id_falls_back_to_text_hash(self):
        table = [row("Blog post (broken id)", "fil", "the same text", [2] * 5)]
        docs, _, _ = self.run_match(table, {"fil_Latn.jsonl": [llm("<urn:uuid:real>", "the same text", [3] * 5)]})
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["doc_id"], "Blog post (broken id)")


    def test_whitespace_only_difference_counts_as_same_text(self):
        table = [row("a", "vie", "Xin chào\n\nbạn", [1] * 5)]
        docs, _, langs = self.run_match(table, {"vie_Latn.jsonl": [llm("a", "Xin  chào\nbạn ", [2] * 5)]})
        self.assertTrue(docs[0]["llm_text_matches"])
        self.assertEqual(langs["vie"]["text_mismatch"], 0)


class Agreement(unittest.TestCase):
    def test_perfect_rank_agreement_and_mae(self):
        docs = [{"language": "vie", "llm": {"reasoning": s}, "human": {"reasoning": s + 0.5}} for s in range(5)]
        a = e.agreement(docs, "reasoning")
        self.assertEqual(a["n"], 5)
        self.assertAlmostEqual(a["mae"], 0.5)
        self.assertAlmostEqual(a["spearman"], 1.0)
        self.assertAlmostEqual(a["pearson"], 1.0)

    def test_skips_missing_scores(self):
        docs = [{"language": "vie", "llm": {"reasoning": None}, "human": {"reasoning": 2.0}},
                {"language": "vie", "llm": {"reasoning": 1}, "human": {"reasoning": None}}]
        self.assertEqual(e.agreement(docs, "reasoning")["n"], 0)

    def test_ranks_average_ties(self):
        self.assertEqual(e._ranks([5, 1, 5, 3]), [3.5, 1.0, 3.5, 2.0])


if __name__ == "__main__":
    unittest.main()
