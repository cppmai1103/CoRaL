import unittest
from collections import Counter
from pathlib import Path

from src import prepare_llm_rater_dataset as p
from src.extract_llm_scores import DIMENSIONS, content_hash


def rec(doc_id, text, score=3):
    r = {"id": doc_id, "text": text, "parse_status": "ok"}
    r.update({d: score for d in DIMENSIONS})
    return r


class FileLanguage(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(p.language_of_file(Path("khm_Khmr.jsonl")), "khmer")
        self.assertEqual(p.language_of_file(Path("zsm_Latn.jsonl")), "malay")
        self.assertEqual(p.language_of_file(Path("indo_Latn.jsonl")), "indo")
        self.assertIsNone(p.language_of_file(Path("mya_Mymr.jsonl")))


class FilterPool(unittest.TestCase):
    def test_leakage_guards(self):
        docs = {"vie": [rec("a", "human test doc"), rec("b", "same text as human val"), rec("c", "pilot test doc"),
                        rec("d", "fine"), rec("e", "fine"), rec("f", "   ")]}
        pool, dropped = p.filter_pool(docs, human_keys={("vie", "a")}, human_hashes={content_hash("same  text as\nhuman val")},
                                      pilot_keys={("vie", "c")})
        self.assertEqual([r["id"] for r in pool["vie"]], ["d"])
        self.assertEqual(dropped["vie"], Counter(human_validation_or_test=2, pilot_validation_or_test=1,
                                                 duplicate_text=1, empty_text=1))


class SplitValidation(unittest.TestCase):
    def test_stratified_fraction_and_failed_parses_left_out(self):
        recs = [rec(f"d{i}", f"t{i}", score=i % 2) for i in range(40)] + [rec("none", "x", score=None)]
        split = p.split_validation(recs, "reasoning", 0.1, seed=1)
        self.assertNotIn("none", split)
        val = [k for k, v in split.items() if v == "validation"]
        self.assertEqual(len(val), 4)  # 2 per score value (20 docs each)
        self.assertEqual(split, p.split_validation(recs, "reasoning", 0.1, seed=1))

    def test_small_score_value_still_gets_one_validation_doc(self):
        recs = [rec("x1", "a", score=5), rec("x2", "b", score=5)] + [rec(f"d{i}", f"t{i}", score=1) for i in range(30)]
        split = p.split_validation(recs, "cleanliness", 0.1, seed=1)
        self.assertEqual(sum(split[k] == "validation" for k in ("x1", "x2")), 1)


class RelabelHumanSplits(unittest.TestCase):
    def test_train_validation_relabelled_test_kept_missing_dropped(self):
        manifest = {"reasoning": [
            {"doc_id": "a", "language": "vie", "duplicate_group_id": "g1", "score": "1.5", "split": "train"},
            {"doc_id": "b", "language": "vie", "duplicate_group_id": "g2", "score": "2.0", "split": "validation"},
            {"doc_id": "c", "language": "vie", "duplicate_group_id": "g3", "score": "3.5", "split": "test"},
            {"doc_id": "d", "language": "vie", "duplicate_group_id": "g4", "score": "0.5", "split": "train"},
            {"doc_id": "e", "language": "vie", "duplicate_group_id": "g5", "score": "4.0", "split": "train"}]}
        llm = {("vie", "a"): {"reasoning": 3}, ("vie", "b"): {"reasoning": 1}, ("vie", "c"): {"reasoning": 0},
               ("vie", "e"): {"reasoning": None}}
        rows, dropped = p.relabel_human_splits(manifest, llm)
        by_id = {r["doc_id"]: r for r in rows["reasoning"]}
        self.assertEqual(by_id["a"]["score"], 3.0)
        self.assertEqual(by_id["a"]["label_source"], "llm")
        self.assertEqual(by_id["b"]["score"], 1.0)
        self.assertEqual(by_id["c"]["score"], "3.5")  # test keeps the human label
        self.assertEqual(by_id["c"]["label_source"], "human")
        self.assertNotIn("d", by_id)
        self.assertNotIn("e", by_id)
        self.assertEqual(dropped["reasoning"], Counter({"train: no LLM record": 1, "train: LLM parse failed": 1}))


if __name__ == "__main__":
    unittest.main()
