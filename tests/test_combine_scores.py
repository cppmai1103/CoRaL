import csv
import tempfile
import unittest
from pathlib import Path

from src.train_gpt2_from_scratch import combine_scores as cs
from src.train_gpt2_from_scratch import score_pool as sp


def write(folder: Path, language: str, rows: dict):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{language}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(sp.OUT_FIELDS)
        for doc_id, raw in rows.items():
            w.writerow([doc_id, language, f"{raw:.6f}", f"{min(5, max(0, raw)):.6f}", 100])


class Combine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write(self.root / "edu", "vie", {"<a>": 2.0, "<b>": 5.6, "<c>": 1.0})
        write(self.root / "clean", "vie", {"<a>": 4.0, "<b>": 5.9, "<c>": -0.4})

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_averages_the_clipped_scores(self):
        rows = {r["doc_id"]: r for r in cs.combine_language("vie", [self.root / "edu", self.root / "clean"], "score")}
        self.assertAlmostEqual(float(rows["<a>"]["score_raw"]), 3.0)
        self.assertAlmostEqual(float(rows["<b>"]["score_raw"]), 5.0)     # (5.0 + 5.0) / 2, both clipped
        self.assertAlmostEqual(float(rows["<c>"]["score_raw"]), 0.5)     # (1.0 + 0.0) / 2, -0.4 clipped to 0
        self.assertEqual(rows["<a>"]["dim_edu"], "2.000000")
        self.assertEqual(rows["<a>"]["dim_clean"], "4.000000")

    def test_raw_column_averages_unclipped_and_clips_only_the_stored_score(self):
        rows = {r["doc_id"]: r for r in cs.combine_language("vie", [self.root / "edu", self.root / "clean"], "score_raw")}
        self.assertAlmostEqual(float(rows["<b>"]["score_raw"]), 5.75)
        self.assertAlmostEqual(float(rows["<b>"]["score"]), 5.0)

    def test_different_document_sets_are_rejected(self):
        write(self.root / "other", "vie", {"<a>": 1.0, "<b>": 1.0})
        with self.assertRaises(ValueError):
            cs.combine_language("vie", [self.root / "edu", self.root / "other"], "score")

    def test_output_can_be_ranked_by_prepare_data(self):
        from src.train_gpt2_from_scratch import prepare_data as pdata
        rows = cs.combine_language("vie", [self.root / "edu", self.root / "clean"], "score")
        cs.write_language(self.root / "vie.csv", rows, ["dim_edu", "dim_clean"])
        scores = pdata.load_scores(self.root / "vie.csv")
        pool = [{"doc_id": d, "char_len": 500} for d in scores]
        top = pdata.select_top_score(pool, scores, 2, "vie")
        self.assertEqual([r["doc_id"] for r in top], ["<b>", "<a>"])


class LanguageDiscovery(unittest.TestCase):
    def test_thresholds_and_correlation_files_are_not_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write(folder, "vie", {"<a>": 1.0})
            write(folder, "thai", {"<b>": 1.0})
            (folder / "thresholds.csv").write_text("language,scored\nvie,1\n", encoding="utf-8")
            (folder / "dimension_correlations.csv").write_text("dimension_a,dimension_b,spearman\n", encoding="utf-8")
            (folder / "score_distribution.png").write_bytes(b"x")
            self.assertEqual(cs.language_files(folder), {"vie", "thai"})


class Correlation(unittest.TestCase):
    def test_spearman_perfect_reversed_and_undefined(self):
        self.assertAlmostEqual(cs.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(cs.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(cs.spearman([1, 1, 1], [1, 2, 3]))

    def test_ties_get_average_ranks(self):
        self.assertEqual(cs.rank([1, 2, 2, 3]), [1.0, 2.5, 2.5, 4.0])

    def test_pairs(self):
        pairs = cs.correlations({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 3, 2]})
        self.assertEqual([(p["dimension_a"], p["dimension_b"]) for p in pairs], [("a", "b"), ("a", "c"), ("b", "c")])


if __name__ == "__main__":
    unittest.main()
