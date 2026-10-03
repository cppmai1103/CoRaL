import csv
import tempfile
import unittest
from pathlib import Path

from src.train_gpt2_from_scratch import score_pool as sp


def write_scores(folder: Path, language: str, values):
    with (folder / f"{language}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(sp.OUT_FIELDS)
        for i, v in enumerate(values):
            w.writerow([f"<{language}{i}>", language, f"{v:.6f}", f"{min(5, max(0, v)):.6f}", 100])


class Threshold(unittest.TestCase):
    def test_kth_highest_is_the_lowest_kept_score(self):
        scores = [1.0, 5.4, 2.5, 4.0, 3.0]
        self.assertEqual(sp.top_k_threshold(scores, 1), 5.4)
        self.assertEqual(sp.top_k_threshold(scores, 3), 3.0)
        self.assertEqual(sp.top_k_threshold(scores, 5), 1.0)

    def test_none_when_fewer_than_k_scored_or_k_invalid(self):
        self.assertIsNone(sp.top_k_threshold([1.0, 2.0], 3))
        self.assertIsNone(sp.top_k_threshold([1.0, 2.0], 0))

    def test_threshold_matches_the_documents_prepare_data_would_keep(self):
        scores = [0.1 * i for i in range(100)]
        t = sp.top_k_threshold(scores, 10)
        self.assertEqual(sum(s >= t for s in scores), 10)

    def test_row_fields(self):
        row = sp.threshold_row("vie", [1.0, 2.0, 3.0, 4.0, 5.5], 2)
        self.assertEqual(row["threshold"], 4.0)
        self.assertAlmostEqual(row["share_kept"], 0.4)
        self.assertAlmostEqual(row["share_above_5"], 0.2)
        self.assertEqual(row["median"], 3.0)
        self.assertEqual(sp.threshold_row("vie", [1.0], 5)["threshold"], "")


class Plot(unittest.TestCase):
    def test_writes_png_and_thresholds_for_every_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_scores(folder, "vie", [i / 10 for i in range(60)])
            write_scores(folder, "thai", [i / 20 for i in range(40)])
            sp.draw_and_report(folder, ["vie", "thai"], 20)
            self.assertGreater((folder / "score_distribution.png").stat().st_size, 5000)
            with (folder / "thresholds.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["language"] for r in rows], ["vie", "thai"])
        self.assertEqual(float(rows[0]["threshold"]), 4.0)   # 20th highest of 0.0..5.9 -> 4.0
        self.assertEqual(float(rows[1]["threshold"]), 1.0)   # 20th highest of 0.0..1.95 step .05 -> 1.0

    def test_language_with_too_few_documents_gets_no_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_scores(folder, "vie", [1.0, 2.0, 3.0])
            sp.draw_and_report(folder, ["vie"], 10)
            with (folder / "thresholds.csv").open(encoding="utf-8") as f:
                self.assertEqual(next(csv.DictReader(f))["threshold"], "")


if __name__ == "__main__":
    unittest.main()
