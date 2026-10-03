import unittest

from src.train_gpt2_from_scratch import eval_downstream as ed


class CommonPrefixLen(unittest.TestCase):
    def test_identical_prefix(self):
        self.assertEqual(ed.common_prefix_len([1, 2, 3], [1, 2, 3, 4, 5]), 3)

    def test_diverges_immediately(self):
        self.assertEqual(ed.common_prefix_len([1, 2], [9, 2]), 0)

    def test_empty(self):
        self.assertEqual(ed.common_prefix_len([], [1, 2]), 0)


class SampleIndices(unittest.TestCase):
    def test_sorted_unique_and_in_range(self):
        idx = ed.sample_indices(900, 100, seed=42, language="vie")
        self.assertEqual(len(idx), 100)
        self.assertEqual(len(set(idx)), 100)
        self.assertEqual(idx, sorted(idx))
        self.assertTrue(all(0 <= i < 900 for i in idx))

    def test_reproducible_and_language_dependent(self):
        a = ed.sample_indices(900, 100, seed=42, language="vie")
        b = ed.sample_indices(900, 100, seed=42, language="vie")
        c = ed.sample_indices(900, 100, seed=42, language="thai")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_caps_at_population_size(self):
        idx = ed.sample_indices(10, 100, seed=1, language="x")
        self.assertEqual(len(idx), 10)


class AccuracyAndBaselines(unittest.TestCase):
    def test_accuracy(self):
        records = [{"correct": True}, {"correct": False}, {"correct": True}, {"correct": True}]
        self.assertAlmostEqual(ed.accuracy(records), 0.75)

    def test_accuracy_empty_is_nan(self):
        self.assertTrue(ed.accuracy([]) != ed.accuracy([]))  # nan != nan

    def test_majority_baseline(self):
        examples = [{"gold_index": 0}, {"gold_index": 0}, {"gold_index": 1}, {"gold_index": 0}]
        self.assertAlmostEqual(ed.majority_baseline(examples), 0.75)


class XcopaConnectors(unittest.TestCase):
    def test_every_supported_language_has_both_connectors(self):
        for language in ed.XCOPA_LANGUAGES:
            self.assertIn("cause", ed.XCOPA_CONNECTORS[language])
            self.assertIn("effect", ed.XCOPA_CONNECTORS[language])


class BenchmarkLanguageCoverage(unittest.TestCase):
    def test_xcopa_only_has_three_languages(self):
        self.assertEqual(set(ed.XCOPA_LANGUAGES), {"indo", "thai", "vie"})

    def test_belebele_and_sib200_have_all_six(self):
        expected = {"fil", "indo", "khmer", "malay", "thai", "vie"}
        self.assertEqual(set(ed.BELEBELE_LANGUAGES), expected)
        self.assertEqual(set(ed.SIB200_LANGUAGES), expected)


if __name__ == "__main__":
    unittest.main()
