import unittest
from pathlib import Path

from src.train_gpt2_from_scratch import eval_wiki_sib200 as ews


class EligibleWikipediaArticle(unittest.TestCase):
    def test_rejects_empty(self):
        self.assertFalse(ews.is_eligible_wikipedia_article(""))

    def test_rejects_too_short(self):
        self.assertFalse(ews.is_eligible_wikipedia_article("short text", min_chars=200))

    def test_rejects_html_artifacts(self):
        text = '<div id="x">' + "word " * 100 + "</div>"
        self.assertFalse(ews.is_eligible_wikipedia_article(text))

    def test_rejects_list_like(self):
        text = "\n".join(f"item {i}" for i in range(20))
        self.assertFalse(ews.is_eligible_wikipedia_article(text))

    def test_accepts_normal_prose(self):
        text = "This is a normal article with real sentences. " * 10
        self.assertTrue(ews.is_eligible_wikipedia_article(text))

    def test_min_chars_is_configurable(self):
        text = "x" * 150
        self.assertFalse(ews.is_eligible_wikipedia_article(text, min_chars=200))
        self.assertTrue(ews.is_eligible_wikipedia_article(text, min_chars=100))


class TextHash(unittest.TestCase):
    def test_matches_extract_corpus_convention(self):
        from src.train_gpt2_from_scratch.extract_corpus import text_hash as extract_text_hash
        self.assertEqual(ews.text_hash("hello world"), extract_text_hash("hello world"))

    def test_strips_before_hashing(self):
        self.assertEqual(ews.text_hash("  hello  "), ews.text_hash("hello"))


class GroupLoss(unittest.TestCase):
    def test_sums_before_dividing_not_average_of_averages(self):
        # two "documents": one short/high-loss, one long/low-loss -- sum-then-divide must weight
        # by token count, not treat both documents as equally important.
        records = [{"nll_sum": 10.0, "valid_target_tokens": 2}, {"nll_sum": 10.0, "valid_target_tokens": 98}]
        g = ews.group_loss(records)
        self.assertEqual(g["valid_target_tokens"], 100)
        self.assertAlmostEqual(g["loss"], 20.0 / 100)
        naive_average_of_per_doc_losses = (5.0 + 10.0 / 98) / 2
        self.assertNotAlmostEqual(g["loss"], naive_average_of_per_doc_losses)

    def test_empty_is_nan(self):
        g = ews.group_loss([])
        self.assertNotEqual(g["loss"], g["loss"])  # nan != nan
        self.assertEqual(g["n_documents"], 0)


class TrainingSeedOf(unittest.TestCase):
    def test_extracts_trailing_seed(self):
        self.assertEqual(ews.training_seed_of(Path("runs/random_20M_ep1_seed42")), "42")
        self.assertEqual(ews.training_seed_of(Path("runs/random_10M_seed43_ep1_seed43")), "43")

    def test_unknown_when_no_seed_suffix(self):
        self.assertEqual(ews.training_seed_of(Path("runs/some_weird_name")), "unknown")


class LanguageCoverage(unittest.TestCase):
    def test_all_six_languages_present_for_both_datasets(self):
        expected = {"fil", "indo", "khmer", "malay", "thai", "vie"}
        self.assertEqual(set(ews.WIKIPEDIA_LANGUAGES), expected)
        self.assertEqual(set(ews.SIB200_LANGUAGES), expected)


if __name__ == "__main__":
    unittest.main()
