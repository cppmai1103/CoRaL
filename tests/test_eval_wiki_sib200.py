import math
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

    def test_bits_per_byte_is_total_nats_over_bytes(self):
        records = [{"nll_sum": 2 * math.log(2), "valid_target_tokens": 1, "text_bytes": 1},
                   {"nll_sum": 6 * math.log(2), "valid_target_tokens": 3, "text_bytes": 3}]
        self.assertAlmostEqual(ews.group_loss(records)["bits_per_byte"], 8 / 4)

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
    def test_all_seven_languages_present_for_both_datasets(self):
        expected = {"burmese", "fil", "indo", "khmer", "malay", "thai", "vie"}
        self.assertEqual(set(ews.WIKIPEDIA_LANGUAGES), expected)
        self.assertEqual(set(ews.SIB200_LANGUAGES), expected)


class ScoreDocument(unittest.TestCase):
    """The batched cross-entropy must count every content token exactly once, like the per-token loop it replaced."""

    def setUp(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=16, n_embd=16, n_layer=1, n_head=2)).eval()

        class Tokenizer:
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1 + ord(c) % 49 for c in text]}

        self.tokenizer = Tokenizer()

    def reference(self, text, seq_len, stride):
        import torch
        import torch.nn.functional as F
        ids = [0] + self.tokenizer(text)["input_ids"]
        nll, count, scored_until, begin = 0.0, 0, 0, 0
        while True:
            end = min(begin + seq_len, len(ids))
            window = ids[begin:end]
            with torch.no_grad():
                log_probs = F.log_softmax(self.model(torch.tensor([window])).logits[0].float(), dim=-1)
            for rel_t in range(1, len(window)):
                if begin + rel_t > scored_until:
                    nll += -log_probs[rel_t - 1, window[rel_t]].item()
                    count += 1
            scored_until = end - 1
            if end == len(ids):
                return nll, count
            begin += stride

    def test_matches_per_token_loop(self):
        for text, seq_len, stride in [("short", 16, 8), ("a much longer text than one window" * 2, 16, 8),
                                      ("uneven stride over a long document!", 16, 5)]:
            nll, n = ews.score_document(self.model, self.tokenizer, text, "cpu", False, seq_len, stride, 10 ** 9)
            ref_nll, ref_n = self.reference(text, seq_len, stride)
            self.assertEqual(n, len(text))
            self.assertEqual(n, ref_n)
            self.assertAlmostEqual(nll, ref_nll, places=3)


class Annotated(unittest.TestCase):
    def test_high_quality_is_top_fraction_per_language_by_avg5(self):
        import csv
        import tempfile
        dims = ["educational_value", "reasoning", "professionalism", "cleanliness", "cultural_nuances"]
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "document_table.csv"
            with table.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["doc_id", "language", "text"] + dims)
                w.writeheader()
                for lang in ("vie", "thai"):
                    for i, score in enumerate([1, 4, 2, 5]):
                        w.writerow({"doc_id": f"{lang}{i}", "language": lang, "text": "x", **{d: score for d in dims}})
            examples = ews.load_annotated(table, ["vie"], ["avg5", "cleanliness"], 0.5)
        self.assertEqual({e["example_id"] for e in examples}, {"vie0", "vie1", "vie2", "vie3"})
        for by in ("avg5", "cleanliness"):
            self.assertEqual({e["example_id"] for e in examples if by in e["high_quality"]}, {"vie1", "vie3"})
        self.assertEqual({e["topic"] for e in examples},
                         {"high_quality_avg5+high_quality_cleanliness", "other"})

    def test_dataset_names(self):
        self.assertEqual(ews.hq_dataset("avg5"), "annotated_hq")
        self.assertEqual(ews.hq_dataset("cleanliness"), "annotated_hq_cleanliness")


if __name__ == "__main__":
    unittest.main()
