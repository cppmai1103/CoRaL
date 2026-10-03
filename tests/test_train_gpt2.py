import math
import unittest

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from src.train_gpt2_from_scratch import train_gpt2 as t

EOS, PAD = 2, 2  # the pilot reuses EOS as the padding id


def docs(lang_sizes):
    """Fake encoded documents: ids from 10 upward, each ending with EOS."""
    out, nxt = {}, 10
    for lang, sizes in lang_sizes.items():
        out[lang] = []
        for n in sizes:
            out[lang].append(list(range(nxt, nxt + n - 1)) + [EOS])
            nxt += n
    return out


class PackStreams(unittest.TestCase):
    def setUp(self):
        self.encoded = docs({"a": [5, 7, 4], "b": [6, 6]})
        self.blocks = t.pack_streams(self.encoded, ["a", "b"], seq_len=8, pad_id=PAD, seed=42)

    def test_shapes_and_language_tags(self):
        self.assertEqual(self.blocks["input_ids"].shape, (4, 8))  # a: 16 tokens -> 2 blocks, b: 12 -> 2 blocks
        self.assertEqual(self.blocks["lang"].tolist(), [0, 0, 1, 1])

    def test_no_real_token_is_lost_or_added(self):
        for lang, li in (("a", 0), ("b", 1)):
            sel = self.blocks["lang"] == li
            real = self.blocks["input_ids"][sel][self.blocks["attention_mask"][sel] == 1].tolist()
            self.assertEqual(sorted(real), sorted(t_ for d in self.encoded[lang] for t_ in d))
            self.assertEqual(self.blocks["stats"][lang]["tokens"], len(real))

    def test_padding_is_only_at_the_end_of_the_last_block(self):
        mask = self.blocks["attention_mask"]
        self.assertTrue(bool((mask[:, 1:] <= mask[:, :-1]).all()))
        self.assertEqual(mask.sum().item(), 16 + 12)
        self.assertEqual(self.blocks["stats"]["b"]["padding_tokens"], 4)

    def test_genuine_eos_is_real_and_only_mask_marks_padding(self):
        ids, mask = self.blocks["input_ids"], self.blocks["attention_mask"]
        self.assertEqual(int(((ids == EOS) & (mask == 1)).sum()), 5)  # one EOS per document
        self.assertGreater(int(((ids == EOS) & (mask == 0)).sum()), 0)  # padding uses the same id

    def test_deterministic_and_seed_dependent(self):
        again = t.pack_streams(self.encoded, ["a", "b"], 8, PAD, 42)
        other = t.pack_streams(self.encoded, ["a", "b"], 8, PAD, 7)
        self.assertTrue(torch.equal(self.blocks["input_ids"], again["input_ids"]))
        self.assertFalse(torch.equal(self.blocks["input_ids"], other["input_ids"]))

    def test_target_count_excludes_first_token_and_padding(self):
        self.assertEqual(t.target_count(self.blocks["attention_mask"]), 28 - 4)


class LossAccounting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=50, n_positions=8, n_embd=16, n_layer=1, n_head=2, use_cache=False)
        self.model = GPT2LMHeadModel(cfg).eval()
        enc = {"a": [[3, 4, 5, 6, 7, 2], [8, 9, 2]], "b": [[11, 12, 13, 2], [20, 21, 22, 23, 24, 25, 2]]}
        self.blocks = t.pack_streams(enc, ["a", "b"], 8, PAD, 1)

    def test_evaluate_matches_manual_cross_entropy(self):
        result = t.evaluate(self.model, self.blocks, ["a", "b"], batch_size=2, device="cpu", use_bf16=False)
        ids, mask, lang = self.blocks["input_ids"], self.blocks["attention_mask"], self.blocks["lang"]
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        for li, name in enumerate(["a", "b"]):
            total, count = 0.0, 0
            for b in (lang == li).nonzero().flatten().tolist():
                n = int(mask[b].sum())
                for pos in range(1, n):
                    total += F.cross_entropy(logits[b, pos - 1].unsqueeze(0), ids[b, pos].unsqueeze(0)).item()
                    count += 1
            self.assertEqual(result["per_language"][name]["target_tokens"], count)
            self.assertAlmostEqual(result["per_language"][name]["loss"], total / count, places=4)
            self.assertAlmostEqual(result["per_language"][name]["perplexity"], math.exp(total / count), places=3)

    def test_macro_and_token_weighted_averages(self):
        r = t.evaluate(self.model, self.blocks, ["a", "b"], batch_size=1, device="cpu", use_bf16=False)
        pl = r["per_language"]
        self.assertAlmostEqual(r["macro"]["loss"], (pl["a"]["loss"] + pl["b"]["loss"]) / 2, places=6)
        weighted = (pl["a"]["loss"] * pl["a"]["target_tokens"] + pl["b"]["loss"] * pl["b"]["target_tokens"]) / \
                   (pl["a"]["target_tokens"] + pl["b"]["target_tokens"])
        self.assertAlmostEqual(r["token_weighted"]["loss"], weighted, places=5)
        self.assertAlmostEqual(r["macro"]["perplexity"], math.exp(r["macro"]["loss"]), places=5)

    def test_batch_size_does_not_change_results(self):
        a = t.evaluate(self.model, self.blocks, ["a", "b"], 1, "cpu", False)
        b = t.evaluate(self.model, self.blocks, ["a", "b"], 4, "cpu", False)
        self.assertAlmostEqual(a["macro"]["loss"], b["macro"]["loss"], places=5)

    def test_padding_content_cannot_change_the_loss(self):
        ids = self.blocks["input_ids"].clone()
        mask = self.blocks["attention_mask"]
        ids[mask == 0] = 9  # change every padding id
        s1, c1 = t.block_losses(self.model, self.blocks["input_ids"], mask, "cpu", False)
        s2, c2 = t.block_losses(self.model, ids, mask, "cpu", False)
        self.assertTrue(torch.allclose(s1, s2, atol=1e-5))
        self.assertTrue(torch.equal(c1, c2))


class DataSummary(unittest.TestCase):
    def test_table_lists_every_language_and_correct_totals(self):
        tr = t.pack_streams(docs({"a": [5, 7, 4], "b": [6, 6]}), ["a", "b"], 8, PAD, 1)
        va = t.pack_streams(docs({"a": [3], "b": [4, 4]}), ["a", "b"], 8, PAD, 1)
        te = t.pack_streams(docs({"a": [2, 2], "b": [5]}), ["a", "b"], 8, PAD, 1)
        text = t.format_data_summary(tr, va, te, ["a", "b"])
        self.assertIn("a", text)
        total_line = text.splitlines()[-1]
        self.assertIn("28", total_line)   # 16 + 12 training tokens
        self.assertIn("11", total_line)   # 3 + 8 validation tokens
        self.assertIn("9", total_line)    # 4 + 5 test tokens
        self.assertEqual(len(text.splitlines()), 2 + 2 + 2)  # header, rule, two languages, rule, total


class Progress(unittest.TestCase):
    def test_progress_wraps_and_can_be_disabled(self):
        t.PROGRESS = False
        try:
            self.assertEqual(list(t.progress(range(3))), [0, 1, 2])
        finally:
            t.PROGRESS = True


if __name__ == "__main__":
    unittest.main()
