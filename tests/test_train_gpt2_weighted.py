import unittest

import torch

from src.train_gpt2_from_scratch import train_gpt2_weighted as w

EOS, PAD = 2, 2


def weighted_docs(lang_weighted_sizes):
    """Fake (encoded, weight) documents: ids from 10 upward, each ending with EOS."""
    out, nxt = {}, 10
    for lang, sizes in lang_weighted_sizes.items():
        out[lang] = []
        for n, weight in sizes:
            out[lang].append((list(range(nxt, nxt + n - 1)) + [EOS], weight))
            nxt += n
    return out


class ScoreToWeight(unittest.TestCase):
    def test_matches_guide_mapping(self):
        self.assertAlmostEqual(w.score_to_weight(0, 0.5, 0.2), 0.5)
        self.assertAlmostEqual(w.score_to_weight(2.5, 0.5, 0.2), 1.0)
        self.assertAlmostEqual(w.score_to_weight(5, 0.5, 0.2), 1.5)


class PackWeightedStreams(unittest.TestCase):
    def setUp(self):
        self.encoded = weighted_docs({"a": [(5, 2.0), (7, 0.5), (4, 1.0)], "b": [(6, 1.5), (6, 1.5)]})
        self.blocks = w.pack_weighted_streams(self.encoded, ["a", "b"], seq_len=8, pad_id=PAD, seed=42)

    def test_shapes_and_language_tags(self):
        self.assertEqual(self.blocks["input_ids"].shape, (4, 8))
        self.assertEqual(self.blocks["lang"].tolist(), [0, 0, 1, 1])
        self.assertEqual(self.blocks["weight"].shape, self.blocks["input_ids"].shape)

    def test_every_real_token_carries_its_own_documents_weight(self):
        # every document here is internally one weight, so every real (non-pad) token's weight
        # must come from its own document's value, not get mixed up with a neighbor's.
        allowed = {2.0, 0.5, 1.0, 1.5}
        mask, weight = self.blocks["attention_mask"], self.blocks["weight"]
        real_weights = weight[mask == 1].tolist()
        self.assertTrue(set(real_weights) <= allowed)

    def test_padding_positions_get_zero_weight(self):
        mask, weight = self.blocks["attention_mask"], self.blocks["weight"]
        self.assertTrue(bool((weight[mask == 0] == 0).all()))

    def test_deterministic_and_seed_dependent(self):
        again = w.pack_weighted_streams(self.encoded, ["a", "b"], 8, PAD, 42)
        other = w.pack_weighted_streams(self.encoded, ["a", "b"], 8, PAD, 7)
        self.assertTrue(torch.equal(self.blocks["weight"], again["weight"]))
        self.assertFalse(torch.equal(self.blocks["weight"], other["weight"]))

    def test_mean_token_weight_stat(self):
        # language "b": two 6-token docs, weight 1.5 each -> every real token has weight 1.5
        self.assertAlmostEqual(self.blocks["stats"]["b"]["mean_token_weight"], 1.5)


class NormalizeWeightsPerLanguage(unittest.TestCase):
    def test_rescales_to_token_weighted_mean_one(self):
        encoded = {"a": [([1, 2], 3.0), ([3, 4, 5, 6], 1.0)]}  # tokens 2 & 4, weights 3 & 1
        m = w.normalize_weights_per_language(encoded)
        self.assertAlmostEqual(m["a"], (2 * 3.0 + 4 * 1.0) / 6)
        total_tokens = sum(len(ids) for ids, _ in encoded["a"])
        total_weight = sum(len(ids) * weight for ids, weight in encoded["a"])
        self.assertAlmostEqual(total_weight / total_tokens, 1.0)

    def test_relative_weight_order_is_preserved(self):
        encoded = {"a": [([1, 2], 3.0), ([3, 4, 5, 6], 1.0)]}
        w.normalize_weights_per_language(encoded)
        (_, w_short), (_, w_long) = encoded["a"]
        self.assertGreater(w_short, w_long)


class WeightedBlockLosses(unittest.TestCase):
    def test_reduces_to_ordinary_loss_when_all_weights_equal(self):
        from src.train_gpt2_from_scratch import train_gpt2 as base
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        config = GPT2Config(vocab_size=50, n_positions=8, n_embd=16, n_layer=1, n_head=2, bos_token_id=EOS,
                            eos_token_id=EOS, pad_token_id=PAD, tie_word_embeddings=True, use_cache=False)
        model = GPT2LMHeadModel(config)
        ids = torch.randint(0, 50, (3, 8))
        mask = torch.ones(3, 8, dtype=torch.long)
        mask[1, -2:] = 0
        ones_weight = torch.ones(3, 8)

        plain_loss_sum, plain_count = base.block_losses(model, ids, mask, "cpu", False)
        weighted_loss_sum, weight_sum = w.weighted_block_losses(model, ids, mask, ones_weight, "cpu", False)
        torch.testing.assert_close(weighted_loss_sum, plain_loss_sum)
        torch.testing.assert_close(weight_sum, plain_count.float())

    def test_weight_scales_contribution_as_expected(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        config = GPT2Config(vocab_size=50, n_positions=8, n_embd=16, n_layer=1, n_head=2, bos_token_id=EOS,
                            eos_token_id=EOS, pad_token_id=PAD, tie_word_embeddings=True, use_cache=False)
        model = GPT2LMHeadModel(config)
        ids = torch.randint(0, 50, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.long)
        double_weight = torch.full((1, 8), 2.0)
        loss_sum_1, weight_sum_1 = w.weighted_block_losses(model, ids, mask, torch.ones(1, 8), "cpu", False)
        loss_sum_2, weight_sum_2 = w.weighted_block_losses(model, ids, mask, double_weight, "cpu", False)
        torch.testing.assert_close(loss_sum_2, loss_sum_1 * 2)
        torch.testing.assert_close(weight_sum_2, weight_sum_1 * 2)


if __name__ == "__main__":
    unittest.main()
