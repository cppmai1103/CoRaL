import math
import tempfile
import unittest
from pathlib import Path

from src.train_gpt2_from_scratch import eval_test_set as e


def result(losses_tokens):
    per = {l: {"loss": loss, "perplexity": math.exp(loss), "target_tokens": n} for l, (loss, n) in losses_tokens.items()}
    macro = sum(v["loss"] for v in per.values()) / len(per)
    nats = sum(v["loss"] * v["target_tokens"] for v in per.values())
    total = sum(v["target_tokens"] for v in per.values())
    return {"per_language": per, "macro": {"loss": macro, "perplexity": math.exp(macro)},
            "token_weighted": {"loss": nats / total, "perplexity": math.exp(nats / total), "target_tokens": total}}


class BitsPerByte(unittest.TestCase):
    def test_formula(self):
        # 100 tokens at ln 2 nats each = 100 bits over 50 bytes = 2 bits/byte
        self.assertAlmostEqual(e.bits_per_byte(math.log(2), 100, 50), 2.0)

    def test_independent_of_tokenization(self):
        # same text (and the same total nats) split into 2x the tokens at half the loss -> same bits/byte
        self.assertAlmostEqual(e.bits_per_byte(4.0, 100, 300), e.bits_per_byte(2.0, 200, 300))

    def test_add_bits_per_byte_macro_and_overall(self):
        r = e.add_bits_per_byte(result({"a": (math.log(2), 100), "b": (math.log(2), 300)}), {"a": 100, "b": 100})
        self.assertAlmostEqual(r["per_language"]["a"]["bits_per_byte"], 1.0)
        self.assertAlmostEqual(r["per_language"]["b"]["bits_per_byte"], 3.0)
        self.assertAlmostEqual(r["macro"]["bits_per_byte"], 2.0)
        self.assertAlmostEqual(r["token_weighted"]["bits_per_byte"], 400 / 200)
        self.assertEqual(r["token_weighted"]["bytes"], 200)


class Comparison(unittest.TestCase):
    def test_flags_different_tokenizer_and_deltas(self):
        b = e.add_bits_per_byte(result({"a": (4.0, 100)}), {"a": 400})
        r = e.add_bits_per_byte(result({"a": (3.5, 150)}), {"a": 400})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cmp.md"
            e.write_comparison(path, "test", [{"name": "base", "tokenizer": "T1", "result": b},
                                              {"name": "new", "tokenizer": "T2", "result": r}])
            text = path.read_text()
        self.assertIn("| new | `T2` | **no** |", text)
        self.assertIn("-0.5000", text)  # loss delta of run "new"


if __name__ == "__main__":
    unittest.main()
