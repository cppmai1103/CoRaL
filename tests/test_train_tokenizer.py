import unittest
import unicodedata

from src.train_gpt2_from_scratch import train_tokenizer as t

SAMPLES = {
    "vie": "Hiện có hai loại màn hình khi lựa chọn mua tablet hay smartphone.",
    "thai": "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์ ผู้คนพูดกันทั่วประเทศ",
    "khmer": "ភាសាខ្មែរជាភាសាផ្លូវការរបស់ប្រទេសកម្ពុជា",
    "indo": "Bahasa Indonesia adalah bahasa resmi Republik Indonesia, 17 Agustus 1945.",
    "fil": "Ang wikang Filipino ay ang pambansang wika ng Pilipinas.",
    "malay": "Bahasa Melayu ialah bahasa kebangsaan Malaysia.",
}


class Pretokenizer(unittest.TestCase):
    def setUp(self):
        self.pre = t.build_tokenizer().pre_tokenizer

    def pieces(self, text):
        return [p for p, _ in self.pre.pre_tokenize_str(text)]

    def test_thai_marks_stay_with_their_letter(self):
        # ที่ = consonant + vowel mark + tone mark: one pre-token, not split at the marks
        self.assertEqual(len(self.pieces("ที่")), 1)

    def test_khmer_subscript_stays_with_its_letter(self):
        self.assertEqual(len(self.pieces("ខ្មែរ")), 1)

    def test_words_split_on_spaces(self):
        self.assertEqual(len(self.pieces("xin chào bạn")), 3)


class TrainedTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = [s for s in SAMPLES.values()] * 50
        cls.tok = t.train_tokenizer(texts, vocab_size=400, min_frequency=1)

    def test_vocab_size_and_eos(self):
        self.assertLessEqual(len(self.tok), 400)
        self.assertEqual(self.tok.eos_token, t.EOS_TOKEN)
        self.assertEqual(self.tok.pad_token_id, self.tok.eos_token_id)
        self.assertEqual(self.tok.convert_tokens_to_ids(t.EOS_TOKEN), 0)

    def test_no_special_tokens_added_on_encode(self):
        ids = self.tok("xin chào", add_special_tokens=False)["input_ids"]
        self.assertEqual(ids, self.tok("xin chào")["input_ids"])
        self.assertNotIn(self.tok.eos_token_id, ids)

    def test_round_trip_every_language_and_unseen_text(self):
        for text in list(SAMPLES.values()) + ["日本語 émoji 🙂 \t tabs\n\nnew  lines"]:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            self.assertEqual(self.tok.decode(ids), text)
            self.assertNotIn(self.tok.unk_token_id, ids) if self.tok.unk_token_id is not None else None

    def test_nfc_normalization(self):
        decomposed = unicodedata.normalize("NFD", "Việt")
        self.assertEqual(self.tok(decomposed)["input_ids"], self.tok("Việt")["input_ids"])

    def test_measure(self):
        m = t.measure(self.tok, list(SAMPLES.values()))
        self.assertEqual(m["documents"], 6)
        self.assertEqual(m["round_trip_exact"], 1.0)
        self.assertGreater(m["chars_per_token"], 1.0)


class BalancedSample(unittest.TestCase):
    def test_equal_characters_per_language_from_smallest(self):
        texts = {"a": {f"a{i}": "x" * 10 for i in range(100)}, "b": {f"b{i}": "y" * 10 for i in range(5)}}
        chosen, stats = t.balanced_sample(texts, 0, seed=1)
        self.assertEqual(stats["b"]["characters"], 50)
        self.assertEqual(stats["a"]["characters"], 50)
        self.assertEqual(len(chosen["a"]), 5)

    def test_explicit_budget_and_seeded(self):
        texts = {"a": {f"a{i}": f"{i:02d}" * 5 for i in range(100)}}  # 10 chars each
        c1, s1 = t.balanced_sample(texts, 30, seed=1)
        c2, _ = t.balanced_sample(texts, 30, seed=1)
        self.assertEqual(c1, c2)
        self.assertEqual(s1["a"]["documents"], 3)

    def test_interleave_keeps_every_text(self):
        out = list(t.interleave({"a": ["1", "2", "3"], "b": ["x"]}))
        self.assertEqual(out, ["1", "x", "2", "3"])


if __name__ == "__main__":
    unittest.main()
