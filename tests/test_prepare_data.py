import unittest

from src.train_gpt2_from_scratch import prepare_data as pd_


def rows(n=200):
    return [{"doc_id": f"<doc-{i:04d}>", "char_len": 500 + i} for i in range(n)]


class SelectRandom(unittest.TestCase):
    def test_size_uniqueness_and_membership(self):
        pool = rows()
        chosen = pd_.select_random(pool, 50, 42, "vie")
        ids = [r["doc_id"] for r in chosen]
        self.assertEqual(len(ids), 50)
        self.assertEqual(len(set(ids)), 50)
        self.assertTrue(set(ids) <= {r["doc_id"] for r in pool})

    def test_reproducible_and_independent_of_file_order(self):
        pool = rows()
        a = pd_.select_random(pool, 50, 42, "vie")
        b = pd_.select_random(list(reversed(pool)), 50, 42, "vie")
        self.assertEqual({r["doc_id"] for r in a}, {r["doc_id"] for r in b})

    def test_depends_on_seed_and_language(self):
        pool = rows()
        base = {r["doc_id"] for r in pd_.select_random(pool, 50, 42, "vie")}
        self.assertNotEqual(base, {r["doc_id"] for r in pd_.select_random(pool, 50, 7, "vie")})
        self.assertNotEqual(base, {r["doc_id"] for r in pd_.select_random(pool, 50, 42, "thai")})

    def test_too_few_documents_raises(self):
        with self.assertRaises(ValueError):
            pd_.select_random(rows(10), 11, 42, "vie")

    def test_summary_totals(self):
        pool = rows()
        chosen = pd_.select_random(pool, 20, 1, "vie")
        for r in chosen:
            r["tokens"], r["tokens_with_eos"] = 100, 101
        s = pd_.summarise("vie", pool, chosen)
        self.assertEqual((s["selected_docs"], s["selected_tokens"], s["selected_tokens_with_eos"]), (20, 2000, 2020))


class SelectTopScore(unittest.TestCase):
    def pool(self):
        return [{"doc_id": f"<d{i}>", "char_len": 500 + i} for i in range(6)]

    def scores(self):
        return {"<d0>": 2.0, "<d1>": 5.3, "<d2>": 4.9, "<d3>": 5.3, "<d4>": 1.0, "<d5>": 5.3}

    def test_highest_first_ties_by_doc_id_and_score_recorded(self):
        top = pd_.select_top_score(self.pool(), self.scores(), 4, "vie")
        self.assertEqual([r["doc_id"] for r in top], ["<d1>", "<d3>", "<d5>", "<d2>"])
        self.assertEqual(top[0]["rater_score"], "5.300000")

    def test_ranks_by_unclipped_score(self):
        pool = [{"doc_id": "<a>", "char_len": 500}, {"doc_id": "<b>", "char_len": 500}, {"doc_id": "<c>", "char_len": 500}]
        top = pd_.select_top_score(pool, {"<a>": 5.9, "<b>": 5.1, "<c>": 4.0}, 2, "vie")
        self.assertEqual([r["doc_id"] for r in top], ["<a>", "<b>"])

    def test_input_order_does_not_matter(self):
        a = [r["doc_id"] for r in pd_.select_top_score(self.pool(), self.scores(), 3, "vie")]
        b = [r["doc_id"] for r in pd_.select_top_score(list(reversed(self.pool())), self.scores(), 3, "vie")]
        self.assertEqual(a, b)

    def test_unscored_candidates_and_too_many_documents_raise(self):
        partial = dict(self.scores())
        del partial["<d0>"]
        with self.assertRaises(ValueError):
            pd_.select_top_score(self.pool(), partial, 3, "vie")
        with self.assertRaises(ValueError):
            pd_.select_top_score(self.pool(), self.scores(), 7, "vie")

    def test_summary_and_markdown_include_scores_and_token_ratio(self):
        import argparse
        pool = self.pool()
        top = pd_.select_top_score(pool, self.scores(), 3, "vie")
        for r in top:
            r["tokens"], r["tokens_with_eos"] = 100, 101
        s = pd_.summarise("vie", pool, top, self.scores())
        self.assertAlmostEqual(s["selected_mean_score"], 5.3)
        self.assertAlmostEqual(s["selected_min_score"], 5.3)
        self.assertLess(s["pool_mean_score"], s["selected_mean_score"])
        args = argparse.Namespace(method="top-score", select_by="docs", num_docs=3, scores_dir="x", pool_dir="y",
                                  tokenizer="t", seed=1)
        text = pd_.build_markdown([s], args, {"vie": 202})
        self.assertIn("Ratio to random", text)
        self.assertIn("1.50", text)          # 3 documents x 101 tokens = 303, against a baseline of 202
        self.assertIn("**Total**", text)

    def test_load_scores(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vie.csv"
            path.write_text("doc_id,language,score_raw,score,mmbert_tokens\n<a>,vie,5.5,5.0,10\n<b>,vie,1.25,1.25,7\n", encoding="utf-8")
            self.assertEqual(pd_.load_scores(path), {"<a>": 5.5, "<b>": 1.25})


class OrderRandomAll(unittest.TestCase):
    def test_is_a_full_permutation(self):
        pool = rows(50)
        ordered = pd_.order_random_all(pool, 42, "vie")
        self.assertEqual(len(ordered), 50)
        self.assertEqual({r["doc_id"] for r in ordered}, {r["doc_id"] for r in pool})

    def test_reproducible_independent_of_file_order_seed_and_language_dependent(self):
        pool = rows(50)
        a = [r["doc_id"] for r in pd_.order_random_all(pool, 42, "vie")]
        b = [r["doc_id"] for r in pd_.order_random_all(list(reversed(pool)), 42, "vie")]
        self.assertEqual(a, b)
        c = [r["doc_id"] for r in pd_.order_random_all(pool, 7, "vie")]
        d = [r["doc_id"] for r in pd_.order_random_all(pool, 42, "thai")]
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_not_the_identity_order(self):
        # with 50 items a seeded shuffle landing back on the original order would be a red flag
        pool = rows(50)
        ordered = pd_.order_random_all(pool, 42, "vie")
        self.assertNotEqual([r["doc_id"] for r in ordered], [r["doc_id"] for r in pool])


class RankAllByScore(unittest.TestCase):
    def pool(self):
        return [{"doc_id": f"<d{i}>", "char_len": 500 + i} for i in range(6)]

    def scores(self):
        return {"<d0>": 2.0, "<d1>": 5.3, "<d2>": 4.9, "<d3>": 5.3, "<d4>": 1.0, "<d5>": 5.3}

    def test_ranks_the_whole_pool_best_first_ties_by_doc_id(self):
        ranked = pd_.rank_all_by_score(self.pool(), self.scores(), "vie")
        self.assertEqual([r["doc_id"] for r in ranked], ["<d1>", "<d3>", "<d5>", "<d2>", "<d0>", "<d4>"])

    def test_missing_score_raises(self):
        partial = dict(self.scores())
        del partial["<d0>"]
        with self.assertRaises(ValueError):
            pd_.rank_all_by_score(self.pool(), partial, "vie")

    def test_select_top_score_is_a_prefix_of_rank_all(self):
        ranked = [r["doc_id"] for r in pd_.rank_all_by_score(self.pool(), self.scores(), "vie")]
        top = [r["doc_id"] for r in pd_.select_top_score(self.pool(), self.scores(), 4, "vie")]
        self.assertEqual(top, ranked[:4])


class SelectByTokenBudget(unittest.TestCase):
    def test_grows_by_whole_groups_and_stops_at_first_group_meeting_target(self):
        ordered = [{"doc_id": f"<d{i}>"} for i in range(20)]
        counts = [10] * 20  # 10 tokens each, group_size=5 -> 50 tokens/group
        selected, total = pd_.select_by_token_budget(ordered, counts, target_tokens=120, group_size=5, language="vie")
        # 1 group (50) < 120; 2 groups (100) < 120; 3 groups (150) >= 120 -> keeps 15 docs
        self.assertEqual(len(selected), 15)
        self.assertEqual(total, 150)
        self.assertEqual([r["doc_id"] for r in selected], [f"<d{i}>" for i in range(15)])

    def test_never_undershoots_and_overshoot_is_reported(self):
        ordered = [{"doc_id": f"<d{i}>"} for i in range(10)]
        counts = [7] * 10
        selected, total = pd_.select_by_token_budget(ordered, counts, target_tokens=30, group_size=5, language="vie")
        self.assertGreaterEqual(total, 30)
        self.assertEqual(total, 35)  # one group of 5 (35) is already >= 30; stops there, not at 3 docs (21)

    def test_exact_multiple_of_group_size_does_not_overshoot_unnecessarily(self):
        ordered = [{"doc_id": f"<d{i}>"} for i in range(10)]
        counts = [10] * 10
        selected, total = pd_.select_by_token_budget(ordered, counts, target_tokens=50, group_size=5, language="vie")
        self.assertEqual(len(selected), 5)
        self.assertEqual(total, 50)

    def test_insufficient_pool_raises(self):
        ordered = [{"doc_id": f"<d{i}>"} for i in range(4)]
        counts = [10] * 4
        with self.assertRaises(ValueError):
            pd_.select_by_token_budget(ordered, counts, target_tokens=1000, group_size=5, language="vie")

    def test_last_partial_group_is_handled(self):
        ordered = [{"doc_id": f"<d{i}>"} for i in range(7)]  # not a multiple of group_size
        counts = [10] * 7
        selected, total = pd_.select_by_token_budget(ordered, counts, target_tokens=65, group_size=5, language="vie")
        self.assertEqual(len(selected), 7)  # group 1 (50) < 65; group 2 is only 2 docs (20) -> total 70 >= 65
        self.assertEqual(total, 70)

    def test_group_size_one_behaves_like_a_tight_cutoff(self):
        ordered = [{"doc_id": f"<d{i}>"} for i in range(10)]
        counts = [10] * 10
        selected, total = pd_.select_by_token_budget(ordered, counts, target_tokens=31, group_size=1, language="vie")
        self.assertEqual(len(selected), 4)
        self.assertEqual(total, 40)


class TokenBudgetSummaryAndMarkdown(unittest.TestCase):
    def test_summarise_reports_target_and_overshoot(self):
        pool = [{"doc_id": f"<d{i}>", "char_len": 500} for i in range(5)]
        selected = [{"doc_id": f"<d{i}>", "char_len": 500, "tokens": 100, "tokens_with_eos": 101} for i in range(3)]
        s = pd_.summarise("vie", pool, selected, scores=None, target_tokens=280)
        self.assertEqual(s["selected_tokens_with_eos"], 303)
        self.assertEqual(s["target_tokens"], 280)
        self.assertEqual(s["tokens_over_target"], 23)

    def test_markdown_shows_target_columns_not_baseline_ratio(self):
        import argparse
        s = {"language": "vie", "pool_docs": 100, "selected_docs": 3, "pool_mean_chars": 500.0,
             "selected_mean_chars": 500.0, "selected_median_chars": 500, "selected_tokens": 300,
             "selected_tokens_with_eos": 303, "selected_mean_tokens": 100.0, "target_tokens": 280,
             "tokens_over_target": 23}
        args = argparse.Namespace(method="random", select_by="tokens", target_tokens=280, group_size=5,
                                  scores_dir="x", pool_dir="y", tokenizer="t", seed=42)
        text = pd_.build_markdown([s], args, baseline=None)
        self.assertIn("Target tokens", text)
        self.assertIn("Over target", text)
        self.assertIn("+23", text)
        self.assertNotIn("Ratio to random", text)


if __name__ == "__main__":
    unittest.main()
