import unittest

from src.train_gpt2_from_scratch import split_corpus as sc


def make_docs():
    docs = []
    # one huge site, several mid sites, many singletons
    for i in range(60):
        docs.append({"doc_id": f"big-{i}", "domain": "big.com"})
    for s in range(6):
        for i in range(4):
            docs.append({"doc_id": f"mid{s}-{i}", "domain": f"mid{s}.com"})
    for i in range(40):
        docs.append({"doc_id": f"one-{i}", "domain": f"one{i}.com"})
    return docs  # 124 documents


SIZES = {"train": 100, "validation": 12, "test": 12}


class SplitDocuments(unittest.TestCase):
    def setUp(self):
        self.docs = make_docs()
        self.assignment = sc.split_documents(self.docs, SIZES, seed=42, max_eval_group=10)

    def test_exact_sizes(self):
        for split, n in SIZES.items():
            self.assertEqual(sum(v == split for v in self.assignment.values()), n, split)

    def test_every_site_stays_in_one_split(self):
        by_domain = {}
        for d in self.docs:
            by_domain.setdefault(d["domain"], set()).add(self.assignment[d["doc_id"]])
        self.assertTrue(all(len(v) == 1 for v in by_domain.values()))

    def test_large_site_is_train_only(self):
        self.assertTrue(all(self.assignment[f"big-{i}"] == "train" for i in range(60)))

    def test_deterministic_and_seed_dependent(self):
        again = sc.split_documents(self.docs, SIZES, seed=42, max_eval_group=10)
        other = sc.split_documents(self.docs, SIZES, seed=7, max_eval_group=10)
        self.assertEqual(self.assignment, again)
        self.assertNotEqual(self.assignment, other)

    def test_size_mismatch_and_unfillable_raise(self):
        with self.assertRaises(ValueError):
            sc.split_documents(self.docs, {"train": 1, "validation": 1, "test": 1}, 42, 10)
        with self.assertRaises(ValueError):  # only 40 + 24 eligible documents, 70 requested for eval
            sc.split_documents(self.docs, {"train": 54, "validation": 35, "test": 35}, 42, 10)


class DomainOf(unittest.TestCase):
    def test_strips_www_and_handles_missing_url(self):
        self.assertEqual(sc.domain_of("https://www.Example.com/a/b", "x"), "example.com")
        self.assertEqual(sc.domain_of("http://id.wikipedia.org/wiki/A", "x"), "id.wikipedia.org")
        self.assertEqual(sc.domain_of("", "doc1"), "no-url:doc1")


if __name__ == "__main__":
    unittest.main()
