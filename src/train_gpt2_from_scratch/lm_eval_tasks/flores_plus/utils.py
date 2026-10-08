# FLORES+ (openlanguagedata/flores_plus, gated: accept the terms on the Hub once): English -> target translation.
# Sentences of the two languages are aligned by `id` within a split; dev serves the demonstrations, devtest is
# scored. Built in memory (custom_dataset) instead of the pre-written JSONL pair files of
# docs/03b_additional_benchmarks.md. Filipino is fil_Latn here (not tgl_Latn as in SIB-200/Belebele).
import datasets


def load_flores_pairs(target, source="eng_Latn", **kwargs):
    """DatasetDict {dev, devtest} of {id, topic, source, target}; extra kwargs (task metadata) are ignored. `topic` (the
    same in every language) lets eval_lm_harness.py draw a fixed per-topic subset of devtest."""
    out = {}
    for split in ("dev", "devtest"):
        src = datasets.load_dataset("openlanguagedata/flores_plus", source, split=split)
        tgt = datasets.load_dataset("openlanguagedata/flores_plus", target, split=split)
        by_id = {row["id"]: row["text"] for row in src}
        rows = [{"id": row["id"], "topic": row["topic"], "source": by_id[row["id"]], "target": row["text"]}
                for row in tgt if row["id"] in by_id]
        if len(rows) != len(tgt):
            raise ValueError(f"FLORES+ {split}: only {len(rows)} of {len(tgt)} {target} sentences have an English pair")
        out[split] = datasets.Dataset.from_list(rows)
    return datasets.DatasetDict(out)
