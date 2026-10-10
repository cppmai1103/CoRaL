import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.train_gpt2_from_scratch import train_gpt2 as t
from src.train_gpt2_from_scratch import train_gpt2_weighted as w

SEQ, BLOCKS, BATCH = 16, 24, 4  # 6 steps per epoch


def tiny_model():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=SEQ, n_embd=16, n_layer=1, n_head=2,
                                      resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1))


def blocks(weighted: bool):
    g = torch.Generator().manual_seed(1)
    b = {"input_ids": torch.randint(0, 50, (BLOCKS, SEQ), generator=g),
         "attention_mask": torch.ones(BLOCKS, SEQ, dtype=torch.long),
         "lang": torch.zeros(BLOCKS, dtype=torch.long), "stats": {}}
    if weighted:
        b["weight"] = torch.rand(BLOCKS, SEQ, generator=g) + 0.5
    return b


def args(**kw):
    return SimpleNamespace(batch_size=BATCH, micro_batch_size=2, epochs=2, lr=1e-2, weight_decay=0.0, warmup_ratio=0.1,
                           grad_clip=1.0, seed=3, seq_len=SEQ, evals_per_epoch=1, eval_at_start=False,
                           eval_batch_size=4, log_every=1, **kw)


class ResumeMatchesUninterrupted(unittest.TestCase):
    def run_case(self, trainer, weighted):
        train_blocks, val_blocks = blocks(weighted), blocks(False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def run(out, **kw):
                out.mkdir()
                model = tiny_model()
                log = t.CsvLog(out / "eval_log.csv", ["step", "epoch", "tokens_consumed", "split", "language", "loss",
                                                      "perplexity", "target_tokens"])
                torch.manual_seed(5)  # dropout RNG at the start of training
                result = trainer(model, train_blocks, val_blocks, ["x"], args(**kw), "cpu", False, out, log)
                log.close()
                return model, result

            straight, straight_result = run(tmp / "straight")
            ckpt = tmp / "ckpt"
            # a run saving every 4 steps leaves its step-8 checkpoint (12 steps in all), as if it was killed after it
            run(tmp / "saving", checkpoint_dir=ckpt, save_every=4, checkpoint_fingerprint={"a": "1"})
            self.assertEqual(t.Checkpointer.saved_step(ckpt), 8)
            resumed, resumed_result = run(tmp / "resumed", checkpoint_dir=ckpt, save_every=4, resume=True,
                                          checkpoint_fingerprint={"a": "1"})
            for (name, a), (_, b) in zip(straight.named_parameters(), resumed.named_parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-6, msg=name)
            self.assertEqual(resumed_result["tokens_consumed"], straight_result["tokens_consumed"])
            with self.assertRaises(SystemExit):  # other settings: refused
                run(tmp / "other", checkpoint_dir=ckpt, save_every=4, resume=True, checkpoint_fingerprint={"a": "2"})

    def test_plain_loss(self):
        self.run_case(t.train, weighted=False)

    def test_weighted_loss(self):
        self.run_case(w.train, weighted=True)


class CsvLogKeepUptoStep(unittest.TestCase):
    def test_drops_rows_after_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            log = t.CsvLog(path, ["step", "loss"])
            for step in (1, 2, 3):
                log.write({"step": step, "loss": step / 10})
            log.close()
            log = t.CsvLog(path, ["step", "loss"], keep_upto_step=2)
            log.write({"step": 3, "loss": 0.9})
            log.close()
            self.assertEqual(path.read_text().split(), ["step,loss", "1,0.1", "2,0.2", "3,0.9"])


if __name__ == "__main__":
    unittest.main()
