"""Offline tests for Labs 1 and 2."""

import subprocess
import sys
import unittest


def run(script):

    return subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120
    )


class TestLabs(unittest.TestCase):

    def test_lab1_first_merge_and_new_word(self):

        out = run("lab1_bpe_from_scratch.py")

        self.assertEqual(out.returncode, 0, out.stderr)

        self.assertIn("Merge 1:", out.stdout)

        self.assertIn("lowest", out.stdout)

    def test_lab2_trains_three_and_prints_summary(self):

        out = run("lab2_train_three_tokenizers.py")

        self.assertEqual(out.returncode, 0, out.stderr)

        for word in (
            "BPE",
            "WordPiece",
            "SentencePiece",
            "SUMMARY"
        ):

            self.assertIn(word, out.stdout)


if __name__ == "__main__":

    unittest.main()