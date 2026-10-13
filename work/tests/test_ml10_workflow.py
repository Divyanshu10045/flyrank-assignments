"""Offline tests for the ML-10 action-playbook workflow.

No network, no warehouse, no token: every test is stdlib-only so it can run in CI
next to the pipeline smoke test. Covers the three failure classes that broke the
notebook in Colab (repo-root resolution, missing output directories, committed
receipts) plus the scientific invariants the assignment must not lose.

Run from the repo root:   python work/tests/test_ml10_workflow.py
or via unittest:          python -m unittest discover -s work/tests
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]                      # work/tests -> work -> repo root
sys.path.insert(0, str(REPO / "work" / "scripts"))

import ml10_bootstrap as mb  # noqa: E402

W07 = REPO / "work" / "notebooks" / "w07_action_playbook.ipynb"


def _w07_cell_sources():
    nb = json.loads(W07.read_text(encoding="utf-8"))
    out = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        out.append(src if isinstance(src, str) else "".join(src))
    return out


class BootstrapPathTests(unittest.TestCase):
    def test_find_repo_root_from_nested_dir(self):
        self.assertEqual(mb.find_repo_root(REPO / "work" / "notebooks"), REPO)

    def test_find_repo_root_needs_git_and_notebooks(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "work" / "notebooks").mkdir(parents=True)   # notebooks but no .git
            self.assertIsNone(mb.find_repo_root(d))
            (Path(d) / ".git").mkdir()                              # now both markers exist
            self.assertEqual(mb.find_repo_root(d), Path(d).resolve())

    def test_ensure_output_dirs_creates_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            made = mb.ensure_output_dirs(d)
            for rel in mb.OUTPUT_DIRS:
                self.assertTrue((Path(d) / rel).is_dir(), rel)
            self.assertEqual(made, mb.ensure_output_dirs(d))        # second call is a no-op

    def test_missing_receipts_are_inputs_not_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sorted(mb.missing_receipts(d)), sorted(mb.REQUIRED_RECEIPTS))
            mb.ensure_output_dirs(d)                                # creating outputs must not satisfy inputs
            self.assertEqual(sorted(mb.missing_receipts(d)), sorted(mb.REQUIRED_RECEIPTS))
            for rel in mb.REQUIRED_RECEIPTS:
                (Path(d) / rel).write_text("{}", encoding="utf-8")
            self.assertEqual(mb.missing_receipts(d), [])


class CommittedArtifactTests(unittest.TestCase):
    def test_ml07_and_ml09_receipts_are_committed(self):
        self.assertEqual(mb.missing_receipts(REPO), [],
                         "ML-07 baseline_metrics.json and ML-09 w06_validation_metrics.json "
                         "must stay committed: the playbook inherits its numbers from them")

    def test_requirements_cover_notebook_runtime_deps(self):
        reqs = (REPO / "requirements.txt").read_text(encoding="utf-8").lower()
        for pkg in ("pandas", "numpy", "scipy", "matplotlib",
                    "huggingface_hub", "fastparquet", "python-dotenv"):
            self.assertIn(pkg, reqs, "%s must be pinned in requirements.txt" % pkg)


class W07InvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = _w07_cell_sources()
        cls.code = "\n".join(cls.sources)

    def test_uses_tested_bootstrap_helper(self):
        self.assertIn("import ml10_bootstrap as mb", self.code)
        self.assertIn("mb.ensure_output_dirs", self.code)
        self.assertIn("mb.missing_receipts", self.code)

    def test_output_dir_created_before_every_write(self):
        self.assertIn("mb.ensure_output_dirs", self.code)          # cell 01, before any write
        for src in self.sources:
            if "QOUT.to_csv(QUEUE" in src:
                self.assertLess(src.index("QUEUE.parent.mkdir"), src.index("QOUT.to_csv(QUEUE"))
            if "RECEIPT.write_text" in src:
                self.assertLess(src.index("RECEIPT.parent.mkdir"), src.index("RECEIPT.write_text"))

    def test_single_deployment_fit(self):
        self.assertIn("assert len(N_FITS) == 1", self.code)

    def test_contract_features_unchanged(self):
        self.assertIn("FEATS5 = [\"log_impr_prior\", \"ctr_prior_pct\", \"wtd_pos_prior\", "
                      "\"active_days_prior\", \"content_age_days\"]", self.code)

    def test_probe_feature_is_read_but_not_a_model_input(self):
        feats5_line = next(l for l in self.code.splitlines() if l.startswith("FEATS5 = "))
        self.assertNotIn("trend_pct_win", feats5_line)
        self.assertIn("trend_pct_win", self.code)                  # still read for the sliding flag

    def test_penalty_is_ml09_nested_selection(self):
        self.assertIn("L2_FINAL = 1.0", self.code)
        self.assertIn("assert L2_FINAL == BA[\"l2_nested\"]", self.code)

    def test_frame_counts_and_base_rate_asserted(self):
        for needle in ('assert len(P) == 67279',
                       'assert int(P["is_material_decline"].sum()) == 7445',
                       'assert P["client_hash_id"].nunique() == 24',
                       'assert float(P["is_material_decline"].mean()) == 0.11065860075209204'):
            self.assertIn(needle, self.code)

    def test_windows_unchanged(self):
        self.assertIn('FEATURE_START, FEATURE_END = "2026-01-01", "2026-02-28"', self.code)
        self.assertIn('LABEL_START,   LABEL_END   = "2026-03-01", "2026-03-31"', self.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
