"""Unit tests for python/glopart_recomb.py (pure, no grid/xrootd access).

Mirrors the style of tests/test_toptag_wp_processor.py. Tiny synthetic NumPy /
awkward fixtures only. The §9 test list from the plan, adapted to the log-score
transform (the heads are NOT a softmax, so the original "sum==1" test is replaced
by a non-negative/finite check, and the transform test proves no NaN on P>1).
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np

sys.path.append(os.path.join(os.getcwd(), "python"))
import glopart_recomb as gr  # noqa: E402


def _auc(s_sig, s_bkg):
    """AUC = P(s_sig > s_bkg), rank-based (no sklearn)."""
    s_sig = np.asarray(s_sig)
    s_bkg = np.asarray(s_bkg)
    allv = np.concatenate([s_sig, s_bkg])
    ranks = allv.argsort().argsort().astype(float) + 1
    r_sig = ranks[: len(s_sig)].sum()
    n1, n2 = len(s_sig), len(s_bkg)
    return (r_sig - n1 * (n1 + 1) / 2) / (n1 * n2)


class TransformTest(unittest.TestCase):
    def test_logscore_handles_values_above_one(self):
        # Xqq-like head exceeding 1.0 must NOT produce NaN/inf under logscore.
        heads = {h: np.array([0.0, 0.5, 1.4248]) for h in gr.RAW_HEADS}
        X = gr.build_features(heads, transform="logscore")
        self.assertEqual(X.shape, (3, gr.N_FEAT))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_clipped_logit_is_finite_but_clips(self):
        heads = {h: np.array([0.0, 1.4248]) for h in gr.RAW_HEADS}
        X = gr.build_features(heads, transform="logit")
        self.assertTrue(np.all(np.isfinite(X)))  # clipping prevents inf/nan
        # both extreme inputs are clipped to the same finite rail magnitude
        self.assertAlmostEqual(X[0, 0], -X[1, 0], places=6)

    def test_logratio_qcd_column_is_zero(self):
        heads = {h: np.linspace(0.1, 0.9, 5) for h in gr.RAW_HEADS}
        X = gr.build_features(heads, transform="logratio")
        qcd_col = gr.RAW_HEADS.index("QCD")
        np.testing.assert_allclose(X[:, qcd_col], 0.0, atol=1e-12)

    def test_assert_heads_sane(self):
        ok = {h: np.array([0.0, 0.5, 1.4]) for h in gr.RAW_HEADS}
        gr.assert_heads_sane(ok)  # no raise; non-negative even though >1 allowed
        bad = dict(ok)
        bad["QCD"] = np.array([0.0, -0.5, 1.0])
        with self.assertRaises(ValueError):
            gr.assert_heads_sane(bad)
        nanny = dict(ok)
        nanny["Xqq"] = np.array([0.0, np.nan, 1.0])
        with self.assertRaises(ValueError):
            gr.assert_heads_sane(nanny)


class BaselineTest(unittest.TestCase):
    def test_baseline_matches_tscore(self):
        import awkward as ak
        import toptag_wp_processor as twp

        rng = np.random.default_rng(0)
        topqq = rng.uniform(0, 1, 50)
        topq = rng.uniform(0, 1, 50)
        qcd = rng.uniform(0, 1, 50)
        fj = ak.Array({
            "globalParT3_TopbWqq": topqq,
            "globalParT3_TopbWq": topq,
            "globalParT3_QCD": qcd,
        })
        ref = ak.to_numpy(twp._topvsqcd_globalParT3(fj))
        got = gr.baseline_topvsqcd({"TopbWqq": topqq, "TopbWq": topq, "QCD": qcd})
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


class MomentTest(unittest.TestCase):
    def test_moment_accumulation_merges(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1000, gr.N_FEAT))
        w = rng.uniform(0.5, 2.0, size=1000)
        full = gr.moment_vec(X, w)
        # split into 3 chunks and add the packed vectors
        merged = gr.empty_moment_vec()
        for sl in (slice(0, 300), slice(300, 700), slice(700, 1000)):
            merged = merged + gr.moment_vec(X[sl], w[sl])
        np.testing.assert_allclose(merged, full, rtol=1e-10, atol=1e-10)

    def test_unpack_shapes(self):
        vec = gr.moment_vec(np.ones((5, gr.N_FEAT)))
        n, S, M = gr.unpack_moments(vec)
        self.assertEqual(n, 5.0)
        self.assertEqual(S.shape, (gr.N_FEAT,))
        self.assertEqual(M.shape, (gr.N_FEAT, gr.N_FEAT))


class LDATest(unittest.TestCase):
    def test_lda_recovers_separable(self):
        rng = np.random.default_rng(2)
        n = 200_000
        d = gr.N_FEAT
        # random shared covariance and a mean offset
        A = rng.normal(size=(d, d))
        Sigma = A @ A.T / d + np.eye(d)
        L = np.linalg.cholesky(Sigma)
        dmu = rng.normal(size=d)
        Xs = (rng.normal(size=(n, d)) @ L.T) + dmu
        Xb = (rng.normal(size=(n, d)) @ L.T)
        w, b = gr.moments_to_lda(gr.moment_vec(Xs), gr.moment_vec(Xb), ridge=1e-6)
        w_true = np.linalg.solve(Sigma, dmu)
        cos = np.dot(w, w_true) / (np.linalg.norm(w) * np.linalg.norm(w_true))
        self.assertGreater(cos, 0.99)
        auc = _auc(gr.apply_score(Xs, w, b), gr.apply_score(Xb, w, b))
        self.assertGreater(auc, 0.9)

    def test_identical_classes_no_separation(self):
        rng = np.random.default_rng(3)
        n, d = 50_000, gr.N_FEAT
        Xs = rng.normal(size=(n, d))
        Xb = rng.normal(size=(n, d))
        w, b = gr.moments_to_lda(gr.moment_vec(Xs), gr.moment_vec(Xb), ridge=1e-4)
        auc = _auc(gr.apply_score(Xs, w, b), gr.apply_score(Xb, w, b))
        self.assertGreater(auc, 0.45)
        self.assertLess(auc, 0.55)


class PerPtTest(unittest.TestCase):
    def test_pt_bin_index_edges(self):
        edges = np.array([400.0, 500.0, 600.0, 800.0, 1200.0, 3000.0])
        pts = np.array([400.0, 499.9, 500.0, 1199.0, 1200.0, 5000.0, 399.0])
        idx = gr.pt_bin_index(pts, edges)
        # 400->0, 499.9->0, 500->1, 1199->3, 1200->4, overflow->4, underflow->0
        np.testing.assert_array_equal(idx, [0, 0, 1, 3, 4, 4, 0])

    def test_apply_per_pt_routes_to_own_bin(self):
        edges = np.array([400.0, 500.0, 600.0])  # 2 bins
        # bin 0 keys on feature 0, bin 1 keys on feature 1
        wpb = {0: (np.array([1.0, 0, 0, 0, 0, 0, 0]), 0.0),
               1: (np.array([0, 1.0, 0, 0, 0, 0, 0]), 0.0)}
        X = np.zeros((2, gr.N_FEAT))
        X[0, 0] = 5.0   # low-pt jet, feature0
        X[1, 1] = 7.0   # high-pt jet, feature1
        pts = np.array([450.0, 550.0])
        s = gr.apply_per_pt(X, pts, edges, wpb)
        np.testing.assert_allclose(s, [5.0, 7.0])


class MetricsTest(unittest.TestCase):
    def _hist(self, samples, edges):
        c, _ = np.histogram(samples, bins=edges)
        return c.astype(float)

    def test_fixed_mistag_inversion(self):
        rng = np.random.default_rng(4)
        edges = np.linspace(0, 1, 1001)
        bkg = rng.uniform(0, 1, 2_000_000)  # flat -> eff above t is (1-t)
        counts = self._hist(bkg, edges)
        thr = gr.threshold_for_target_eff(counts, edges, 0.005)
        self.assertAlmostEqual(thr, 0.995, places=2)
        eff = gr.eff_above_threshold(counts, edges, thr)
        self.assertAlmostEqual(eff, 0.005, places=3)

    def test_sigeff_at_fixed_mistag_separable(self):
        rng = np.random.default_rng(5)
        edges = np.linspace(0, 1, 1001)
        bkg = np.clip(rng.normal(0.3, 0.1, 500_000), 0, 1)
        sig = np.clip(rng.normal(0.7, 0.1, 500_000), 0, 1)
        sc, bc = self._hist(sig, edges), self._hist(bkg, edges)
        eff, thr = gr.sigeff_at_fixed_mistag(sc, bc, edges, 0.005)
        self.assertGreater(eff, 0.5)  # well separated -> high sig eff at 0.5% mistag

    def test_decorrelation_flat_vs_correlated(self):
        # 2D counts: axis = mass slice, disc axis. Build a mass-FLAT score and a
        # mass-CORRELATED score; flatness() must distinguish them.
        rng = np.random.default_rng(6)
        edges = np.linspace(0, 1, 201)
        n_axis = 10
        flat_mistag, corr_mistag = [], []
        flat2d, corr2d = [], []
        thr = 0.8
        for i in range(n_axis):
            disc_flat = rng.uniform(0, 1, 20000)                     # independent of slice
            disc_corr = np.clip(rng.uniform(0, 1, 20000) + 0.03 * i, 0, 1)  # drifts with slice
            flat2d.append(self._hist(disc_flat, edges))
            corr2d.append(self._hist(disc_corr, edges))
        flat2d, corr2d = np.array(flat2d), np.array(corr2d)
        x = np.arange(n_axis)
        mt_flat = gr.mistag_vs_axis(flat2d, edges, thr)
        mt_corr = gr.mistag_vs_axis(corr2d, edges, thr)
        f_flat = gr.flatness(x, mt_flat)
        f_corr = gr.flatness(x, mt_corr)
        self.assertLess(abs(f_flat["slope"]), abs(f_corr["slope"]))


class PersistenceTest(unittest.TestCase):
    def test_weights_roundtrip(self):
        rng = np.random.default_rng(7)
        edges = [400.0, 600.0, 3000.0]
        wpb = {0: (rng.normal(size=gr.N_FEAT), 0.5),
               1: (rng.normal(size=gr.N_FEAT), -0.3)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.json")
            gr.save_weights(path, edges, wpb, transform="logscore",
                            metadata={"iov": "2024"})
            data = gr.load_weights(path)
        self.assertEqual(data["transform"], "logscore")
        self.assertEqual(data["raw_heads"], gr.RAW_HEADS)
        for pb in wpb:
            np.testing.assert_allclose(data["weights_per_bin"][pb][0], wpb[pb][0])
            self.assertAlmostEqual(data["weights_per_bin"][pb][1], wpb[pb][1])
        # scores reproduce bit-for-bit
        X = rng.normal(size=(20, gr.N_FEAT))
        s1 = gr.apply_score(X, *wpb[0])
        s2 = gr.apply_score(X, *data["weights_per_bin"][0])
        np.testing.assert_array_equal(s1, s2)


if __name__ == "__main__":
    unittest.main()
