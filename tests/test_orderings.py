"""Checks of the ordering regions and Bayesian mean calculation."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "behavior"))
from orderings import TRANSFORM, multivariate_t, niw_marginal, region_masks


def test_chained_reset_false_positive_is_rejected():
    # A chain of adjacent near-equalities is not a group: HL > LL, and HH differs from HL by more than delta.
    cells = np.array([[.105, .095, .110, .080]])  # LL, LH, HL, HH
    ll, lh, hl, hh = cells[0]
    assert ll - hh > .02 and abs(hh - lh) <= .02 and abs(lh - hl) <= .02
    assert not region_masks(cells, .02, .02)["reset"][0]


def test_current_groups_require_every_cross_group_comparison():
    # LL > HH alone does not imply both Low cells exceed both High cells.
    cells = np.array([[.100, .095, .085, .075]])
    assert not region_masks(cells, .02, .02)["current"][0]


def test_unambiguous_examples_select_the_intended_pattern():
    examples = np.array([[.06, .00, .12, .06], [.10, .00, .10, .00], [.10, .00, .00, .00], [.05, .05, .05, .05]])
    masks = region_masks(examples, .02, .02)
    for index, name in enumerate(["reliability", "current", "reset", "flat"]):
        assert masks[name][index]
        assert sum(mask[index] for mask in masks.values()) == 1


def test_translation_invariance_and_orthonormal_contrast_basis():
    rng = np.random.default_rng(74)
    values = rng.normal(size=(1000, 4))
    np.testing.assert_allclose(TRANSFORM @ TRANSFORM.T, np.eye(3))
    np.testing.assert_allclose(TRANSFORM @ np.ones(4), np.zeros(3))
    np.testing.assert_allclose(values @ TRANSFORM.T @ TRANSFORM, values - values.mean(axis=1, keepdims=True), atol=1e-14)
    for key in region_masks(values, .02, .02):
        np.testing.assert_array_equal(region_masks(values, .02, .02)[key], region_masks(values + 100, .02, .02)[key])


def test_gap_conventions_have_different_overlap_properties():
    cells = np.array([[.030, .018, .025, .020]])
    masks = region_masks(cells, .02, 0.)
    assert masks["current"][0] and masks["reset"][0]
    samples = np.random.default_rng(6).normal(0, .1, (10000, 4))
    masks = region_masks(samples, .02, .02)
    assert np.stack(list(masks.values())).sum(axis=0).max() == 1


def test_proper_prior_and_posterior_parameters():
    location, scale, df = niw_marginal(np.empty((0, 3)), .1)
    np.testing.assert_array_equal(location, [0, 0, 0])
    np.testing.assert_allclose(scale, .01 * np.eye(3))
    assert df == 3
    observations = np.array([[.03, -.02, .01], [.07, .02, .03]])
    location, scale, df = niw_marginal(observations, .1)
    np.testing.assert_allclose(location, observations.sum(axis=0) / 3)
    scatter = (observations - observations.mean(axis=0)).T @ (observations - observations.mean(axis=0))
    expected = (.03 * np.eye(3) + scatter + 2 / 3 * np.outer(observations.mean(axis=0), observations.mean(axis=0))) / 15
    np.testing.assert_allclose(scale, expected)
    assert df == 5


def test_t_integration_agrees_with_analytic_marginal_probability():
    rng = np.random.default_rng(50)
    location = np.array([.02, -.01, .03])
    scale = np.array([[.0004, .0001, 0], [.0001, .0006, 0], [0, 0, .0003]])
    samples = multivariate_t(rng, location, scale, 25, 100000)
    expected = stats.t.sf((0 - location[0]) / np.sqrt(scale[0, 0]), 25)
    assert abs(np.mean(samples[:, 0] > 0) - expected) < .005
    np.testing.assert_allclose(np.cov(samples.T), scale * 25 / 23, rtol=.10, atol=.00001)
