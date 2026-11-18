"""sft_noise: deterministic per-presentation noise at a fixed SNR (DECISIONS 24)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sft_noise as sn  # noqa: E402

WAV = "stimuli/T01_b3.wav"


def test_seed_is_stable_and_slot_specific():
    assert sn.seed_for("study", 1, 2, "s1") == sn.seed_for("study", 1, 2, "s1")
    assert sn.seed_for("study", 1, 2, "s1") != sn.seed_for("study", 1, 2, "s2")
    assert sn.seed_for("study", 1, 2, "s1") != sn.seed_for("probe", 1, 2, 1)
    assert 0 <= sn.seed_for("x") < 2 ** 63


def test_realised_snr_and_level():
    for db in (-12.0, -6.0, 0.0, 6.0):
        assert abs(sn.realised_snr(WAV, db, 7) - db) < 0.05
    sr, y = sn.mix(WAV, -6.0, 7)
    assert abs(np.sqrt(np.mean(y ** 2)) - sn.sa.OUTPUT_RMS) < 1e-9     # loudness is not a cue
    assert len(y) / sr < 1.0


def test_same_seed_same_bytes_different_seed_different_noise(tmp_path):
    a = sn.mix(WAV, -6.0, 11)[1]
    assert np.array_equal(a, sn.mix(WAV, -6.0, 11)[1])
    assert not np.array_equal(a, sn.mix(WAV, -6.0, 12)[1])
    p1 = sn.rebuild(WAV, -6.0, 11, str(tmp_path / "a.wav"))
    p2 = sn.NoiseMixer(str(tmp_path / "out"), -6.0).render(WAV, 11)
    assert open(p1, "rb").read() == open(p2, "rb").read()
    assert os.path.basename(p2) == "11_T01_b3.wav"


def test_validate_runs():
    sn.validate()
