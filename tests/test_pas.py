import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.personas import get_pas_sub_dims


def test_pas_sub_dims_engineering():
    dims = get_pas_sub_dims("engineering")
    assert len(dims) == 4
    assert "tone_appropriateness"  in dims
    assert "structural_compliance" in dims
    assert "audience_fit"          in dims
    assert "constraint_adherence"  in dims


def test_pas_sub_dims_marketing():
    dims = get_pas_sub_dims("marketing")
    assert len(dims) == 4


def test_pas_sub_dims_invalid():
    with pytest.raises(KeyError):
        get_pas_sub_dims("unknown_persona")
