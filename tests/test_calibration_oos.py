import numpy as np
from research.calibration_oos import _clip_normalize, _ece

def test_calibration_probabilities_normalize():
    p=_clip_normalize(np.array([[0.2,0.3,0.5],[0.8,0.1,0.1]],dtype=float))
    assert np.allclose(p.sum(axis=1),1.0)

def test_ece_is_zero_for_perfect_binary_confidence():
    p=np.array([[1.0,0.0],[0.0,1.0]],dtype=float); y=np.array([0,1])
    assert _ece(p,y)==0.0
