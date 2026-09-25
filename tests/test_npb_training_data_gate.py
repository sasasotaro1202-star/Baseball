from pathlib import Path
import json

import pytest

from baseball_backtest import BaseballBacktest


def test_npb_training_data_gate_exists():
    text = Path('baseball_backtest.py').read_text()
    assert 'NPB training-data gate failed' in text
    assert 'training_rows < 200' in text


@pytest.mark.parametrize('category,expected',[
    ('regular', True),
    ('interleague', True),
    ('excluded', False),
])
def test_training_categories_are_explicit(category, expected):
    bt = BaseballBacktest()
    assert (category in bt.training_set) is expected