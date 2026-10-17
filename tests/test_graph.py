import numpy as np
import pytest
from engine.tensor import Tensor

def test_shared_parent_accumulation():
    x=Tensor([2.,3.],True); (x*x+x).sum().backward()
    np.testing.assert_allclose(x.grad,[5,7])

def test_branched_graph_order():
    x=Tensor(2.,True); a=x*x; b=a*3; c=a*b; c.backward()
    assert x.grad==96

def test_fresh_graphs_accumulate_leaf_gradients():
    x=Tensor(2.,True); (x*3).backward(); (x*4).backward()
    assert x.grad==7

def test_repeated_backward_rejected():
    x=Tensor(2.,True); y=x*x; y.backward()
    with pytest.raises(RuntimeError):y.backward()

def test_constant_and_seed_validation():
    with pytest.raises(ValueError): Tensor(2.).backward()
    with pytest.raises(ValueError): (Tensor([2.,3.],True)*2).backward()

def test_deep_graph():
    x=Tensor(1.,True); y=x
    for _ in range(1500):y=y+1
    y.backward(); assert x.grad==1
