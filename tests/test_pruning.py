import numpy as np
from nn.layers import make_mlp
from optim.adam import Adam
from prune.controller import PruningController
from prune.schedule import target_sparsity
from prune.scores import magnitude, update_saliency

def test_schedule_endpoints():
    values=[target_sparsity(t,2,8,.9) for t in range(11)]
    assert values[0]==0 and values[-1]==.9
    assert all(a<=b for a,b in zip(values,values[1:]))

def test_exact_budget_and_deterministic_ties():
    model=make_mlp([3,4,2]); opt=Adam(model.parameters()); controller=PruningController()
    params=[p for p in model.parameters() if p.prunable]
    scores={p:np.ones_like(p.data) for p in params}
    assert controller.update(model,opt,scores,.75)==5
    mask=np.concatenate([p.mask.ravel() for p in params])
    np.testing.assert_array_equal(np.flatnonzero(mask),np.arange(5))
    assert controller.update(model,opt,scores,.9)==2
    assert controller.update(model,opt,scores,.5)==2  # no automatic regrowth
    assert all(np.all(p.data[~p.mask]==0) for p in params)

def test_scores():
    w=np.array([-2.,3.]); g=np.array([.5,-2.])
    np.testing.assert_array_equal(magnitude(w),[2.,3.])
    np.testing.assert_allclose(update_saliency(np.zeros(2),w,g,beta=.5),[.5,3.])
