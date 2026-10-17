import numpy as np
from nn.layers import make_mlp
from optim.adam import Adam
from nn.losses import cross_entropy
from train.loop import evaluate

def test_network_learns_separable_task():
    rng=np.random.default_rng(1)
    x=np.r_[rng.normal(-1,.2,(20,2)),rng.normal(1,.2,(20,2))]
    y=np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int)]
    model=make_mlp([2,8,2],seed=3); opt=Adam(model.parameters(),lr=.03)
    before=evaluate(model,x,y)['loss']
    for _ in range(60):
        opt.zero_grad(); loss=cross_entropy(model(x),y); loss.backward(); opt.step()
    after=evaluate(model,x,y)
    assert after['loss'] < before*.1 and after['accuracy']==1.0
