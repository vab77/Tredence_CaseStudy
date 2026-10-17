import numpy as np
from nn.module import Parameter
from optim.adam import Adam

def test_multi_step_reference_updates():
    p=Parameter([1.,-2.]); opt=Adam([p],lr=.03)
    expected=p.data.copy(); m=np.zeros(2); v=np.zeros(2)
    for t,g in enumerate([np.array([.2,-.4]),np.array([-.7,.3]),np.array([0.,.8])],1):
        m=.9*m+.1*g; v=.999*v+.001*g*g
        expected-=.03*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        p.grad=g; opt.step()
        np.testing.assert_allclose(p.data,expected,atol=1e-12)

def test_zero_gradient_momentum_can_move_active_weight():
    p=Parameter([1.]); opt=Adam([p]); p.grad=np.array([1.]); opt.step()
    before=p.data.copy(); p.grad=np.array([0.]); opt.step()
    assert p.data[0] < before[0]
