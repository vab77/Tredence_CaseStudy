import numpy as np
import pytest
from engine.tensor import Tensor
from engine.gradcheck import numerical_gradient
from nn.losses import cross_entropy
from nn.layers import Tanh

@pytest.mark.parametrize('operation', [lambda a,b:a+b, lambda a,b:a-b, lambda a,b:a*b, lambda a,b:a/b])
@pytest.mark.parametrize('shapes', [((2,3),(2,3)), ((2,3),(3,)), ((2,3),(1,)), ((2,1,3),(1,4,1))])
def test_binary_gradients(operation, shapes):
    rng = np.random.default_rng(10)
    a, b = [rng.uniform(.5,2,size=shape) for shape in shapes]
    x,y = Tensor(a,True), Tensor(b,True)
    operation(x,y).sum().backward()
    ga = numerical_gradient(lambda v: operation(Tensor(v),Tensor(b)).data.sum(),a)
    gb = numerical_gradient(lambda v: operation(Tensor(a),Tensor(v)).data.sum(),b)
    np.testing.assert_allclose(x.grad,ga,rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(y.grad,gb,rtol=1e-5,atol=1e-7)

def test_matmul():
    rng = np.random.default_rng(2)
    a,b = rng.normal(size=(2,3)),rng.normal(size=(3,4))
    seed = rng.normal(size=(2,4))
    x,y=Tensor(a,True),Tensor(b,True)
    (x@y).backward(seed)
    np.testing.assert_allclose(x.grad,numerical_gradient(lambda v:((v@b)*seed).sum(),a),atol=1e-7)
    np.testing.assert_allclose(y.grad,numerical_gradient(lambda v:((a@v)*seed).sum(),b),atol=1e-7)

@pytest.mark.parametrize('operation', ['sum','mean'])
@pytest.mark.parametrize('axis', [None,0,1,-1,(0,2),()])
@pytest.mark.parametrize('keepdims', [False,True])
def test_reductions(operation,axis,keepdims):
    values = np.arange(24,dtype=float).reshape(2,3,4)/10
    x = Tensor(values,True)
    getattr(x,operation)(axis=axis,keepdims=keepdims).sum().backward()
    expected=numerical_gradient(lambda v:getattr(v,operation)(axis=axis,keepdims=keepdims).sum(),values)
    np.testing.assert_allclose(x.grad,expected,atol=1e-7)

@pytest.mark.parametrize('name', ['relu','tanh'])
def test_activations(name):
    values=np.array([[-1.2,-.3],[.2,1.5]])
    x=Tensor(values,True)
    getattr(x,name)().sum().backward()
    expected=numerical_gradient(lambda v:getattr(Tensor(v),name)().data.sum(),values)
    np.testing.assert_allclose(x.grad,expected,atol=1e-7)

def test_tanh_layer_uses_differentiable_tensor_operation():
    values=np.array([[-1.2,-.3],[.2,1.5]])
    x=Tensor(values,True)
    output=Tanh()(x)
    output.sum().backward()
    expected=numerical_gradient(lambda v:Tanh()(Tensor(v)).data.sum(),values)
    np.testing.assert_allclose(output.data,np.tanh(values),atol=1e-12)
    np.testing.assert_allclose(x.grad,expected,atol=1e-7)

def test_cross_entropy():
    values=np.array([[.4,-.2,.8],[1.2,.3,-.1]])
    labels=np.array([1,0])
    x=Tensor(values,True); cross_entropy(x,labels).backward()
    expected=numerical_gradient(lambda v:cross_entropy(v,labels).item(),values)
    np.testing.assert_allclose(x.grad,expected,atol=1e-7)

def test_extreme_logits():
    x=Tensor([[1000,-1000,0],[-1000,1000,0]],True)
    loss=cross_entropy(x,np.array([1,0])); loss.backward()
    assert loss.item()==2000
    assert np.isfinite(x.grad).all()

def test_reverse_operators_and_negation():
    values=np.array([.7,1.3])
    def expression(x): return ((2-x)+(3/x)+(-x)+(2*x)+(2+x)).sum()
    x=Tensor(values,True); expression(x).backward()
    np.testing.assert_allclose(x.grad,numerical_gradient(lambda v:expression(Tensor(v)).item(),values),atol=1e-7)

def test_scalar_reduction():
    x=Tensor(2.,True); x.mean().backward(); assert x.grad==1
