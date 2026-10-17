import numpy as np
from nn.layers import make_mlp
from optim.adam import Adam
from prune.sparse import export_sparse,sparse_forward,dense_forward,weight_macs
from train.artifacts import save_checkpoint,load_checkpoint

def test_dense_sparse_equivalence_and_bias_only_neuron():
    model=make_mlp([3,4,2], hidden_activations=['tanh']); opt=Adam(model.parameters())
    for p in model.parameters():
        if p.prunable:
            mask=np.ones(p.shape,bool); mask[:,0]=False; opt.set_mask(p,mask)
    x=np.random.default_rng(4).normal(size=(5,3)); exported=export_sparse(model)
    np.testing.assert_allclose(sparse_forward(exported,x),model(x).data,atol=1e-12)
    np.testing.assert_allclose(dense_forward(model,x),model(x).data,atol=1e-12)
    assert weight_macs(exported)==13

def test_checkpoint_roundtrip(tmp_path):
    model=make_mlp([3,4,2], hidden_activations=['tanh']); path=tmp_path/'model.npz'
    save_checkpoint(path,model,[3,4,2]); loaded=load_checkpoint(path)
    x=np.ones((2,3)); np.testing.assert_array_equal(loaded(x).data,model(x).data)
