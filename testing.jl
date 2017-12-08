using MNIST

include("mf.jl")
include("../MNIST_utils/src/MNIST_utils.jl")

X, y = testdata()

m_v, m_h = get_fp(rbm, X[:,1:100], tap_v_fp, tap_h_fp)

s = samplesToImg(m_v);

