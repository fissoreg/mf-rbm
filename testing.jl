using MNIST

include("mf.jl")
include("../MNIST_utils/src/MNIST_utils.jl")

X, y = testdata()

m_v, m_h = fp_iter(rbm, X[:,1:100], tap_v_fp, tap_h_fp; max_iter=1000)

s = samplesToImg(m_v);

