using Boltzmann

const Spin = Boltzmann.IsingSpin
const Binary = Boltzmann.Bernoulli

sigmoid = Boltzmann.logistic

# mf parameters with corresponding default values
KNOWN_PARAMS = Dict(:eps => 1e-8, 
                    :max_iter => 20,
                    :dump => 0.5
               )

#function S(::Type{Ising}, m)
#
#end

# TODO: separating means and probs in Boltzmann, in order
#       to call Boltzmann.means() here

# mean-field visible fixed point equation
function mf_v_fp(rbm::RBM{T,V,Spin}, m_h::Array{T,2}) where {T,V}
  tanh.(rbm.W' * m_h .+ rbm.vbias)
end

function mf_v_fp(rbm::RBM{T,V,Binary}, m_h::Array{T,2}) where {T,V}
  sigmoid(rbm.W' * m_h .+ rbm.vbias)
end

# mean-field hidden fixed point equation
function mf_h_fp(rbm::RBM{T,V,Spin}, m_v::Array{T,2}) where {T,V}
  tanh.(rbm.W * m_v .+ rbm.hbias)
end

function mf_h_fp(rbm::RBM{T,V,Binary}, m_v::Array{T,2}) where {T,V}
  sigmoid(rbm.W * m_v .+ rbm.hbias)
end

# tap visible fixed point equation
function tap_v_fp(rbm::RBM{T,V,Spin}, m_v::Array{T,2}, m_h::Array{T,2};
                             W2::Array{T,2}=abs2.(rbm.W)) where {T,V}
  tanh.(rbm.W' * m_h .+ rbm.vbias - W2' * (1 - abs2.(m_h)) .* m_v)
end

function tap_v_fp(rbm::RBM{T,V,Binary}, m_v::Array{T,2}, m_h::Array{T,2};
                             W2::Array{T,2}=abs2.(rbm.W)) where {T,V}
  sigmoid(rbm.W' * m_h .+ rbm.vbias - W2' * (m_h - abs2.(m_h)) .* (m_v - 0.5))
end

# tap hidden fixed point equation
function tap_h_fp(rbm::RBM{T,V,Spin}, m_v::Array{T,2}, m_h::Array{T,2};
                             W2::Array{T,2}=abs2.(rbm.W)) where {T,V}
  tanh.(rbm.W * m_v .+ rbm.hbias - W2 * (1 - abs2.(m_v)) .* m_h)
end

function tap_h_fp(rbm::RBM{T,V,Binary}, m_v::Array{T,2}, m_h::Array{T,2};
                             W2::Array{T,2}=abs2.(rbm.W)) where {T,V}
  sigmoid(rbm.W * m_v .+ rbm.hbias - W2 * (m_v - abs2.(m_v)) .* (m_h - 0.5))
end

function get_mf_iter_params(ctx::Dict)
  params = Dict()
  for (k,v) in KNOWN_PARAMS
    params[k] = haskey(ctx, k) ? ctx[k] : KNOWN_PARAMS[k] 
  end 

  params
end

function get_fp(rbm::RBM, vis::Array{T,2}, v_fp::Function, h_fp::Function;
              eps=0, max_iter=20, dump=0.5) where T
  hid = Boltzmann.hid_means(rbm, vis)
  m_v = v_fp(rbm, vis, hid)
  m_h = h_fp(rbm, m_v, hid)

  for i=1:max_iter
    println(i)
    new_m_v = (1-dump) * v_fp(rbm, m_v, m_h) + dump * m_v
    new_m_h = (1-dump) * h_fp(rbm, m_v, m_h) + dump * m_h 
    
    if eps > 0 && all(x -> x < eps, abs.(m_v .- new_m_v)) && all(x -> x < eps, abs.(m_h - new_m_h))
      println("Done! ", i)
      break;
    end
    
    m_v = new_m_v
    m_h = new_m_h
  end

  m_v, m_h
end

function nmf_gradient(rbm::RBM, vis::Array{T,2}, ctx::Dict) where {T}
  params = get_mf_iter_params(ctx::Dict)
  m_v, m_h = get_fp(rbm, vis, mf_v_fp, mf_h_fp; params...)
  vis, Boltzmann.hid_means(rbm, vis), m_v, m_h
end

function tap_gradient(rbm::RBM, vis::Array{T,2}, ctx::Dict) where {T}
  params = get_mf_iter_params(ctx::Dict)
  m_v, m_h = get_fp(rbm, vis, tap_v_fp, tap_h_fp; params...)
  vis, Boltzmann.hid_means(rbm, vis), m_v, m_h
end
