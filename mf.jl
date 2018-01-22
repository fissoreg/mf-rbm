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

# fixed point iteration
function fp_iter(rbm::RBM, vis::Array{T,2}, v_fp::Function, h_fp::Function;
              eps=0, max_iter=20, dump=0.5) where T
  hid = Boltzmann.hid_means(rbm, vis)
  m_v = v_fp(rbm, vis, hid)
  m_h = h_fp(rbm, m_v, hid)
iter = 0
  for i=1:max_iter
    new_m_v = (1-dump) * v_fp(rbm, m_v, m_h) + dump * m_v
    new_m_h = (1-dump) * h_fp(rbm, m_v, m_h) + dump * m_h 
    if eps > 0 && all(x -> x < eps, abs.(m_v .- new_m_v)) && all(x -> x < eps, abs.(m_h - new_m_h))
      break;
    end
   iter  = i 
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

function merge_fps(fps; acc=[], delta=10*KNOWN_PARAMS[:eps])
  # to deal with empty acc
  get_acc = x -> length(acc) > 0 ? hcat(acc, x) : x

  n = size(fps, 2)
  if n == 1
    return get_acc(fps)
  end

  head = fps[:, 1]
  tail = fps[:, 2:n]
  diff = sum(abs(tail .- head), 1)

  if all(x -> x > delta, diff)
    new_acc = get_acc(head)
  else
    new_acc = acc
  end
  
  merge_fps(tail, acc=new_acc, delta=delta)
end

function get_fps(rbm::RBM; fps=[], init=[], params=Dict())
  if length(init) == 0
    init = rand([-1.0, 1.0], (length(rbm.vbias),100))
  end

  params = get_mf_iter_params(params)
  delta = 100 * params[:eps]

  new_fps, _ = fp_iter(rbm, init, tap_v_fp, tap_h_fp;
                    params...)

  if length(fps) == 0
    return merge_fps(new_fps; delta=delta)
  end

  old_fps, _ = fp_iter(rbm, fps, tap_v_fp, tap_h_fp;
                    params...)

  merge_fps(hcat(old_fps, new_fps); delta=delta)
end

# Legacy - To refactor ############################################

# condensed fixed points initialization
# TODO: generalize to many fixed points! - Kind of done I think...
function cfp_init(mva, cond_dir)
  # number of condensed modes
  ncm = 2^sum(cond_dir)
  println(sum(cond_dir))
  ncm = ncm > 1000 ? 1000 : ncm
  println("Condensing...",ncm) 
  cond_matrix = zeros(size(mva,1), ncm)
  n, m = size(cond_matrix)
  counter = 0

  for i=1:n
    if cond_dir[i] == 0
      for k=1:ncm
        cond_matrix[i,k] = mva[i]
      end
      continue
    end
    
    up = false
    counter += 1

    for j=1:m
      cond_matrix[i,j] = up ? 1 : -1
      if j % 2^(counter-1) == 0
        up = ! up
      end
    end
  end
  
  cond_matrix
end

function condensed_fps(rbm::RBM; fps=[], init=[], params=Dict())
  X = params[:X]
  samples = X[:,rand(1:size(X,2), 100)]
  params = get_mf_iter_params(params)
  cond_eps = 10 * params[:eps]
  scaling = sqrt(*(size(rbm.W)...))
  U,s,V = svd(rbm.W)

  if length(init) == 0
    init = 1e-3 * rand(length(rbm.vbias), 1)
  end

  #rbm0 = deepcopy(rbm)
  #rbm0.vbias = zeros(size(rbm.vbias))
  #rbm0.hbias = zeros(size(rbm.hbias))
  #mv0, mh0 = fp_iter(rbm0, init, tap_v_fp, tap_h_fp;
  #                   params...)

  if length(fps) == 0
    fps = samples[:,1:2] #mv0
  end

  # TODO: write functions for the changes of basis!
  #mva = 1/scaling * V' * mv0

  #condensed_dir = map(x -> x > 1e-3, mva)
  ## condensed fixed points initialization
  #cfpi = cfp_init(mva, condensed_dir)
  #mv = scaling * V * mva
  new_fps, _ = fp_iter(rbm, samples, tap_v_fp, tap_h_fp;
                    params...)
  old_fps, _ = fp_iter(rbm, fps, tap_v_fp, tap_h_fp;
                    params...)

  merge_fps(hcat(old_fps, new_fps); delta=cond_eps)
end
