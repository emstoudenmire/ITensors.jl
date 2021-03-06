
"""
    TruncSVD{N}

ITensor factorization type for a truncated singular-value 
decomposition, returned by `svd`.
"""
struct TruncSVD{N1,N2}
  U::ITensor{N1}
  S::ITensor{2}
  V::ITensor{N2}
  spec::Spectrum
  u::Index
  v::Index
end

# iteration for destructuring into components `U,S,V,spec,u,v = S`
Base.iterate(S::TruncSVD) = (S.U, Val(:S))
Base.iterate(S::TruncSVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::TruncSVD, ::Val{:V}) = (S.V, Val(:spec))
Base.iterate(S::TruncSVD, ::Val{:spec}) = (S.spec, Val(:u))
Base.iterate(S::TruncSVD, ::Val{:u}) = (S.u, Val(:v))
Base.iterate(S::TruncSVD, ::Val{:v}) = (S.v, Val(:done))
Base.iterate(S::TruncSVD, ::Val{:done}) = nothing

@doc """
    svd(A::ITensor, inds::Index...; <keyword arguments>)

Singular value decomposition (SVD) of an ITensor `A`, computed
by treating the "left indices" provided collectively
as a row index, and the remaining "right indices" as a
column index (matricization of a tensor).

The first three return arguments are `U`, `S`, and `V`, such that
`A ≈ U * S * V`.

Whether or not the SVD performs a trunction depends on the keyword
arguments provided. 

# Arguments
- `maxdim::Int`: the maximum number of singular values to keep.
- `mindim::Int`: the minimum number of singular values to keep.
- `cutoff::Float64`: set the desired truncation error of the SVD, by default defined as the sum of the squares of the smallest singular values.
- `lefttags::String = "Link,u"`: set the tags of the Index shared by `U` and `S`.
- `righttags::String = "Link,v"`: set the tags of the Index shared by `S` and `V`.
- `alg::String = "recursive"`. Options:
  - `"recursive"` - ITensor's custom svd. Very reliable, but may be slow if high precision is needed. To get an `svd` of a matrix `A`, an eigendecomposition of ``A^{\\dagger} A`` is used to compute `U` and then a `qr` of ``A^{\\dagger} U`` is used to compute `V`. This is performed recursively to compute small singular values.
  - `"divide_and_conquer"` - A divide-and-conquer algorithm. LAPACK's gesdd.
  - `"qr_iteration"` - Typically slower but more accurate than `"divide_and_conquer"`. LAPACK's gesvd.
- `use_absolute_cutoff::Bool = false`: set if all probability weights below the `cutoff` value should be discarded, rather than the sum of discarded weights.
- `use_relative_cutoff::Bool = true`: set if the singular values should be normalized for the sake of truncation.

See also: [`factorize`](@ref)
"""
function LinearAlgebra.svd(A::ITensor,
                           Linds...;
                           kwargs...)
  utags::TagSet = get(kwargs, :lefttags, get(kwargs, :utags, "Link,u"))
  vtags::TagSet = get(kwargs, :righttags, get(kwargs, :vtags, "Link,v"))

  # Keyword argument deprecations
  #if haskey(kwargs, :utags) || haskey(kwargs, :vtags)
  #  @warn "Keyword arguments `utags` and `vtags` are deprecated in favor of `leftags` and `righttags`."
  #end

  Lis = commoninds(A, IndexSet(Linds...))
  Ris = uniqueinds(A, Lis)

  if length(Lis) == 0 || length(Ris) == 0
    error("In `svd`, the left or right indices are empty (the indices of `A` are ($(inds(A))), but the input indices are ($Lis)). For now, this is not supported. You may have accidentally input the wrong indices.")
  end

  CL = combiner(Lis...)
  CR = combiner(Ris...)

  AC = A * CR * CL

  cL = combinedind(CL)
  cR = combinedind(CR)
  if inds(AC) != IndexSet(cL, cR)
    AC = permute(AC, cL, cR)
  end

  UT,ST,VT,spec = svd(tensor(AC); kwargs...)
  UC,S,VC = itensor(UT),itensor(ST),itensor(VT)

  u = commonind(S,UC)
  v = commonind(S,VC)

  if hasqns(A)
    # Fix the flux of UC,S,VC
    # such that flux(UC) == flux(VC) == QN()
    # and flux(S) == flux(A)
    for b in nzblocks(UC)
      i1 = inds(UC)[1]
      i2 = inds(UC)[2]
      newqn = -dir(i2)*qn(i1,b[1])
      setblockqn!(i2,newqn,b[2])
      setblockqn!(u,newqn,b[2])
    end

    for b in nzblocks(VC)
      i1 = inds(VC)[1]
      i2 = inds(VC)[2]
      newqn = -dir(i2)*qn(i1,b[1])
      setblockqn!(i2,newqn,b[2])
      setblockqn!(v,newqn,b[2])
    end
  end

  U = UC*dag(CL)
  V = VC*dag(CR)

  settags!(U,utags,u)
  settags!(S,utags,u)
  settags!(S,vtags,v)
  settags!(V,vtags,v)

  u = settags(u,utags)
  v = settags(v,vtags)

  return TruncSVD(U,S,V,spec,u,v)
end


"""
    TruncEigen{N}

ITensor factorization type for a truncated eigenvalue 
decomposition, returned by `eigen`.
"""
struct TruncEigen{N}
  U::ITensor{N}
  D::ITensor{2}
  spec::Spectrum
  u::Index
  v::Index
end

# iteration for destructuring into components `U,D,spec,u,v = E`
Base.iterate(E::TruncEigen) = (E.U, Val(:D))
Base.iterate(E::TruncEigen, ::Val{:D}) = (E.D, Val(:spec))
Base.iterate(E::TruncEigen, ::Val{:spec}) = (E.spec, Val(:u))
Base.iterate(E::TruncEigen, ::Val{:u}) = (E.u, Val(:v))
Base.iterate(E::TruncEigen, ::Val{:v}) = (E.v, Val(:done))
Base.iterate(E::TruncEigen, ::Val{:done}) = nothing

function LinearAlgebra.eigen(A::ITensor,
                             Linds = inds(A; plev=0),
                             Rinds = prime(IndexSet(Linds));
                             kwargs...)
  ishermitian::Bool = get(kwargs, :ishermitian, false)
  tags::TagSet = get(kwargs, :tags, "Link,eigen")
  lefttags::TagSet = get(kwargs, :lefttags, tags)
  righttags::TagSet = get(kwargs, :righttags, tags)
  leftplev = get(kwargs, :leftplev, 0)
  rightplev = get(kwargs, :rightplev, 1)

  if lefttags == righttags && leftplev == rightplev
    error("In eigen, left tags and prime level must be different from right tags and prime level")
  end

  Lis = commoninds(A, IndexSet(Linds))

  Ris = commoninds(A, IndexSet(Rinds))

  if length(Lis) == 0 || length(Ris) == 0
    error("In `eigen`, the left or right indices are empty (the indices of `A` are ($(inds(A))), but the input indices are ($Lis)). For now, this is not supported. You may have accidentally input the wrong indices.")
  end

  CL = combiner(Lis...)
  CR = combiner(Ris...)

  AC = A * CR * CL

  cL = combinedind(CL)
  cR = combinedind(CR)
  if inds(AC) != IndexSet(cL,cR)
    AC = permute(AC,cL,cR)
  end

  AT = ishermitian ? Hermitian(tensor(AC)) : tensor(AC)
  UT,DT,spec = eigen(AT;kwargs...)
  UC,D = itensor(UT),itensor(DT)

  u = commonind(UC,D)

  if hasqns(A)
    # Fix the flux of UC,D
    # such that flux(UC) == QN()
    # and flux(D) == flux(A)
    for b in nzblocks(UC)
      i1 = inds(UC)[1]
      i2 = inds(UC)[2]
      newqn = -dir(i2)*qn(i1,b[1])
      setblockqn!(i2,newqn,b[2])
      setblockqn!(u,newqn,b[2])
    end
  end

  U = UC*dag(CL)

  # Set left index tags
  u = commonind(D,U)
  settags!(U,lefttags,u)
  settags!(D,lefttags,u)

  # Set left index plev
  u = commonind(D,U)
  U = setprime(U,leftplev,u)
  D = setprime(D,leftplev,u)

  # Set right index tags and plev
  v = uniqueind(D,U)
  replaceind!(D,v,setprime(settags(u,righttags),rightplev))

  u = commonind(D,U) 
  v = uniqueind(D,U)
  return TruncEigen(U,D,spec,u,v)
end

function LinearAlgebra.qr(A::ITensor,
                          Linds...;
                          kwargs...)
  tags::TagSet = get(kwargs, :tags, "Link,qr")
  Lis = commoninds(A,IndexSet(Linds...))
  Ris = uniqueinds(A,Lis)
  Lpos,Rpos = NDTensors.getperms(inds(A),Lis,Ris)
  QT,RT = qr(tensor(A),Lpos,Rpos;kwargs...)
  Q,R = itensor(QT),itensor(RT)
  q = commonind(Q,R)
  settags!(Q,tags,q)
  settags!(R,tags,q)
  q = settags(q,tags)
  return Q,R,q
end

# TODO: allow custom tags in internal indices?
function NDTensors.polar(A::ITensor,
                       Linds...;
                       kwargs...)
  Lis = commoninds(A,IndexSet(Linds...))
  Ris = uniqueinds(A,Lis)
  Lpos,Rpos = NDTensors.getperms(inds(A),Lis,Ris)
  UT,PT = polar(tensor(A),Lpos,Rpos)
  U,P = itensor(UT),itensor(PT)
  u = commoninds(U,P)
  p = uniqueinds(P,U)
  replaceinds!(U,u,p')
  replaceinds!(P,u,p')
  return U,P,commoninds(U,P)
end


function factorize_svd(A::ITensor,
                       Linds...;
                       kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  tags::TagSet = get(kwargs, :tags, "Link,fact")
  alg::String = get(kwargs, :svd_alg, "recursive")
  U,S,V,spec,u,v = svd(A, Linds...; kwargs..., alg = alg)
  if ortho == "left"
    L,R = U,S*V
  elseif ortho == "right"
    L,R = U*S,V
  elseif ortho == "none"
    sqrtS = S
    sqrtS .= sqrt.(S)
    L,R = U*sqrtS,sqrtS*V
    replaceind!(L,v,u)
  else
    error("In factorize using svd decomposition, ortho keyword $ortho not supported. Supported options are left, right, or none.")
  end

  # Set the tags properly
  l = commonind(L,R)
  settags!(L, tags, l)
  settags!(R, tags, l)
  l = settags(l, tags)

  return L,R,spec,l
end

function factorize_eigen(A::ITensor,
                         Linds...;
                         kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  delta_A2 = get(kwargs, :eigen_perturbation, nothing)
  if ortho == "left"
    Lis = commoninds(A, IndexSet(Linds...))
    simLis = sim(Lis)
    A2 = A * replaceinds(dag(A), Lis, simLis)
    if !isnothing(delta_A2)
      # This assumes delta_A2 has indices:
      # (Lis..., prime(Lis)...)
      A2 += replaceinds(delta_A2, prime(Lis), simLis)
    end
    L, D, spec = eigen(A2, Lis, simLis; ishermitian=true,
                                        kwargs...)
    R = dag(L)*A
  elseif ortho == "right"
    Ris = uniqueinds(A, IndexSet(Linds...))
    simRis = sim(Ris)
    A2 = A * replaceinds(dag(A), Ris, simRis)
    if !isnothing(delta_A2)
      # This assumes delta_A2 has indices:
      # (Ris..., prime(Ris)...)
      A2 += replaceinds(delta_A2, prime(Ris), simRis)
    end
    R, D, spec = eigen(A2, Ris, simRis; ishermitian=true,
                                        kwargs...)
    L = A * dag(R)
  else
    error("In factorize using eigen decomposition, ortho keyword $ortho not supported. Supported options are left or right.")
  end
  return L, R, spec, commonind(L, R)
end

"""
    factorize(A::ITensor, Linds::Index...; <keyword arguments>)

Perform a factorization of `A` into ITensors `L` and `R` such that `A ≈ L * R`.

# Arguments
- `ortho::String = "left"`: Choose orthogonality properties of the factorization.
  - `"left"`: the left factor `L` is an orthogonal basis such that `L * dag(prime(L, commonind(L,R))) ≈ I`. 
  - `"right"`: the right factor `R` forms an orthogonal basis. 
  - `"none"`, neither of the factors form an orthogonal basis, and in general are made as symmetrically as possible (depending on the decomposition used).
- `which_decomp::Union{String, Nothing} = nothing`: choose what kind of decomposition is used. 
  - `nothing`: choose the decomposition automatically based on the other arguments. For example, when `"automatic"` is chosen and `ortho = "left"` or `"right"`, `svd` or `eigen` is used depending on the provided cutoff (`eigen` is only used when the cutoff is greater than `1e-12`, since it has a lower precision).
  - `"svd"`: `L = U` and `R = S * V` for `ortho = "left"`, `L = U * S` and `R = V` for `ortho = "right"`, and `L = U * sqrt.(S)` and `R = sqrt.(S) * V` for `ortho = "none"`. To control which `svd` algorithm is choose, use the `svd_alg` keyword argument. See the documentation for `svd` for the supported algorithms, which are the same as those accepted by the `alg` keyword argument.
  - `"eigen"`: `L = U` and ``R = U^{\\dagger} A`` where `U` is determined from the eigendecompositon ``A A^{\\dagger} = U D U^{\\dagger}`` for `ortho = "left"` (and vice versa for `ortho = "right"`). `"eigen"` is not supported for `ortho = "none"`.

In the future, other decompositions like QR, polar, cholesky, LU, etc. are expected to be supported.

For truncation arguments, see: [`svd`](@ref)
"""
function LinearAlgebra.factorize(A::ITensor,
                                 Linds...;
                                 kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  which_decomp::Union{String, Nothing} = get(kwargs, :which_decomp, nothing)
  cutoff::Float64 = get(kwargs, :cutoff, 0.0)
  eigen_perturbation = get(kwargs, :eigen_perturbation, nothing)
  if !isnothing(eigen_perturbation)
    if !(isnothing(which_decomp) || which_decomp == "eigen")
      error("""when passing a non-trivial eigen_perturbation to `factorize`,
               the which_decomp keyword argument must be either "automatic" or
               "eigen" """)
    end
    which_decomp = "eigen"
  end

  # Deprecated keywords
  if haskey(kwargs, :dir)
    error("""dir keyword in factorize has been replace by ortho.
    Note that the default is now `left`, meaning for the results L,R = factorize(A), L forms an orthogonal basis.""")
  end

  if haskey(kwargs, :which_factorization)
    error("""which_factorization keyword in factorize has been replace by which_decomp.""")
  end

  # Determines when to use eigen vs. svd (eigen is less precise,
  # so eigen should only be used if a larger cutoff is requested)
  automatic_cutoff = 1e-12
  if which_decomp == "svd" || 
     (isnothing(which_decomp) && cutoff ≤ automatic_cutoff)
    L, R, spec, l = factorize_svd(A, Linds...; kwargs...)
  elseif which_decomp == "eigen" ||
         (isnothing(which_decomp) && cutoff > automatic_cutoff)
    L, R, spec, l = factorize_eigen(A, Linds...; kwargs...)
  else
    return throw(ArgumentError("""In factorize, factorization $which_decomp is not currently supported. Use `"svd"`, `"eigen"`, or `nothing`."""))
  end
  return L,R,spec,l
end

