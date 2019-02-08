struct Dense{T,S<:AbstractArray{<:T}} <: TensorStorage
  data::S
  Dense{T, S}(data::S) where {T<:Number, S<:AbstractArray{<:T}} = new{T, S}(data)
  Dense{T, S}(size::Integer) where {T<:Number, S<:AbstractArray{<:T}} = new{T, S}(zeros(T, size))
  Dense{T, S}(x::T, size::Integer) where {T<:Number, S<:AbstractArray{<:T}} = new{T, S}(fill(x, size))
  Dense{T, S}() where {T<:Number, S<:AbstractArray{<:T}} = new{T, S}(S{T}())
end
  
Base.collect(x::Dense{T, S}) where {T<:Number, S<:Array} = x

data(D::Dense) = D.data
length(D::Dense) = length(data(D))
eltype(D::Dense) = eltype(data(D))
getindex(D::Dense,i::Int) = data(D)[i]
#TODO: this should do proper promotions of the storage data
#e.g. ComplexF64*Dense{Float64} -> Dense{ComplexF64}
*(D::T,x::Number) where {T<:Dense} = T(x*data(D))
*(x::Number,D::Dense) = D*x

copy(D::Dense{T, S}) where {T, S} = Dense{T, S}(copy(data(D)))

storage_convert(::Type{Array},D::Dense,is::IndexSet) = reshape(data(D),dims(is))

function storage_getindex(Tstore::Dense,
                          Tis::IndexSet,
                          vals::Int...)
  return getindex(reshape(data(Tstore),dims(Tis)),vals...)
end

function storage_setindex!(Tstore::Dense,Tis::IndexSet,x::Number,vals::Int...)
  return setindex!(reshape(data(Tstore),dims(Tis)),x,vals...)
end

# TODO: optimize this permutation (this does an extra unnecassary permutation
# since permutedims!() doesn't give the option to add the permutation to the original array)
# Maybe wrap the c version?
function storage_add!(Bstore::Dense,Bis::IndexSet,Astore::Dense,Ais::IndexSet)
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if is_trivial_permutation(p)
    Bdata .+= Adata
  else
    reshapeBdata = reshape(Bdata,dims(Bis))
    permAdata = permutedims(reshape(Adata,dims(Ais)),p)
    reshapeBdata .+= permAdata
  end
end

# TODO: make this a special version of storage_add!()
# Make sure the permutation is optimized
function storage_permute!(Bstore::Dense,Bis::IndexSet,Astore::Dense,Ais::IndexSet)
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if is_trivial_permutation(p)
    Bdata .= Adata
  else
    reshapeBdata = reshape(Bdata,dims(Bis))
    permutedims!(reshapeBdata,reshape(Adata,dims(Ais)),p)
  end
end

function storage_dag(Astore::Dense,Ais::IndexSet)
  return dag(Ais),storage_conj(Astore)
end

function storage_scalar(D::Dense)
  if length(D)==1
    return D[1]
  else
    throw(ErrorException("Cannot convert Dense -> Number for length of data greater than 1"))
  end
end

# TODO: make this storage_contract!(), where C is pre-allocated. 
#       This will allow for in-place multiplication
# TODO: optimize the contraction logic so C doesn't get permuted?
function storage_contract(Astore::TensorStorage,
                          Ais::IndexSet,
                          Bstore::TensorStorage,
                          Bis::IndexSet)
  if length(Ais)==0
    Cis = Bis
    Cstore = storage_scalar(Astore)*Bstore
  elseif length(Bis)==0
    Cis = Ais
    Cstore = storage_scalar(Bstore)*Astore
  else
    #TODO: check for special case when Ais and Bis are disjoint sets
    #I think we should do this analysis outside of storage_contract, at the ITensor level
    #(since it is universal for any storage type and just analyzes in indices)
    (Alabels,Blabels) = compute_contraction_labels(Ais,Bis)
    (Cis,Clabels) = contract_inds(Ais,Alabels,Bis,Blabels)
    Cstore = contract(Cis,Clabels,Astore,Ais,Alabels,Bstore,Bis,Blabels)
  end
  return (Cis,Cstore)
end

function storage_svd(Astore::Dense{T, S},
                     Lis::IndexSet,
                     Ris::IndexSet;
                     kwargs...
                    ) where {T, S<:Array}
    maxm::Int = get(kwargs,:maxm,min(dim(Lis),dim(Ris)))
    minm::Int = get(kwargs,:minm,1)
    cutoff::Float64 = get(kwargs,:cutoff,0.0)
    absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
    doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
    utags::String = get(kwargs,:utags,"Link,u")
    vtags::String = get(kwargs,:vtags,"Link,v")

    MU,MS,MV = svd(reshape(data(Astore),dim(Lis),dim(Ris)))

    sqr(x) = x^2
    P = sqr.(MS)
    truncate!(P;maxm=maxm,cutoff=cutoff,absoluteCutoff=absoluteCutoff,doRelCutoff=doRelCutoff)
    dS = length(P)

    if dS < length(MS)
        MU = MU[:,1:dS]
        resize!(MS,dS)
        MV = MV[:,1:dS]
    end

    u = Index(dS,utags)
    v = u(vtags)
    Uis,Ustore = IndexSet(Lis...,u),Dense{T, S}(vec(MU))
     #TODO: make a diag storage
    Sis,Sstore = IndexSet(u,v),Dense{T, S}(vec(Matrix(Diagonal(MS))))
    Vis,Vstore = IndexSet(Ris...,v),Dense{T, S}(S(vec(MV)))
  
    return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
end

function storage_eigen(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet,matrixtype::Type{HS},truncate::Int,tags::String) where {T<:Array,S<:Number, HS}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MD,MU = eigen(HS(reshape(data(Astore),dim_left,dim_right)))
  #TODO: include truncation parameters as keyword arguments
  dim_middle = min(dim_left,dim_right,truncate)
  u = Index(dim_middle,tags)
  v = prime(u)
  Uis,Ustore = IndexSet(Lis...,u),Dense{S, T}(vec(MU[:,1:dim_middle]))
  #TODO: make a diag storage
  Dis,Dstore = IndexSet(u,v),Dense{S, T}(vec(Matrix(Diagonal(MD[1:dim_middle]))))
  return (Uis,Ustore,Dis,Dstore)
end

function polar(A::AbstractMatrix)
  U,S,V = svd(A)
  return U*V',V*Diagonal(S)*V'
end

#TODO: make one generic function storage_factorization(Astore,Lis,Ris,factorization)
function storage_qr(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:Array, S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = qr(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,"Link,u")
  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
  #form of the decomposition
  Qis,Qstore = IndexSet(Lis...,u),vec(Matrix(MQ))
  Pis,Pstore = IndexSet(u,Ris...),vec(Matrix(MP))
  return (Qis,Dense{S, T}(Qstore),Pis,Dense{S, T}(Pstore))
end

function storage_polar(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:Array, S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = polar(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  #u = Index(dim_middle,"Link,u")
  Uis = addtags(Ris,"u")
  Qis,Qstore = IndexSet(Lis...,Uis...),Dense{S, T}(vec(Matrix(MQ)))
  Pis,Pstore = IndexSet(Uis...,Ris...),Dense{S, T}(vec(Matrix(MP)))
  return (Qis,Qstore,Pis,Pstore)
end

