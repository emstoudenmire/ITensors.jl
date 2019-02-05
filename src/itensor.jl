
abstract type Tensor end
struct ITensor <: Tensor
  inds::IndexSet
  store::TensorStorage
  #TODO: check that the storage is consistent with the
  #total dimension of the indices
  ITensor(is::IndexSet,st::TensorStorage) = new(is,st)
end

struct CuITensor <: Tensor
  inds::IndexSet
  store::TensorStorage
  #TODO: check that the storage is consistent with the
  #total dimension of the indices
  CuITensor(is::IndexSet,st::TensorStorage) = new(is,st)
end

# ITensor c-tors
function ITensor(::Type{T},inds::IndexSet) where {T<:Number}
    return ITensor(inds,Dense{T, Vector{T}}(dim(inds)))
end
ITensor(::Type{T},inds::Index...) where {T<:Number} = ITensor(T,IndexSet(inds...))

ITensor(is::IndexSet) = ITensor(Float64,is...)
ITensor(inds::Index...) = ITensor(IndexSet(inds...))

function ITensor(x::S,inds::IndexSet) where {S<:Number}
    return ITensor(inds,Dense{S, Vector{S}}(x,dim(inds)))
end
ITensor(x::S,inds::Index...) where {S<:Number} = ITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function ITensor(A::Array{S},inds::IndexSet) where {S<:Number}
    return ITensor(inds,Dense{S, Vector{S}}(A))
end
ITensor(A::Array{S},inds::Index...) where {S<:Number} = ITensor(A,IndexSet(inds...))

ITensor() = ITensor(IndexSet(),Dense{Nothing}())

# CuITensor c-tors
function CuITensor(::Type{T},inds::IndexSet) where {T<:Number}
    return CuITensor(inds,Dense{T, CuVector{T}}(dim(inds)))
end
CuITensor(::Type{T},inds::Index...) where {T<:Number} = CuITensor(T,IndexSet(inds...))

CuITensor(is::IndexSet) = CuITensor(Float64,is...)
CuITensor(inds::Index...) = CuITensor(IndexSet(inds...))

function CuITensor(x::S,inds::IndexSet) where {S<:Number}
    return CuITensor(inds,Dense{S, CuVector{S}}(x,dim(inds)))
end
CuITensor(x::S,inds::Index...) where {S<:Number} = CuITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function CuITensor(A::Array{S},inds::IndexSet) where {S<:Number}
    return CuITensor(inds,Dense{S, CuVector{S}}(CuArray(A)))
end
CuITensor(A::Array{S},inds::Index...) where {S<:Number} = CuITensor(CuArray(A),IndexSet(inds...))
function CuITensor(A::CuArray{S},inds::IndexSet) where {S<:Number}
    return CuITensor(inds,Dense{S, CuVector{S}}(A))
end
CuITensor(A::CuArray{S},inds::Index...) where {S<:Number} = CuITensor(A,IndexSet(inds...))

CuITensor() = CuITensor(IndexSet(),Dense{Nothing}())

CuITensor(I::ITensor) = CuITensor(CuVector{eltype(I.store.data)}(I.store.data), I.inds)
Base.collect(CI::CuITensor) = ITensor(collect(CI.store.data), CI.inds) 
# This is just a stand-in for a proper delta/diag storage type
function delta(::Type{T},inds::Index...) where {T}
  d = ITensor(zero(T),inds...)
  minm = min(dims(d)...)
  for i ∈ 1:minm
    d[IndexVal.(inds,i)...] = one(T)
  end
  return d
end
delta(inds::Index...) = delta(Float64,inds...)
const δ = delta

inds(T::Tensor) = T.inds
store(T::Tensor) = T.store
eltype(T::Tensor) = eltype(store(T))

order(T::Tensor) = order(inds(T))
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))

copy(T::IT) where {IT<:Tensor} = IT(copy(inds(T)),copy(store(T)))

convert(::Type{Array},T::ITensor) = storage_convert(Array,store(T),inds(T))
convert(::Type{CuArray},T::CuITensor) = storage_convert(CuArray,store(T),inds(T))
Array(T::ITensor) = convert(Array,T::ITensor)

getindex(T::ITensor) = storage_getindex(store(T),inds(T))
getindex(T::ITensor,vals::Int...) = storage_getindex(store(T),inds(T),vals...)
function getindex(T::ITensor,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return getindex(T,vals...)
end

setindex!(T::ITensor,x::Number,vals::Int...) = storage_setindex!(store(T),inds(T),x,vals...)
function setindex!(T::ITensor,x::Number,ivs::IndexVal...)
  p = calculate_permutation(inds(T),ivs)
  vals = val.(ivs)[p]
  return setindex!(T,x,vals...)
end

function commonindex(A::T,B::S) where {T<:Tensor, S<:Tensor}
  return commonindex(inds(A),inds(B))
end
function commoninds(A::T,B::S) where {T<:Tensor, S<:Tensor}
  return inds(A)∩inds(B)
end

hasindex(T::S,I::Index) where {S<:Tensor} = hasindex(inds(T),I)

# TODO: should this make a copy of the storage?
function prime(A::T,vargs...) where {T<:Tensor}
  return T(prime(inds(A),vargs...),store(A))
end
adjoint(A::T) where {T<:Tensor} = prime(A)
function primeexcept(A::T,vargs...) where {T<:Tensor}
  return T(primeexcept(inds(A),vargs...),store(A))
end
function setprime(A::T,vargs...) where {T<:Tensor}
  return T(setprime(inds(A),vargs...),store(A))
end
function noprime(A::T,vargs...) where {T<:Tensor}
  return T(noprime(inds(A),vargs...),store(A))
end
function mapprime(A::T,vargs...) where {T<:Tensor}
  return T(mapprime(inds(A),vargs...),store(A))
end
function swapprime(A::T,vargs...) where {T<:Tensor}
  return T(swapprime(inds(A),vargs...),store(A))
end

function addtags(A::T,vargs...) where {T<:Tensor}
  return T(addtags(inds(A),vargs...),store(A))
end

function removetags(A::T,vargs...) where {T<:Tensor}
  return T(removetags(inds(A),vargs...),store(A))
end

function replacetags(A::T,vargs...) where {T<: Tensor}
  return T(replacetags(inds(A),vargs...),store(A))
end

function swaptags(A::T,vargs...) where {T<: Tensor}
  return T(swaptags(inds(A),vargs...),store(A))
end


function ==(A::ITensor,B::ITensor)::Bool
  inds(A)!=inds(B) && throw(ErrorException("ITensors must have the same Indices to be equal"))
  p = calculate_permutation(inds(B),inds(A))
  for i ∈ CartesianIndices(dims(A))
    A[Tuple(i)...]≠B[Tuple(i)[p]...] && return false
  end
  return true
end

function isapprox(A::T,
                  B::T;
                  atol::Real=0.0,
                  rtol::Real=Base.rtoldefault(eltype(A),eltype(B),atol)) where {T<:Tensor}
  return norm(A-B) <= atol + rtol*max(norm(A),norm(B))
end

function scalar(T::Tensor)
  if !(order(T)==0 || dim(T)==1)
    error("ITensor is not a scalar")
  end
  return storage_scalar(store(T))
end

randn!(T::Tensor) = storage_randn!(store(T))

function randomITensor(::Type{S},inds::IndexSet) where {S<:Number}
  T = ITensor(S,inds)
  randn!(T)
  return T
end
randomITensor(::Type{S},inds::Index...) where {S<:Number} = randomITensor(S,IndexSet(inds...))
randomITensor(inds::IndexSet) = randomITensor(Float64,inds)
randomITensor(inds::Index...) = randomITensor(Float64,IndexSet(inds...))

function randomCuITensor(::Type{S},inds::IndexSet) where {S<:Number}
  T = CuITensor(S,inds)
  randn!(T)
  return T
end
randomCuITensor(::Type{S},inds::Index...) where {S<:Number} = randomCuITensor(S,IndexSet(inds...))
randomCuITensor(inds::IndexSet) = randomCuITensor(Float64,inds)
randomCuITensor(inds::Index...) = randomCuITensor(Float64,IndexSet(inds...))

norm(T::Tensor) = storage_norm(store(T))
dag(T::ITensor) = ITensor(storage_dag(store(T),inds(T))...)
dag(T::CuITensor) = CuITensor(storage_dag(store(T),inds(T))...)

function permute(T::IT,permTinds::IndexSet) where {IT<:Tensor}
  permTstore = typeof(store(T))(dim(T))
  storage_permute!(permTstore,permTinds,store(T),inds(T))
  return IT(permTinds,permTstore)
end
permute(T::Tensor,new_inds::Index...) = permute(T,IndexSet(new_inds...))

function add!(A::T, B::T) where {T<:Tensor}
  storage_add!(store(A),inds(A),store(B),inds(B))
end

#TODO: improve these using a storage_mult call
*(A::ITensor,x::Number) = A*ITensor(x)
*(A::CuITensor,x::Number) = A*CuITensor(x)
*(x::Number,A::ITensor) = A*x
*(x::Number,A::CuITensor) = A*x

/(A::ITensor,x::Number) = A*ITensor(1.0/x)
/(A::CuITensor,x::Number) = A*CuITensor(1.0/x)

-(A::Tensor) = -one(eltype(A))*A
function +(A::T,B::T) where {T<:Tensor}
  #A==B && return 2*A
  C = copy(A)
  add!(C,B)
  return C
end
-(A::T,B::T) where {T<:Tensor} = A+(-B)

function *(A::T,B::T) where {T<:Tensor}
  #TODO: Add special case of A==B
  #A==B && return ITensor(norm(A)^2)
  (Cis,Cstore) = storage_contract(store(A),inds(A),store(B),inds(B))
  C = T(Cis,Cstore)
  return C
end

function findtags(T::Tensor,
                  tags::String)::Index
  ts = TagSet(tags)
  for i in inds(T)
    if hastags(i,ts)
      return i
    end
  end
  error("findtags: ITensor has no Index with given tags: $ts")
  return Index()
end

function eigen(A::ITensor,
               left_inds::Index...;
               truncate::Int=100,
               tags::String="Link,u",
               matrixtype::Type{T}=Hermitian) where {T}
  Lis = IndexSet(left_inds...)
  #TODO: make this a debug level check
  Lis⊈inds(A) && throw(ErrorException("Input indices must be contained in the ITensor"))

  Ris = difference(inds(A),Lis)
  #TODO: check if A is already ordered properly
  #and avoid doing this permute, since it makes a copy
  #AND/OR use svd!() to overwrite the data of A to save memory
  A = permute(A,Lis...,Ris...)
  #TODO: More of the index analysis should be moved out of storage_eigen
  Uis,Ustore,Dis,Dstore = storage_eigen(store(A),Lis,Ris,matrixtype,truncate,tags)
  return ITensor(Uis,Ustore),ITensor(Dis,Dstore)
end

function show(io::IO,
              T::ITensor)
  print(io,"ITensor ord=$(order(T))")
  for i = 1:order(T)
    print(io," ",inds(T)[i])
  end
  #@printf(io,"\n{%s log(scale)=%.1f}",storageTypeName(store(T)),lnum(scale(T)))
end

