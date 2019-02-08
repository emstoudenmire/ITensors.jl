Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:CuArray, SB<:Array} = Dense{T, S}(CuArray(x))
Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:Array, SB<:CuArray} = Dense{T, S}(collect(x.data))
Base.collect(x::Dense{T, S}) where {T<:Number, S<:CuArray} = Dense{T, Vector{T}}(collect(x.data))

function storage_svd(Astore::Dense{T, S},
                     Lis::IndexSet,
                     Ris::IndexSet;
                     kwargs...
                    ) where {T, S<:CuArray}
  maxm::Int = get(kwargs,:maxm,min(dim(Lis),dim(Ris)))
  minm::Int = get(kwargs,:minm,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  utags::String = get(kwargs,:utags,"Link,u")
  vtags::String = get(kwargs,:vtags,"Link,v")
  rsd = reshape(data(Astore),dim(Lis),dim(Ris))
  MU,MS,MV = CUSOLVER.svd(rsd)

  sqr(x) = x^2
  P = collect(sqr.(MS))
  truncate!(P;maxm=maxm,cutoff=cutoff,absoluteCutoff=absoluteCutoff,doRelCutoff=doRelCutoff)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    MS = MS[1:dS]
    MV = MV[:,1:dS]
  end

  u = Index(dS,utags)
  v = u(vtags)
  Uis,Ustore = IndexSet(Lis...,u),Dense{T, CuVector{T}}(vec(MU))
  #TODO: make a diag storage
  Sis,Sstore = IndexSet(u,v),Dense{T, CuVector{T}}(vec(CuMatrix(Diagonal(MS))))
  Vis,Vstore = IndexSet(Ris...,v),Dense{T, CuVector{T}}(CuVector{T}(vec(MV)))

  return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
end

function storage_eigen(Astore::Dense{S, T}, Lis::IndexSet,Ris::IndexSet,matrixtype::Type{T},truncate::Int,tags::String) where {T<:CuArray,S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  local d_W, d_V
  d_A = reshape(data(Astore),dim_left,dim_right)
  if( S <: Complex )
    d_W, d_V   = CUSOLVER.heevd!('V','U', d_A)
  else
    d_W, d_V   = CUSOLVER.syevd!('V','U', d_A)
  end
  #TODO: include truncation parameters as keyword arguments
  dim_middle = min(dim_left,dim_right,truncate)
  u = Index(dim_middle,tags)
  v = prime(u)
  Uis,Ustore = IndexSet(Lis...,u),Dense{S, T}(vec(d_V[:,1:dim_middle]))
  #TODO: make a diag storage
  Dis,Dstore = IndexSet(u,v),Dense{S, T}(vec(Matrix(Diagonal(d_W[1:dim_middle]))))
  return (Uis,Ustore,Dis,Dstore)
end

function storage_qr(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:CuArray, S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  dQR = qr!(reshape(data(Astore),dim_left,dim_right))
  MQ = dQR.Q
  MP = dQR.R
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,"Link,u")
  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
  #form of the decomposition
  Qis,Qstore = IndexSet(Lis...,u),Dense{S, T}(vec(CuArray(MQ)))
  Pis,Pstore = IndexSet(u,Ris...),Dense{S, T}(vec(CuArray(MP)))
  return (Qis,Qstore,Pis,Pstore)
end

function storage_polar(Astore::Dense{S, T},Lis::IndexSet,Ris::IndexSet) where {T<:CuArray, S<:Number}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = polar(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  #u = Index(dim_middle,"Link,u")
  Uis = addtags(Ris,"u")
  Qis,Qstore = IndexSet(Lis...,Uis...),Dense{S, T}(vec(MQ))
  Pis,Pstore = IndexSet(Uis...,Ris...),Dense{S, T}(vec(MP))
  return (Qis,Qstore,Pis,Pstore)
end

