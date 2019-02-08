
function contract!(C::CuArray{T},
                   p::CProps,
                   A::CuArray{T},
                   B::CuArray{T},
                   α::Tα=1.0,
                   β::Tβ=0.0) where {T,Tα<:Number,Tβ<:Number}

  # TODO: This is because the permutation convention in C++ ITensor and
  # permutedims in Julia is different
  p.PA = inv(Permutation(p.PA)).data
  p.PB = inv(Permutation(p.PB)).data
  p.PC = inv(Permutation(p.PC)).data
  tA = 'N'
  if p.permuteA
    aref = reshape(permutedims(A,p.PA),p.dmid,p.dleft)
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(p)
      aref = reshape(A,p.dmid,p.dleft)
      tA = 'T'
    else
      aref = reshape(A,p.dleft,p.dmid)
    end
  end

  tB = 'N'
  if p.permuteB
    bref = reshape(permutedims(B,p.PB),p.dmid,p.dright)
  else
    if Btrans(p)
      bref = reshape(B,p.dright,p.dmid)
      tB = 'T'
    else
      bref = reshape(B,p.dmid,p.dright)
    end
  end

  if p.permuteC
    cref = reshape(C,p.dleft,p.dright)
  else
    if Ctrans(p)
      cref = reshape(C,p.dleft,p.dright)
      if tA=='N' && tB=='N'
        (aref,bref) = (bref,aref)
        tA = tB = 'T'
      elseif tA=='T' && tB=='T'
        (aref,bref) = (bref,aref)
        tA = tB = 'N'
      end
    else
      cref = reshape(C,p.dleft,p.dright)
    end
  end

  CUBLAS.gemm_wrapper!(cref, tA,tB,aref,bref,promote_type(T,Tα)(α),promote_type(T,Tβ)(β))

  if p.permuteC
    permutedims!(C,reshape(cref,p.newCrange...),p.PC)
  end
  return
end

function contract(Cinds::IndexSet,
                  Clabels::Vector{Int},
                  Astore::Dense{SA, TA},
                  Ainds::IndexSet,
                  Alabels::Vector{Int},
                  Bstore::Dense{SB, TB},
                  Binds::IndexSet,
                  Blabels::Vector{Int}) where {SA<:Number,SB<:Number, TA <: CuArray, TB <: CuArray}
  Adims = dims(Ainds)
  Bdims = dims(Binds)
  Cdims = dims(Cinds)

  # Create storage for output tensor
  Cstore = Dense{promote_type(SA,SB), CuVector{promote_type(SA,SB)}}(prod(Cdims))

  Adata = reshape(data(Astore),Adims)
  Bdata = reshape(data(Bstore),Bdims)
  Cdata = reshape(data(Cstore),Cdims)

  contract!(Cdata,Clabels,Adata,Alabels,Bdata,Blabels)
  return Cstore
end

