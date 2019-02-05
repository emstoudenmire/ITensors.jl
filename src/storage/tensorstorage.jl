
abstract type TensorStorage end

#
# Generic ITensor storage functions
#
using CuArrays.CUBLAS
storage_randn!(S::TensorStorage) = randn!(data(S))
storage_norm(S::TensorStorage) = CUBLAS.norm(CuArray(data(S)))
storage_conj(S::T) where {T<:TensorStorage}= T(conj(data(S)))

