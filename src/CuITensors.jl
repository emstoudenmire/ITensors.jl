
module CuITensors

using CuArrays
using CuArrays.CUBLAS
using CuArrays.CUSOLVER
using LinearAlgebra
using ..ITensors
using ..ITensors: CProps, Permutation, Atrans, Btrans, Ctrans, truncate!
import ITensors.randn!, ITensors.storage_polar, ITensors.storage_qr, ITensors.storage_eigen, ITensors.storage_svd, ITensors.qr!, ITensors.contract!, ITensors.contract
include("storage/cudense.jl")
include("storage/cucontract.jl")
include("cuitensor.jl")

export cuITensor,
       randomCuITensor

end #module
