export fparity,
       isfermionic,
       parity_sign

"""
    parity_sign(P)

Given an array or tuple of integers representing  
a permutation or a subset of a permutation, 
compute the parity sign defined as -1 for a 
permutation consisting of an odd number of swaps 
and +1 for an even number of swaps. This 
implementation uses an O(n^2) algorithm and is 
intended for small permutations only.
"""
function parity_sign(P)::Int
  len = length(P)
  par = +1
  for i=1:len
    for j=i+1:len
      par *= sign(P[j]-P[i])
    end
  end
  return par
end

isfermionic(qv::QNVal) = (modulus(qv) < 0)

#function isfermionic(qn::QN)
#  for qv in qn
#    isfermionic(qv) && return true
#  end
#  return false
#end
#
#isfermionic(iv::IndexVal) = isfermionic(qn(ind(iv),val(iv)))

"""
    fparity(qn::QN)

Compute the fermion parity (0 or 1) of a QN, 
defined as the sum mod 2 of each of its fermionic 
QNVals (QNVals with negative modulus).
"""
function fparity(qn::QN)
  p = 0
  for qv in qn
    if isfermionic(qv)
      p += val(qv)
    end
  end
  return mod(p,2)
end

fparity(iv::IndexVal) = fparity(qn(ind(iv),val(iv)))

"""
    permfactor(p,ivs::Vararg{QNIndexVal,N})

Given a permutation p and a set of QNIndexVals,
if the subset of index vals which are fermion-parity
odd undergo an odd permutation (odd number of swaps)
according to p, then return -1. Otherwise return +1.
"""
function permfactor(p,ivs::Vararg{QNIndexVal,N}) where {N}
  oddp = @MVector zeros(Int,N)
  n = 0
  for j=1:N
    if fparity(ivs[p[j]]) == 1
      n += 1
      oddp[n] = p[j]
    end
  end
  return parity_sign(oddp[1:n])
end

# Default implementation for non-QN IndexVals
permfactor(p,ivs::Vararg{IndexVal,N}) where {N} = 1.0

# May not need this to be part of Tensors
function Tensors.permfactor(p,block::NTuple{N,Int},inds::IndexSet) where {N}
  ivs = [inds[n](block[n]) for n=1:length(block)]
  return permfactor(p,ivs...)
end

function Tensors.scale_by_permfactor!(B,perm,block::NTuple{N,Int},inds::IndexSet) where {N}
  fac = Tensors.permfactor(perm,block,inds)
  scale!(B,fac)
end

