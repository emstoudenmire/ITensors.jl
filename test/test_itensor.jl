using ITensors,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      CuArrays,
      Test

digits(::Type{T},i,j,k) where {T} = T(i*10^2+j*10+k)

@testset "ITensor, Dense{$SType} storage" for SType ∈ (Float64,)#,ComplexF64)
  mi,mj,mk,ml,mα = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  α = Index(mα,"α") 
  @testset "Set and get values with IndexVals" begin
    A = ITensor(SType,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[k(kk),j(jj),i(ii)] = digits(SType,ii,jj,kk)
    end
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[j(jj),k(kk),i(ii)]==digits(SType,ii,jj,kk)
    end
  end
  @testset "Test permute(ITensor,Index...)" begin
    A = randomITensor(SType,i,k,j)
    permA = permute(A,k,j,i)
    @test k==inds(permA)[1]
    @test j==inds(permA)[2]
    @test i==inds(permA)[3]
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[k(kk),i(ii),j(jj)]==permA[i(ii),j(jj),k(kk)]
    end
  end
  @testset "Set and get values with Ints" begin
    A = ITensor(SType,i,j,k)
    A = permute(A,k,i,j)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[kk,ii,jj] = digits(SType,ii,jj,kk)
    end
    A = permute(A,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[ii,jj,kk]==digits(SType,ii,jj,kk)
    end
  end
  @testset "Test scalar(ITensor)" begin
    x = SType(34)
    A = ITensor(x)
    @test x==scalar(A)
  end
  @testset "Test norm(ITensor)" begin
    A = randomITensor(SType,i,j,k)
    B = dag(A)*A
    @test norm(A)≈sqrt(scalar(B))
  end
  @testset "Test add ITensors" begin
    A = randomITensor(SType,i,j,k)
    B = randomITensor(SType,k,i,j)
    C = A+B
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test C[i(ii),j(jj),k(kk)]==A[j(jj),i(ii),k(kk)]+B[i(ii),k(kk),j(jj)]
    end
    @test Array(permute(C,i,j,k))==Array(permute(A,i,j,k))+Array(permute(B,i,j,k))
  end

  @testset "Test factorizations of an ITensor" begin

    A = randomITensor(SType,i,j,k,l)

    @testset "Test SVD of an ITensor" begin
      U,S,V = svd(A,j,l)
      u = commonindex(U,S)
      v = commonindex(S,V)
      @test A≈U*S*V
      @test U*dag(prime(U,u))≈δ(SType,u,u') atol=1e-14
      @test V*dag(prime(V,v))≈δ(SType,v,v') atol=1e-14
    end

    @testset "Test SVD truncation" begin 
        M = randn(4,4)
        (U,s,V) = svd(M)
        ii = Index(4)
        jj = Index(4)
        S = Diagonal(s)
        T = ITensor(IndexSet(ii,jj),Dense{Float64, Vector{Float64}}(vec(U*S*V')))
        (U,S,V) = svd(T,ii;maxm=2)
        @test norm(U*S*V-T)≈sqrt(s[3]^2+s[4]^2)
    end 

    @testset "Test QR decomposition of an ITensor" begin
      Q,R = qr(A,i,l)
      q = commonindex(Q,R)
      @test A≈Q*R
      @test Q*dag(prime(Q,q))≈δ(SType,q,q') atol=1e-14
    end

    @testset "Test polar decomposition of an ITensor" begin
      U,P = polar(A,k,l)
      @test A≈U*P
      #Note: this is only satisfied when left dimensions 
      #are greater than right dimensions
      uinds = commoninds(U,P)
      UUᵀ =  U*dag(prime(U,"u"))
      for ii ∈ dim(uinds[1]), jj ∈ dim(uinds[2])
        @test UUᵀ[uinds[1](ii),uinds[2](jj),prime(uinds[1])(ii),prime(uinds[2])(jj)]≈one(SType) atol=1e-14
      end
    end

  end # End ITensor factorization testset
end # End Dense storage test
# gpu tests!
@testset "CuITensor, Dense{$SType} storage" for SType ∈ (Float64,)#,ComplexF64)
  mi,mj,mk,ml,mα = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  α = Index(mα,"α") 
  @testset "Set and get values with IndexVals" begin
    A = ITensor(SType,i,j,k)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[k(kk),j(jj),i(ii)] = digits(SType,ii,jj,kk)
    end
    CA = CuITensor(A)
    AA = collect(CA)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test AA[j(jj),k(kk),i(ii)]==digits(SType,ii,jj,kk)
    end
  end
  @testset "Test permute(CuITensor,Index...)" begin
    CA = randomCuITensor(SType,i,k,j)
    permCA = permute(CA,k,j,i)
    permA = collect(permCA)
    @test k==inds(permA)[1]
    @test j==inds(permA)[2]
    @test i==inds(permA)[3]
    A = collect(CA)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[k(kk),i(ii),j(jj)]==permA[i(ii),j(jj),k(kk)]
    end
  end
  @testset "Set and get values with Ints" begin
    A = ITensor(SType,i,j,k)
    A = permute(A,k,i,j)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[kk,ii,jj] = digits(SType,ii,jj,kk)
    end
    CA = CuITensor(A)
    CA = permute(CA,i,j,k)
    A = collect(CA)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[ii,jj,kk]==digits(SType,ii,jj,kk)
    end
  end
  @testset "Test scalar(CuITensor)" begin
    x = SType(34)
    A = CuITensor(x)
    @test x==scalar(A)
  end
  @testset "Test norm(CuITensor)" begin
    A = randomCuITensor(SType,i,j,k)
    B = dag(A)*A
    @test norm(A)≈sqrt(scalar(B))
  end
  @testset "Test add CuITensors" begin
    dA = randomCuITensor(SType,i,j,k)
    dB = randomCuITensor(SType,k,i,j)
    C = collect(dA+dB)
    A = collect(dA)
    B = collect(dB)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test C[i(ii),j(jj),k(kk)]==A[j(jj),i(ii),k(kk)]+B[i(ii),k(kk),j(jj)]
    end
    @test Array(permute(C,i,j,k))==Array(permute(A,i,j,k))+Array(permute(B,i,j,k))
  end

  @testset "Test factorizations of a CuITensor" begin

    A = randomCuITensor(SType,i,j,k,l)

    @testset "Test SVD of a CuITensor" begin
      U,S,V = svd(A,j,l)
      u = commonindex(U,S)
      v = commonindex(S,V)
      @test collect(A)≈U*S*V
      @test U*dag(prime(U,u))≈δ(SType,u,u') atol=1e-14
      @test V*dag(prime(V,v))≈δ(SType,v,v') atol=1e-14
    end

    @testset "Test SVD truncation" begin 
        M = randn(4,4)
        (U,s,V) = svd(M)
        ii = Index(4)
        jj = Index(4)
        S = Diagonal(s)
        T = CuITensor(IndexSet(ii,jj),Dense{Float64, CuVector{Float64}}(vec(CuArray(U*S*V'))))
        (U,S,V) = svd(T,ii;maxm=2)
        @test norm(CuITensor(U*S*V)-T)≈sqrt(s[3]^2+s[4]^2)
    end 

    @testset "Test QR decomposition of an ITensor" begin
      Q,R = qr(A,i,l)
      q = commonindex(Q,R)
      @test collect(A)≈collect(Q*R)
      @test collect(Q*dag(prime(Q,q)))≈δ(SType,q,q') atol=1e-14
    end

    @testset "Test polar decomposition of an ITensor" begin
      U,P = polar(A,k,l)
      @test collect(A)≈collect(U*P)
      #Note: this is only satisfied when left dimensions 
      #are greater than right dimensions
      uinds = commoninds(U,P)
      UUᵀ =  collect(U*dag(prime(U,"u")))
      for ii ∈ dim(uinds[1]), jj ∈ dim(uinds[2])
        @test UUᵀ[uinds[1](ii),uinds[2](jj),prime(uinds[1])(ii),prime(uinds[2])(jj)]≈one(SType) atol=1e-14
      end
    end

  end # End ITensor factorization testset
end # End Dense storage test
