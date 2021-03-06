using ITensors,
      Test

include("util.jl")

function basicRandomMPO(N::Int, sites;dim=4)
  #sites = [Index(2,"Site") for n=1:N]
  M = MPO(sites)
  links = [Index(dim,"n=$(n-1),Link") for n=1:N+1]
  for n=1:N
    M[n] = randomITensor(links[n],sites[n],sites[n]',links[n+1])
  end
  M[1] *= delta(links[1])
  M[N] *= delta(links[N+1])
  return M
end

@testset "MPO Basics" begin
  N = 6
  sites = [Index(2,"Site") for n=1:N]
  @test length(MPO()) == 0
  O = MPO(sites)
  @test length(O) == N

  str = split(sprint(show, O), '\n')
  @test str[1] == "MPO"
  @test length(str) == length(O) + 2

  O[1] = ITensor(sites[1], prime(sites[1]))
  @test hasind(O[1],sites[1])
  @test hasind(O[1],prime(sites[1]))
  P = copy(O)
  @test hasind(P[1],sites[1])
  @test hasind(P[1],prime(sites[1]))
  # test constructor from Vector{ITensor}
  K = randomMPO(sites)
  @test ITensors.data(MPO(copy(ITensors.data(K)))) == ITensors.data(K)

  @testset "orthogonalize!" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    orthogonalize!(phi, 1)
    orthogonalize!(K, 1)
    orig_inner = ⋅(phi, K, phi)
    orthogonalize!(phi, div(N, 2))
    orthogonalize!(K, div(N, 2))
    @test ⋅(phi, K, phi) ≈ orig_inner
  end

  @testset "inner <y|A|x>" begin
    phi = randomMPS(sites)
    K = randomMPO(sites)
    @test maxlinkdim(K) == 1
    psi = randomMPS(sites)
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1]*K[1]*psi[1]
    for j = 2:N
      phiKpsi *= phidag[j]*K[j]*psi[j]
    end
    @test phiKpsi[] ≈ inner(phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(phi,K,badpsi)
    
    # make bigger random MPO...
    for link_dim in 2:5
        mpo_tensors  = ITensor[ITensor() for ii in 1:N]
        mps_tensors  = ITensor[ITensor() for ii in 1:N]
        mps_tensors2 = ITensor[ITensor() for ii in 1:N]
        mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mpo_tensors[1] = randomITensor(mpo_link_inds[1], sites[1], sites[1]') 
        mps_tensors[1] = randomITensor(mps_link_inds[1], sites[1]) 
        mps_tensors2[1] = randomITensor(mps_link_inds[1], sites[1]) 
        for ii in 2:N-1
            mpo_tensors[ii] = randomITensor(mpo_link_inds[ii], mpo_link_inds[ii-1], sites[ii], sites[ii]') 
            mps_tensors[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
            mps_tensors2[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
        end
        mpo_tensors[N] = randomITensor(mpo_link_inds[N-1], sites[N], sites[N]')
        mps_tensors[N] = randomITensor(mps_link_inds[N-1], sites[N])
        mps_tensors2[N] = randomITensor(mps_link_inds[N-1], sites[N])
        K   = MPO(N, mpo_tensors, 0, N+1)
        psi = MPS(N, mps_tensors, 0, N+1)
        phi = MPS(N, mps_tensors2, 0, N+1)
        orthogonalize!(psi, 1; maxdim=link_dim)
        orthogonalize!(K, 1; maxdim=link_dim)
        orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
        phidag = dag(phi)
        prime!(phidag)
        phiKpsi = phidag[1]*K[1]*psi[1]
        for j = 2:N
          phiKpsi *= phidag[j]*K[j]*psi[j]
        end
        @test scalar(phiKpsi) ≈ inner(phi,K,psi)
    end
  end

  @testset "inner <By|A|x>" begin
    phi = makeRandomMPS(sites)

    K = makeRandomMPO(sites,chi=2)
    J = makeRandomMPO(sites,chi=2)

    psi = makeRandomMPS(sites)
    phidag = dag(phi)
    prime!(phidag, 2)
    Jdag = dag(J)
    prime!(Jdag)
    for j ∈ eachindex(Jdag)
      swapprime!(Jdag[j],2,3)
      swapprime!(Jdag[j],1,2)
      swapprime!(Jdag[j],3,1)
    end

    phiJdagKpsi = phidag[1]*Jdag[1]*K[1]*psi[1]
    for j ∈ eachindex(psi)[2:end]
      phiJdagKpsi = phiJdagKpsi*phidag[j]*Jdag[j]*K[j]*psi[j]
    end

    @test phiJdagKpsi[] ≈ inner(J,phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch inner(J,phi,K,badpsi)
  end

  @testset "error_contract" begin
    phi = makeRandomMPS(sites)
    K = makeRandomMPO(sites,chi=2)

    psi = makeRandomMPS(sites)

    dist = sqrt(abs(1 + (inner(phi,phi) - 2*real(inner(phi,K,psi)))
                        /inner(K,psi,K,psi)))
    @test dist ≈ error_contract(phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    # Apply K to phi and check that error_contract is close to 0.
    Kphi = contract(K, phi; method="naive", cutoff=1E-8)
    @test error_contract(Kphi, K, phi) ≈ 0. atol=1e-4

    @test_throws DimensionMismatch contract(K,badpsi;method="naive", cutoff=1E-8)
    @test_throws DimensionMismatch error_contract(phi,K,badpsi)
  end

  @testset "contract" begin
    phi = randomMPS(sites)
    K   = randomMPO(sites)
    @test maxlinkdim(K) == 1
    psi = randomMPS(sites)
    psi_out = contract(K, psi,maxdim=1)
    @test inner(phi,psi_out) ≈ inner(phi,K,psi)
    @test_throws ArgumentError contract(K, psi, method="fakemethod")

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomMPS(badsites)
    @test_throws DimensionMismatch contract(K,badpsi)

    # make bigger random MPO...
    for link_dim in 2:5
      mpo_tensors  = ITensor[ITensor() for ii in 1:N]
      mps_tensors  = ITensor[ITensor() for ii in 1:N]
      mps_tensors2 = ITensor[ITensor() for ii in 1:N]
      mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
      mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
      mpo_tensors[1] = randomITensor(mpo_link_inds[1], sites[1], sites[1]') 
      mps_tensors[1] = randomITensor(mps_link_inds[1], sites[1]) 
      mps_tensors2[1] = randomITensor(mps_link_inds[1], sites[1]) 
      for ii in 2:N-1
        mpo_tensors[ii] = randomITensor(mpo_link_inds[ii], mpo_link_inds[ii-1], sites[ii], sites[ii]') 
        mps_tensors[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
        mps_tensors2[ii] = randomITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
      end
      mpo_tensors[N] = randomITensor(mpo_link_inds[N-1], sites[N], sites[N]')
      mps_tensors[N] = randomITensor(mps_link_inds[N-1], sites[N])
      mps_tensors2[N] = randomITensor(mps_link_inds[N-1], sites[N])
      K   = MPO(N, mpo_tensors, 0, N+1)
      psi = MPS(N, mps_tensors, 0, N+1)
      phi = MPS(N, mps_tensors2, 0, N+1)
      orthogonalize!(psi, 1; maxdim=link_dim)
      orthogonalize!(K, 1; maxdim=link_dim)
      orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
      psi_out = contract(deepcopy(K), deepcopy(psi); maxdim=10*link_dim, cutoff=0.0)
      @test inner(phi, psi_out) ≈ inner(phi, K, psi)
    end
  end

  @testset "add(::MPO, ::MPO)" begin
    shsites = siteinds("S=1/2", N)
    K = randomMPO(shsites)
    L = randomMPO(shsites)
    M = add(K, L)
    @test length(M) == N
    psi = randomMPS(shsites)
    k_psi = contract(K, psi, maxdim=1)
    l_psi = contract(L, psi, maxdim=1)
    @test inner(psi, k_psi + l_psi) ≈ ⋅(psi, M, psi) atol=5e-3
    @test inner(psi, sum([k_psi, l_psi])) ≈ dot(psi, M, psi) atol=5e-3
    for dim in 2:4
        shsites = siteinds("S=1/2",N)
        K = basicRandomMPO(N, shsites; dim=dim)
        L = basicRandomMPO(N, shsites; dim=dim)
        M = K + L
        @test length(M) == N
        psi = randomMPS(shsites)
        k_psi = contract(K, psi)
        l_psi = contract(L, psi)
        @test inner(psi, k_psi + l_psi) ≈ dot(psi, M, psi) atol=5e-3
        @test inner(psi, sum([k_psi, l_psi])) ≈ inner(psi, M, psi) atol=5e-3
        psi = randomMPS(shsites)
        M = add(K, L; cutoff=1E-9)
        k_psi = contract(K, psi)
        l_psi = contract(L, psi)
        @test inner(psi, k_psi + l_psi) ≈ inner(psi, M, psi) atol=5e-3
    end
  end

  @testset "contract(::MPO, ::MPO)" begin
    psi = randomMPS(sites)
    K = randomMPO(sites)
    L = randomMPO(sites)
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = contract(prime(K), L, maxdim=1)
    psi_kl_out = contract(prime(K), contract(L, psi, maxdim=1), maxdim=1)
    @test inner(psi,KL,psi) ≈ inner(psi, psi_kl_out) atol=5e-3

    # where both K and L have differently labelled sites
    othersitesk = [Index(2,"Site,aaa") for n=1:N]
    othersitesl = [Index(2,"Site,bbb") for n=1:N]
    K = randomMPO(sites)
    L = randomMPO(sites)
    for ii in 1:N
      replaceind!(K[ii], sites[ii]', othersitesk[ii])
      replaceind!(L[ii], sites[ii]', othersitesl[ii])
    end
    KL = contract(K, L, maxdim=1)
    psik = randomMPS(othersitesk)
    psil = randomMPS(othersitesl)
    psi_kl_out = contract(K, contract(L, psil, maxdim=1), maxdim=1)
    @test inner(psik,KL,psil) ≈ inner(psik, psi_kl_out) atol=5e-3
    
    badsites = [Index(2,"Site") for n=1:N+1]
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch contract(K,badL)
  end

  @testset "*(::MPO, ::MPO)" begin
    psi = randomMPS(sites)
    K = randomMPO(sites)
    L = randomMPO(sites)
    @test maxlinkdim(K) == 1
    @test maxlinkdim(L) == 1
    KL = *(prime(K), L, maxdim=1)
    psi_kl_out = *(prime(K), *(L, psi, maxdim=1), maxdim=1)
    @test ⋅(psi, KL, psi) ≈ dot(psi, psi_kl_out) atol=5e-3

    # where both K and L have differently labelled sites
    othersitesk = [Index(2,"Site,aaa") for n=1:N]
    othersitesl = [Index(2,"Site,bbb") for n=1:N]
    K = randomMPO(sites)
    L = randomMPO(sites)
    for ii in 1:N
      replaceind!(K[ii], sites[ii]', othersitesk[ii])
      replaceind!(L[ii], sites[ii]', othersitesl[ii])
    end
    KL = *(K, L, maxdim=1)
    psik = randomMPS(othersitesk)
    psil = randomMPS(othersitesl)
    psi_kl_out = *(K, *(L, psil, maxdim=1), maxdim=1)
    @test dot(psik, KL, psil) ≈ psik ⋅ psi_kl_out atol=5e-3
    
    badsites = [Index(2,"Site") for n=1:N+1]
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch K * badL
  end

  sites = siteinds("S=1/2",N)
  O = MPO(sites,"Sz")
  @test length(O) == N # just make sure this works

  @test_throws ArgumentError randomMPO(sites, 2)
  @test_throws ErrorException linkind(MPO(N, fill(ITensor(), N), 0, N + 1), 1)
end

@testset "sweepnext function" begin

  @testset "one site" begin
    N = 6
    count = 1
    output = [(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),
              (6,2),(5,2),(4,2),(3,2),(2,2),(1,2)]
    for (b,ha) in sweepnext(N;ncenter=1)
      @test (b,ha) == output[count]
      count += 1
    end
    @test count == 2*N+1
  end
  
  @testset "two site" begin
    N = 6
    count = 1
    output = [(1,1),(2,1),(3,1),(4,1),(5,1),
              (5,2),(4,2),(3,2),(2,2),(1,2)]
    for (b,ha) in sweepnext(N)
      @test (b,ha) == output[count]
      count += 1
    end
    @test count == 2*(N-1)+1
  end

  @testset "three site" begin
    N = 6
    count = 1
    output = [(1,1),(2,1),(3,1),(4,1),
              (4,2),(3,2),(2,2),(1,2)]
    for (b,ha) in sweepnext(N;ncenter=3)
      @test (b,ha) == output[count]
      count += 1
    end
    @test count == 2*(N-2)+1
  end

end

nothing
