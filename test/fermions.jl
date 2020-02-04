using ITensors,
      Test

@testset "Fermions" begin

  @testset "Fermionic QNs" begin
    q = QN("Nf",1,-1)
    @test isfermionic(q[1])
    @test fparity(q) == 1

    q = q+q+q
    @test val(q,"Nf") == 3

    p = QN("P",1,-2)
    @test fparity(p) == 1
    @test fparity(p+p) == 0
    @test fparity(p+p+p) == 1
  end

  @testset "Fermionic IndexVals" begin
    sn = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")
    @test fparity(sn(1)) == 0
    @test fparity(sn(2)) == 1

    sp = Index([QN("Nfp",0,-2)=>1,QN("Nfp",1,-2)=>1],"sp")
    @test fparity(sp(1)) == 0
    @test fparity(sp(2)) == 1
  end

  @testset "Get and Set Elements" begin
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"s")

    N = ITensor(s',dag(s))

    N[s'(2),s(2)] = 1.0
    @test N[s'(2),s(2)] ≈ +1.0
    @test N[s(2),s'(2)] ≈ -1.0

    N[s(2),s'(2)] = 1.0
    @test N[s'(2),s(2)] ≈ -1.0
    @test N[s(2),s'(2)] ≈ 1.0

    C = ITensor(s',dag(s))

    C[s'(1),s(2)] = 1.0
    @test C[s'(1),s(2)] ≈ 1.0
    @test C[s(2),s'(1)] ≈ 1.0


    I = ITensor(s',dag(s))
    I[s'(1),s(1)] = 1.0
    I[s'(2),s(2)] = 1.0
    @test I[s'(1),s(1)] ≈ 1.0
    @test I[s'(2),s(2)] ≈ 1.0

    @test I[s(1),s'(1)] ≈ 1.0
    @test I[s(2),s'(2)] ≈ -1.0
  end

  @testset "Permute and Add Fermionic ITensors" begin

    @testset "Permute Operators" begin
      s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"s")

      N1 = ITensor(s',dag(s))
      N1[s'(2),s(2)] = 1.0

      N2 = ITensor(dag(s),s')
      N2[s'(2),s(2)] = 1.0

      pN1 = permute(N1,dag(s),s')
      @test pN1[s'(2),s(2)] ≈ 1.0

      pN2 = permute(N2,s',dag(s))
      @test pN2[s'(2),s(2)] ≈ 1.0
    end

    @testset "Add Operators" begin
      s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")

      N1 = ITensor(s',dag(s))
      N1[s'(2),s(2)] = 1.0

      N2 = ITensor(dag(s),s')
      N2[s'(2),s(2)] = 1.0

      NN = N1+N2
      @test NN[s'(2),s(2)] ≈ 2.0

      NN = N1+N1
      @test NN[s'(2),s(2)] ≈ 2.0

      NN = N2+N2
      @test NN[s'(2),s(2)] ≈ 2.0
    end

    @testset "Wavefunction Tests" begin
      s = [Index([QN("N",0,-2)=>2,QN("N",1,-2)=>2],"s$n") for n=1:4]

      psi0 = ITensor(s...)

      psi0[s[1](1),s[2](1),s[3](1),s[4](1)] = 1111
      psi0[s[1](3),s[2](3),s[3](1),s[4](1)] = 3311
      psi0[s[1](1),s[2](3),s[3](1),s[4](3)] = 1313

      psi1 = permute(psi0,s[2],s[1],s[3],s[4])
      @test norm(psi1-psi0) ≈ 0.0

      @test psi0[s[1](1),s[2](1),s[3](1),s[4](1)] ≈ 1111
      @test psi1[s[1](1),s[2](1),s[3](1),s[4](1)] ≈ 1111
      @test psi0[s[2](1),s[1](1),s[3](1),s[4](1)] ≈ 1111
      @test psi1[s[2](1),s[1](1),s[3](1),s[4](1)] ≈ 1111

      @test psi0[s[1](3),s[2](3),s[3](1),s[4](1)] ≈ 3311
      @test psi1[s[1](3),s[2](3),s[3](1),s[4](1)] ≈ 3311
      @test psi0[s[2](3),s[1](3),s[3](1),s[4](1)] ≈ -3311
      @test psi1[s[2](3),s[1](3),s[3](1),s[4](1)] ≈ -3311
      @test psi0[s[4](1),s[2](3),s[1](3),s[3](1)] ≈ -3311
      @test psi1[s[4](1),s[2](3),s[1](3),s[3](1)] ≈ -3311

      psi2 = permute(psi0,s[4],s[1],s[3],s[2])
      @test norm(psi2-psi0) ≈ 0.0
      @test norm(psi2-psi1) ≈ 0.0

      @test psi0[s[1](1),s[2](3),s[3](1),s[4](3)] ≈ 1313
      @test psi1[s[1](1),s[2](3),s[3](1),s[4](3)] ≈ 1313
      @test psi2[s[1](1),s[2](3),s[3](1),s[4](3)] ≈ 1313
      @test psi0[s[4](3),s[1](1),s[3](1),s[2](3)] ≈ -1313
      @test psi1[s[4](3),s[1](1),s[3](1),s[2](3)] ≈ -1313
      @test psi2[s[4](3),s[1](1),s[3](1),s[2](3)] ≈ -1313
    end
  end


end

