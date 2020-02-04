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
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")

    N = ITensor(QN("Nf",0,-1),s',dag(s))

    N[s'(2),s(2)] = 1.0
    @test N[s'(2),s(2)] ≈ +1.0
    @test N[s(2),s'(2)] ≈ -1.0

    N[s(2),s'(2)] = 1.0
    @test N[s'(2),s(2)] ≈ -1.0
    @test N[s(2),s'(2)] ≈ 1.0

    C = ITensor(QN("Nf",-1,-1),s',dag(s))

    C[s'(1),s(2)] = 1.0
    @test C[s'(1),s(2)] ≈ 1.0
    @test C[s(2),s'(1)] ≈ 1.0


    I = ITensor(QN("Nf",0,-1),s',dag(s))
    I[s'(1),s(1)] = 1.0
    I[s'(2),s(2)] = 1.0
    @test I[s'(1),s(1)] ≈ 1.0
    @test I[s'(2),s(2)] ≈ 1.0

    @test I[s(1),s'(1)] ≈ 1.0
    @test I[s(2),s'(2)] ≈ -1.0
  end

  @testset "Permute Fermionic ITensors" begin
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")

    #
    # Operator tests
    # 

    N1 = ITensor(QN("Nf",0,-1),s',dag(s))
    N1[s'(2),s(2)] = 1.0
    #@show N1

    N2 = ITensor(QN("Nf",0,-1),dag(s),s')
    N2[s'(2),s(2)] = 1.0
    #@show N2

    pN1 = permute(N1,dag(s),s')
    #@show pN1
    @test pN1[s'(2),s(2)] ≈ 1.0

    pN2 = permute(N2,s',dag(s))
    #@show pN2
    @test pN2[s'(2),s(2)] ≈ 1.0
  end

  @testset "Add Fermionic ITensors" begin
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")

    N1 = ITensor(QN("Nf",0,-1),s',dag(s))
    N1[s'(2),s(2)] = 1.0

    N2 = ITensor(QN("Nf",0,-1),dag(s),s')
    N2[s'(2),s(2)] = 1.0

    NN = N1+N2
    @show NN
    @test NN[s'(2),s(2)] ≈ 2.0

    NN = N1+N1
    @test NN[s'(2),s(2)] ≈ 2.0

    NN = N2+N2
    @test NN[s'(2),s(2)] ≈ 2.0
  end

end

