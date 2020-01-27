using ITensors,
      Test

@testset "Fermions" begin

  @testset "Fermionic QNs" begin
    q = QN("Nf",1,-1)
    @test isfermionic(q[1])
  end


end

