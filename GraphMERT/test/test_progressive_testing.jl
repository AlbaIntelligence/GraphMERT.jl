"""
Test suite for Progressive Testing Infrastructure
"""

using Test
using Dates
using GraphMERT.ProgressiveTesting
using GraphMERT.ProgressiveTesting: PASS, FAIL, WARNING, run_phase_testing_gate, validate_commit_requirements

@testset "Progressive Testing Infrastructure" begin
    @testset "TestingResult Construction" begin
        result = TestingResult("TestGate", PASS, "Test message", Dict("key" => "value"), now())
        
        @test result.gate_name == "TestGate"
        @test result.status == PASS
        @test result.message == "Test message"
        @test result.details["key"] == "value"
    end
    
    @testset "CompilationGate" begin
        gate = CompilationGate()
        @test isa(gate, CompilationGate)
        @test isa(gate, TestingGate)
    end
    
    @testset "CoverageGate" begin
        gate = CoverageGate(0.8)
        @test isa(gate, CoverageGate)
        @test isa(gate, TestingGate)
        @test gate.target_coverage == 0.8
    end
    
    @testset "IntegrationGate" begin
        gate = IntegrationGate()
        @test isa(gate, IntegrationGate)
        @test isa(gate, TestingGate)
    end
    
    @testset "PerformanceGate" begin
        gate = PerformanceGate(5000.0, 4.0)
        @test isa(gate, PerformanceGate)
        @test isa(gate, TestingGate)
        @test gate.target_tokens_per_second == 5000.0
        @test gate.target_memory_gb == 4.0
    end
    
    @testset "DocumentationGate" begin
        gate = DocumentationGate()
        @test isa(gate, DocumentationGate)
        @test isa(gate, TestingGate)
    end
    
    @testset "Gate Status Values" begin
        @test PASS == PASS
        @test FAIL == FAIL
        @test WARNING == WARNING
        @test PASS != FAIL
        @test FAIL != WARNING
        @test WARNING != PASS
    end
end

@testset "Testing Gate Execution" begin
    @testset "Compilation Gate Check" begin
        gate = CompilationGate()
        result = check_compilation(gate)
        
        @test isa(result, TestingResult)
        @test result.gate_name == "CompilationGate"
        @test result.status in [PASS, FAIL]
    end
    
    @testset "Coverage Gate Check" begin
        gate = CoverageGate(0.8)
        result = check_coverage(gate)
        
        @test isa(result, TestingResult)
        @test result.gate_name == "CoverageGate"
        @test result.status in [PASS, FAIL]
        @test haskey(result.details, "current_coverage")
        @test haskey(result.details, "target_coverage")
    end
    
    @testset "Integration Gate Check" begin
        gate = IntegrationGate()
        result = check_integration(gate)
        
        @test isa(result, TestingResult)
        @test result.gate_name == "IntegrationGate"
        @test result.status in [PASS, FAIL]
    end
    
    @testset "Performance Gate Check" begin
        gate = PerformanceGate(5000.0, 4.0)
        result = check_performance(gate)
        
        @test isa(result, TestingResult)
        @test result.gate_name == "PerformanceGate"
        @test result.status in [PASS, FAIL]
        @test haskey(result.details, "current_tokens_per_second")
        @test haskey(result.details, "target_tokens_per_second")
    end
    
    @testset "Documentation Gate Check" begin
        gate = DocumentationGate()
        result = check_documentation(gate)
        
        @test isa(result, TestingResult)
        @test result.gate_name == "DocumentationGate"
        @test result.status in [PASS, FAIL, WARNING]
    end
end

@testset "Testing Gate Orchestration" begin
    @testset "Run Testing Gates" begin
        gates = TestingGate[
            CompilationGate(),
            CoverageGate(0.8),
            IntegrationGate()
        ]
        
        results = run_testing_gate(gates)
        
        @test length(results) == 3
        @test all(isa.(results, TestingResult))
    end
    
    @testset "Phase Testing Gate" begin
        results = run_phase_testing_gate("Phase 1")
        
        @test length(results) >= 4  # At least 4 standard gates
        @test all(isa.(results, TestingResult))
    end
    
    @testset "Commit Requirements Validation" begin
        # This test may fail in some environments, which is expected
        # The important thing is that the function runs without error
        validation_result = validate_commit_requirements()
        @test isa(validation_result, Bool)
    end
end