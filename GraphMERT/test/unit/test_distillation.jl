using Test
using GraphMERT
using GraphMERT.Distillation
using Flux

@testset "Distillation" begin
    # 1. Config
    config = DistillationConfig(enabled=true, teacher_weight=0.5, temperature=2.0)
    @test config.enabled == true
    @test config.teacher_weight == 0.5f0
    @test config.temperature == 2.0f0
    
    # 2. Loss Calculation
    # Mock logits: 2 samples, 3 classes
    student_logits = Float32[1.0 2.0; 0.5 1.5; 0.2 0.8]
    teacher_logits = Float32[0.9 2.1; 0.6 1.4; 0.3 0.7]
    
    loss = calculate_distillation_loss(student_logits, teacher_logits, config)
    @test loss isa Float32
    @test loss >= 0.0
    
    # Test disabled
    config_disabled = DistillationConfig(enabled=false)
    loss_disabled = calculate_distillation_loss(student_logits, teacher_logits, config_disabled)
    @test loss_disabled == 0.0f0
end
