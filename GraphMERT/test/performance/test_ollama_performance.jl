"""
Performance tests for Ollama-based entity extraction.

Tests the performance requirements:
- T029: Performance benchmark for entity extraction
- T030: Extraction completes in under 5 minutes for Wikipedia article
- T031: Memory usage verification for 8GB RAM constraint

To run these tests:
    julia --project=. test/performance/test_ollama_performance.jl
"""

using Test
using GraphMERT
using GraphMERT.OllamaClient
using Dates

const WIKIPEDIA_SAMPLE = """
Louis XIV, also known as the Sun King, was the King of France from 1643 until his death in 1715. 
His reign of 72 years and 110 days is the longest of any major European sovereign. 
Louis XIV was born at the Château de Saint-Germain-en-Laye on September 5, 1638, 
and died at Versailles on September 1, 1715. He was the third son of Louis XIII and Anne of Austria. 
He succeeded his father on May 14, 1643. During his reign, France became the dominant power in Europe.
"""

const LONG_WIKIPEDIA_ARTICLE = """
Louis XIV (5 September 1638 – 1 September 1715), the Sun King, was King of France from 1643 to 1715.
His reign of 72 years and 110 days is the longest of any major European sovereign in history.
He was born at the Château de Saint-Germain-en-Laye in France, the third son of Louis XIII and Anne of Austria.
He succeeded his father on 14 May 1643 and became king under the regency of his mother, Anne of Austria.
During his early reign, Cardinal Mazarin, chief minister of France, consolidated his power.
In 1661, Louis XIV began his personal rule, and appointed Jean-Baptiste Colbert as controller-general of finances.
His wars and building projects were funded by efficient taxation and a professional bureaucracy.
Louis XIV's aggression led to the War of the Spanish Succession after his death.
He married Maria Theresa of Spain in 1660, and had several children with her.
His brother, Philippe I, Duke of Orléans, was his successor's regent for Louis XV.
Louis XIV was succeeded by his great-grandson Louis XV, as both his son and grandson died before him.
Louis XIV built the Palace of Versailles and moved the French court there in 1682.
His ministers included Colbert, Louvois, and Seignelay.
France fought multiple wars: the War of Devolution, the Franco-Dutch War, the War of the League of Augsburg, and the War of the Spanish Succession.
The Palace of Versailles became a symbol of absolute monarchy and French cultural excellence.
Louis XIV's domestic policies promoted commerce and manufacturing through the system of mercantilism.
He also pursued colonial expansion, establishing French colonies in North America and the Caribbean.
The Code Noir, promulgated in 1685, regulated the treatment of slaves in French colonies.
Louis XIV died at Versailles in 1715 and was buried at the Basilica of Saint-Denis.
"""

function get_memory_usage_mb()
    return Base.gc_live_bytes() / (1024 * 1024)
end

@testset "Ollama Entity Extraction Performance Tests" begin

    @testset "T029: Entity Extraction Benchmark" begin
        if !is_available()
            @info "Skipping Ollama benchmark test - Ollama not available"
            skip = true
        else
            config = OllamaConfig(model="lfm2.5-thinking:latest", timeout=180)
            client = OllamaLLMClient(config)

            test_texts = [
                ("Short text", "Louis XIV was King of France."),
                ("Medium text", WIKIPEDIA_SAMPLE),
                ("Long text", LONG_WIKIPEDIA_ARTICLE),
            ]

            results = []
            for (name, text) in test_texts
                start_time = time()
                entities = discover_entities(client, text, "wikipedia")
                end_time = time()

                elapsed = end_time - start_time
                chars_per_sec = length(text) / elapsed

                push!(results, (name=name, text=text, elapsed=elapsed, entities=entities, chars_per_sec=chars_per_sec))

                @info "Entity extraction ($name): $(round(elapsed, digits=2))s, $(length(entities)) entities, $(round(chars_per_sec, digits=0)) chars/sec"
            end

            @test all(r -> r.elapsed < 120, results)
            @test all(r -> length(r.entities) > 0, results)

            @info "Benchmark results:"
            for r in results
                @info "  $(r.name): $(round(r.elapsed, digits=2))s for $(length(r.text)) chars"
            end
        end
    end

    @testset "T030: Wikipedia Article Extraction (5-min requirement)" begin
        if !is_available()
            @info "Skipping 5-min verification test - Ollama not available"
        else
            config = OllamaConfig(model="lfm2.5-thinking:latest", timeout=300)
            client = OllamaLLMClient(config)

            article = LONG_WIKIPEDIA_ARTICLE

            start_time = time()
            entities = discover_entities(client, article, "wikipedia")
            relations = match_relations(client, entities, article)
            end_time = time()

            elapsed = end_time - start_time

            @info "Full extraction: $(round(elapsed, digits=2))s (limit: 300s)"
            @info "  Entities found: $(length(entities))"
            @info "  Relations found: $(length(relations))"

            @test elapsed < 300
            @test length(entities) > 0
        end
    end

    @testset "T031: Memory Usage Verification (8GB RAM)" begin
        if !is_available()
            @info "Skipping memory test - Ollama not available"
        else
            config = OllamaConfig(model="lfm2.5-thinking:latest", timeout=300)
            client = OllamaLLMClient(config)

            GC.gc()
            memory_before = get_memory_usage_mb()

            for i in 1:3
                entities = discover_entities(client, LONG_WIKIPEDIA_ARTICLE, "wikipedia")
            end

            GC.gc()
            memory_after = get_memory_usage_mb()

            memory_used = memory_after - memory_before

            @info "Memory usage: Before=$(round(memory_before, digits=1))MB, After=$(round(memory_after, digits=1))MB, Delta=$(round(memory_used, digits=1))MB"

            @test memory_used < 4096

            total_memory = Sys.total_memory() / (1024 * 1024)
            available_memory = total_memory - memory_after

            @info "System memory: Total=$(round(total_memory, digits=0))MB, Available=$(round(available_memory, digits=0))MB"

            @test available_memory > 1024
        end
    end

    @testset "Concurrent Extraction Performance" begin
        if !is_available()
            @info "Skipping concurrent test - Ollama not available"
        else
            config = OllamaConfig(model="lfm2.5-thinking:latest", timeout=180)
            client = OllamaLLMClient(config)

            texts = [
                "Louis XIV was King of France.",
                "Marie Curie discovered radium.",
                "Albert Einstein developed relativity.",
                "The Eiffel Tower is in Paris.",
                "Shakespeare wrote Hamlet.",
            ]

            start_time = time()
            results = [discover_entities(client, text, "wikipedia") for text in texts]
            elapsed = time() - start_time

            throughput = length(texts) / elapsed

            @info "Concurrent extraction: $(round(elapsed, digits=2))s for $(length(texts)) texts ($(round(throughput, digits=2)) texts/sec)"

            @test all(r -> length(r) > 0, results)
            @test elapsed < 60
        end
    end

    @testset "Tail Formation Performance" begin
        if !is_available()
            @info "Skipping tail formation test - Ollama not available"
        else
            config = OllamaConfig(model="lfm2.5-thinking:latest", timeout=180)
            client = OllamaLLMClient(config)

            head = "Louis XIV"
            relation = "born in"
            context = "Louis XIV was born at the Château de Saint-Germain-en-Laye on September 5, 1638."

            start_time = time()
            tail = form_tail_from_tokens(client, [head], context; relation=relation)
            elapsed = time() - start_time

            @info "Tail formation: $(round(elapsed, digits=2))s"

            @test elapsed < 60
        end
    end
end

println("✅ Ollama performance tests completed!")
