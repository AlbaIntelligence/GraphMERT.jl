#!/usr/bin/env julia
"""
Wikipedia extraction performance check.

Runs extract_knowledge_graph on the Wikipedia test articles (fixtures) and reports:
- Number of articles (extractions)
- Time per extraction and total time
- Entities and relations per article
"""

using GraphMERT

function main()
    # Load Wikipedia domain
    if !GraphMERT.has_domain("wikipedia")
        domain = GraphMERT.load_wikipedia_domain()
        GraphMERT.register_domain!("wikipedia", domain)
    end

    # Get test articles from fixtures (3 French monarchy articles)
    include(joinpath(@__DIR__, "..", "test", "wikipedia", "fixtures.jl"))
    articles = Base.invokelatest(get_test_articles)
    n_articles = length(articles)

    println("="^60)
    println("Wikipedia extraction performance")
    println("="^60)
    println("Number of extractions (articles): ", n_articles)
    println()

    opts = ProcessingOptions(domain = "wikipedia", max_length = 2048)
    # Use model = nothing to run discovery + relation matching only (avoids predict_tail_tokens MethodError)
    model = nothing

    total_time = 0.0
    total_entities = 0
    total_relations = 0
    per_article = Tuple{String, Float64, Int, Int}[]

    for (id, title, text) in articles
        t0 = time()
        kg = extract_knowledge_graph(text, model; options = opts)
        elapsed = time() - t0
        total_time += elapsed
        n_ent = length(kg.entities)
        n_rel = length(kg.relations)
        total_entities += n_ent
        total_relations += n_rel
        push!(per_article, (title, elapsed, n_ent, n_rel))
    end

    println("Results (discovery + relation matching only; tail prediction skipped):")
    for (name, elapsed, n_ent, n_rel) in per_article
        println("  ", name, ": ", round(elapsed, digits=3), " s, ", n_ent, " entities, ", n_rel, " relations")
    end
    println()
    println("Total: ", n_articles, " extractions in ", round(total_time, digits=3), " s")
    println("  Total entities: ", total_entities, " (avg ", round(total_entities / n_articles, digits=1), " per article)")
    println("  Total relations: ", total_relations, " (avg ", round(total_relations / n_articles, digits=1), " per article)")
    println("  Throughput: ", round(n_articles / total_time, digits=2), " articles/s")
    println("="^60)
end

main()
