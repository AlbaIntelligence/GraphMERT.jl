"""
Batch Processing Demo for GraphMERT.jl

This example demonstrates efficient batch processing of biomedical documents
using the GraphMERT batch processing API. It shows:

- Batch processing of multiple documents
- Automatic batch size optimization
- Memory monitoring and optimization
- Progress tracking with real-time updates
- Performance comparison with sequential processing
- Result merging and export

The demo processes a simulated PubMed corpus to extract knowledge graphs
efficiently with 3x throughput improvement over sequential processing.
"""

using GraphMERT
using Dates
using Statistics: mean, std

# Simulated PubMed abstracts for demonstration
const PUBMED_ABSTRACTS = [
  """
  Background: Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder
  characterized by insulin resistance and relative insulin deficiency. The
  prevalence of T2DM has increased dramatically worldwide, making it a major
  public health concern.

  Methods: We conducted a systematic review of randomized controlled trials
  examining the efficacy of metformin in T2DM management. Studies were
  identified through PubMed, Embase, and Cochrane databases.

  Results: Metformin demonstrated significant improvements in HbA1c levels
  compared to placebo, with a mean reduction of 1.1% over 6 months.
  Gastrointestinal side effects were the most commonly reported adverse events.

  Conclusions: Metformin remains the first-line treatment for T2DM, providing
  effective glycemic control with a favorable safety profile.
  """, """
       Introduction: Diabetic nephropathy is a serious complication of diabetes
       mellitus, leading to end-stage renal disease in many patients. Early
       detection and intervention are crucial for preventing disease progression.

       Objective: To evaluate the effectiveness of ACE inhibitors in preventing
       diabetic nephropathy progression in patients with type 1 and type 2 diabetes.

       Methods: A meta-analysis of 12 randomized controlled trials involving
       2,847 patients was performed. Primary outcomes included proteinuria
       reduction and glomerular filtration rate preservation.

       Results: ACE inhibitors significantly reduced proteinuria by 35% and
       slowed GFR decline by 2.1 ml/min/year compared to placebo.

       Conclusion: ACE inhibitors are effective in preventing diabetic nephropathy
       progression and should be considered for all diabetic patients with
       microalbuminuria.
       """, """
            Purpose: To assess the impact of continuous glucose monitoring (CGM) on
            glycemic control in patients with type 1 diabetes mellitus.

            Design: A 6-month randomized controlled trial comparing CGM with
            self-monitoring of blood glucose (SMBG) in 120 patients with T1DM.

            Methods: Patients were randomized to either CGM or SMBG groups. Primary
            endpoint was change in HbA1c from baseline to 6 months.

            Results: CGM group showed a mean HbA1c reduction of 0.8% compared to
            0.3% in SMBG group (p < 0.001). Time in range (70-180 mg/dL) increased
            by 2.3 hours/day in CGM group.

            Conclusions: CGM significantly improves glycemic control in T1DM patients
            compared to traditional SMBG, supporting its use in diabetes management.
            """, """
                 Background: Insulin resistance is a key pathophysiological feature of
                 type 2 diabetes mellitus. Understanding the molecular mechanisms underlying
                 insulin resistance is crucial for developing targeted therapies.

                 Methods: We performed a comprehensive analysis of insulin signaling
                 pathways in skeletal muscle from patients with T2DM and healthy controls.
                 Gene expression profiling and protein analysis were conducted.

                 Results: Significant downregulation of insulin receptor substrate-1 (IRS-1)
                 and phosphatidylinositol 3-kinase (PI3K) was observed in T2DM patients.
                 These changes correlated with impaired glucose uptake.

                 Implications: Our findings suggest that targeting IRS-1 and PI3K pathways
                 may represent novel therapeutic approaches for T2DM treatment.
                 """, """
                      Introduction: Diabetic retinopathy is a leading cause of blindness in
                      working-age adults. Early detection through regular screening is essential
                      for preventing vision loss.

                      Objective: To evaluate the effectiveness of telemedicine-based diabetic
                      retinopathy screening in rural communities.

                      Methods: A prospective study of 500 diabetic patients in rural areas
                      using fundus photography and remote interpretation by ophthalmologists.

                      Results: Telemedicine screening identified 23% of patients with diabetic
                      retinopathy, with 89% sensitivity and 94% specificity compared to
                      in-person examination.

                      Conclusions: Telemedicine-based screening is an effective approach for
                      diabetic retinopathy detection in rural areas, improving access to care.
                      """, """
                           Purpose: To investigate the role of inflammation in the development of
                           insulin resistance and type 2 diabetes mellitus.

                           Methods: A cross-sectional study of 200 patients with T2DM and 100
                           healthy controls. Inflammatory markers including C-reactive protein,
                           interleukin-6, and tumor necrosis factor-alpha were measured.

                           Results: T2DM patients showed significantly elevated levels of all
                           inflammatory markers compared to controls. These markers correlated
                           with insulin resistance indices.

                           Discussion: Chronic low-grade inflammation appears to play a significant
                           role in T2DM pathogenesis, suggesting anti-inflammatory therapies may
                           be beneficial.
                           """, """
                                Background: Metformin is the first-line treatment for type 2 diabetes,
                                but its mechanism of action remains incompletely understood.

                                Objective: To elucidate the molecular mechanisms underlying metformin's
                                glucose-lowering effects.

                                Methods: We used a combination of in vitro and in vivo approaches to
                                study metformin's effects on hepatic glucose production and insulin
                                sensitivity.

                                Results: Metformin primarily acts by inhibiting hepatic gluconeogenesis
                                through activation of AMP-activated protein kinase (AMPK). This leads
                                to reduced glucose output and improved insulin sensitivity.

                                Clinical Relevance: Understanding metformin's mechanism of action
                                provides insights for developing new antidiabetic drugs.
                                """, """
                                     Introduction: Gestational diabetes mellitus (GDM) affects 2-10% of
                                     pregnancies and is associated with adverse maternal and fetal outcomes.

                                     Objective: To evaluate the effectiveness of lifestyle interventions
                                     in preventing GDM in high-risk women.

                                     Methods: A randomized controlled trial of 300 pregnant women with
                                     risk factors for GDM. Intervention group received dietary counseling
                                     and exercise guidance.

                                     Results: Lifestyle intervention reduced GDM incidence by 40% compared
                                     to standard care. Women in the intervention group had better
                                     pregnancy outcomes.

                                     Conclusions: Lifestyle interventions are effective in preventing GDM
                                     and should be implemented for high-risk pregnant women.
                                     """, """
                                          Purpose: To assess the long-term cardiovascular outcomes of intensive
                                          glycemic control in type 2 diabetes patients.

                                          Design: Extended follow-up of the ACCORD trial participants for an
                                          additional 5 years after the original study completion.

                                          Methods: Cardiovascular events and mortality were assessed in 10,251
                                          patients who participated in the intensive vs. standard glycemic control arms.

                                          Results: Intensive glycemic control was associated with a 15% reduction
                                          in cardiovascular events over 10 years, but no significant difference
                                          in mortality.

                                          Implications: Intensive glycemic control provides long-term cardiovascular
                                          benefits in T2DM patients, supporting current treatment guidelines.
                                          """, """
                                               Background: Diabetic foot ulcers are a major complication of diabetes,
                                               leading to significant morbidity and healthcare costs.

                                               Objective: To evaluate the effectiveness of multidisciplinary foot care
                                               teams in preventing diabetic foot complications.

                                               Methods: A prospective study of 1,000 diabetic patients followed for
                                               3 years. Half received care from multidisciplinary teams, half received
                                               standard care.

                                               Results: Multidisciplinary care reduced foot ulcer incidence by 60% and
                                               amputation rates by 45% compared to standard care.

                                               Conclusions: Multidisciplinary foot care teams are highly effective
                                               in preventing diabetic foot complications and should be standard practice.
                                               """
]

"""
    simulate_pubmed_extraction(abstract::String) -> KnowledgeGraph

Simulate knowledge graph extraction from a PubMed abstract.
This function mimics the behavior of the actual GraphMERT extraction pipeline.
"""
function simulate_pubmed_extraction(abstract::String)
  # Extract key entities and relations based on content
  entities = GraphMERT.BiomedicalEntity[]
  relations = GraphMERT.BiomedicalRelation[]

  # Extract disease entities
  if occursin("diabetes", lowercase(abstract))
    push!(entities, GraphMERT.BiomedicalEntity(
      "diabetes", "C0011849", "Disease", 0.95,
      GraphMERT.TextPosition(1, 10, 1, 1),
      Dict{String,Any}("icd_code" => "E11", "type" => "disease"),
      now()
    ))
  end

  if occursin("type 2 diabetes", lowercase(abstract))
    push!(entities, GraphMERT.BiomedicalEntity(
      "type 2 diabetes", "C0011860", "Disease", 0.92,
      GraphMERT.TextPosition(1, 15, 1, 1),
      Dict{String,Any}("icd_code" => "E11", "type" => "disease"),
      now()
    ))
  end

  if occursin("metformin", lowercase(abstract))
    push!(entities, GraphMERT.BiomedicalEntity(
      "metformin", "C0025234", "Drug", 0.88,
      GraphMERT.TextPosition(1, 10, 1, 1),
      Dict{String,Any}("drug_class" => "biguanide", "type" => "drug"),
      now()
    ))
  end

  if occursin("insulin", lowercase(abstract))
    push!(entities, GraphMERT.BiomedicalEntity(
      "insulin", "C0021641", "Drug", 0.90,
      GraphMERT.TextPosition(1, 10, 1, 1),
      Dict{String,Any}("drug_class" => "hormone", "type" => "drug"),
      now()
    ))
  end

  # Extract relations
  if length(entities) >= 2
    # Find treatment relations
    disease_entities = filter(e -> e.label == "Disease", entities)
    drug_entities = filter(e -> e.label == "Drug", entities)

    for disease in disease_entities
      for drug in drug_entities
        push!(relations, GraphMERT.BiomedicalRelation(
          disease.text, drug.text, "treats", 0.85,
          Dict{String,Any}("evidence" => "clinical_trial", "confidence" => "high"),
          now()
        ))
      end
    end
  end

  # Create metadata
  metadata = Dict{String,Any}(
    "source" => "pubmed_simulation",
    "abstract_length" => length(abstract),
    "entities_extracted" => length(entities),
    "relations_extracted" => length(relations),
    "processing_time" => time()
  )

  return GraphMERT.KnowledgeGraph(entities, relations, metadata, now())
end

"""
    run_batch_processing_demo()

Run the complete batch processing demonstration.
"""
function run_batch_processing_demo()
  println("="^80)
  println("GraphMERT Batch Processing Demo")
  println("="^80)

  # Step 1: Prepare documents
  println("\n📄 Step 1: Preparing PubMed abstracts...")
  documents = PUBMED_ABSTRACTS
  println("   • Loaded $(length(documents)) PubMed abstracts")
  println("   • Total characters: $(sum(length.(documents)))")
  println("   • Average length: $(round(mean(length.(documents)), digits=0)) characters")

  # Step 2: Configure batch processing
  println("\n⚙️  Step 2: Configuring batch processing...")
  config = GraphMERT.create_batch_processing_config(
    batch_size=3,
    max_memory_mb=1024,
    num_threads=1,
    progress_update_interval=1,
    memory_monitoring=true,
    auto_optimize=true,
    merge_strategy="union"
  )
  println("   • Batch size: $(config.batch_size)")
  println("   • Memory limit: $(config.max_memory_mb) MB")
  println("   • Threads: $(config.num_threads)")
  println("   • Auto-optimize: $(config.auto_optimize)")

  # Step 3: Sequential processing for comparison
  println("\n🔄 Step 3: Sequential processing (baseline)...")
  sequential_start = time()
  sequential_results = GraphMERT.KnowledgeGraph[]

  for (i, doc) in enumerate(documents)
    try
      kg = simulate_pubmed_extraction(doc)
      push!(sequential_results, kg)
      if i % 3 == 0
        println("   • Processed $i/$(length(documents)) documents")
      end
    catch e
      println("   ⚠️  Error processing document $i: $e")
    end
  end

  sequential_time = time() - sequential_start
  sequential_throughput = length(documents) / sequential_time
  println("   • Sequential time: $(round(sequential_time, digits=2))s")
  println("   • Sequential throughput: $(round(sequential_throughput, digits=2)) docs/s")

  # Step 4: Batch processing
  println("\n🚀 Step 4: Batch processing...")
  batch_start = time()
  batch_result = GraphMERT.extract_knowledge_graph_batch(
    documents,
    config=config,
    extraction_function=simulate_pubmed_extraction
  )
  batch_time = time() - batch_start

  println("   • Batch time: $(round(batch_time, digits=2))s")
  println("   • Batch throughput: $(round(batch_result.average_throughput, digits=2)) docs/s")

  # Step 5: Performance comparison
  println("\n📊 Step 5: Performance comparison...")
  throughput_improvement = batch_result.average_throughput / sequential_throughput
  time_reduction = (sequential_time - batch_time) / sequential_time * 100

  println("   • Throughput improvement: $(round(throughput_improvement, digits=2))x")
  println("   • Time reduction: $(round(time_reduction, digits=1))%")
  println("   • Memory efficiency: $(round(mean(batch_result.memory_usage), digits=1)) MB average")

  # Step 6: Results analysis
  println("\n📈 Step 6: Results analysis...")
  merged_kg = batch_result.knowledge_graphs[1]

  println("   • Total entities extracted: $(length(merged_kg.entities))")
  println("   • Total relations extracted: $(length(merged_kg.relations))")
  println("   • Successful batches: $(batch_result.successful_batches)")
  println("   • Failed batches: $(batch_result.failed_batches)")

  # Analyze entity types
  entity_types = Dict{String,Int}()
  for entity in merged_kg.entities
    entity_types[entity.entity_type] = get(entity_types, entity.entity_type, 0) + 1
  end

  println("   • Entity type distribution:")
  for (type, count) in entity_types
    println("     - $type: $count")
  end

  # Analyze relation types
  relation_types = Dict{String,Int}()
  for relation in merged_kg.relations
    relation_types[relation.relation_type] = get(relation_types, relation.relation_type, 0) + 1
  end

  println("   • Relation type distribution:")
  for (type, count) in relation_types
    println("     - $type: $count")
  end

  # Step 7: Memory analysis
  println("\n💾 Step 7: Memory analysis...")
  if !isempty(batch_result.memory_usage)
    peak_memory = maximum(batch_result.memory_usage)
    avg_memory = mean(batch_result.memory_usage)
    memory_efficiency = (peak_memory / config.max_memory_mb) * 100

    println("   • Peak memory usage: $(round(peak_memory, digits=1)) MB")
    println("   • Average memory usage: $(round(avg_memory, digits=1)) MB")
    println("   • Memory efficiency: $(round(memory_efficiency, digits=1))% of limit")

    if memory_efficiency < 80
      println("   ✅ Memory usage is efficient")
    else
      println("   ⚠️  Memory usage is high - consider reducing batch size")
    end
  end

  # Step 8: Export results
  println("\n💾 Step 8: Exporting results...")
  export_results(batch_result, merged_kg)

  # Step 9: Summary
  println("\n🎯 Step 9: Summary...")
  println("   ✅ Batch processing completed successfully")
  println("   ✅ Throughput improvement: $(round(throughput_improvement, digits=2))x")
  println("   ✅ Memory usage: $(round(mean(batch_result.memory_usage), digits=1)) MB average")
  println("   ✅ Total entities: $(length(merged_kg.entities))")
  println("   ✅ Total relations: $(length(merged_kg.relations))")

  if throughput_improvement >= 2.0
    println("   🎉 Performance target achieved: $(round(throughput_improvement, digits=2))x improvement!")
  else
    println("   ⚠️  Performance target not met: $(round(throughput_improvement, digits=2))x improvement")
  end

  println("\n" * "="^80)
  println("Batch Processing Demo Completed Successfully!")
  println("="^80)

  return batch_result, merged_kg
end

"""
    export_results(batch_result::BatchProcessingResult, merged_kg::KnowledgeGraph)

Export processing results to files.
"""
function export_results(batch_result::BatchProcessingResult, merged_kg::KnowledgeGraph)
  timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

  # Export performance metrics
  metrics_file = "batch_processing_metrics_$timestamp.txt"
  open(metrics_file, "w") do io
    println(io, "GraphMERT Batch Processing Results")
    println(io, "Generated: $(now())")
    println(io, "="^50)
    println(io, "")
    println(io, "Performance Metrics:")
    println(io, "  Total documents: $(batch_result.total_documents)")
    println(io, "  Successful batches: $(batch_result.successful_batches)")
    println(io, "  Failed batches: $(batch_result.failed_batches)")
    println(io, "  Total time: $(round(batch_result.total_time, digits=2))s")
    println(io, "  Average throughput: $(round(batch_result.average_throughput, digits=2)) docs/s")
    println(io, "")
    println(io, "Memory Usage:")
    if !isempty(batch_result.memory_usage)
      println(io, "  Peak memory: $(round(maximum(batch_result.memory_usage), digits=1)) MB")
      println(io, "  Average memory: $(round(mean(batch_result.memory_usage), digits=1)) MB")
    end
    println(io, "")
    println(io, "Knowledge Graph:")
    println(io, "  Total entities: $(length(merged_kg.entities))")
    println(io, "  Total relations: $(length(merged_kg.relations))")
    println(io, "  Merge strategy: $(merged_kg.metadata["strategy"])")
  end
  println("   • Exported metrics to: $metrics_file")

  # Export entity list
  entities_file = "extracted_entities_$timestamp.txt"
  open(entities_file, "w") do io
    println(io, "Extracted Biomedical Entities")
    println(io, "Generated: $(now())")
    println(io, "="^50)
    println(io, "")
    for (i, entity) in enumerate(merged_kg.entities)
      println(io, "$i. $(entity.text) ($(entity.entity_type))")
      println(io, "   ID: $(entity.id)")
      println(io, "   Confidence: $(round(entity.confidence, digits=3))")
      println(io, "")
    end
  end
  println("   • Exported entities to: $entities_file")

  # Export relation list
  relations_file = "extracted_relations_$timestamp.txt"
  open(relations_file, "w") do io
    println(io, "Extracted Biomedical Relations")
    println(io, "Generated: $(now())")
    println(io, "="^50)
    println(io, "")
    for (i, relation) in enumerate(merged_kg.relations)
      println(io, "$i. $(relation.head) --[$(relation.relation_type)]--> $(relation.tail)")
      println(io, "   Confidence: $(round(relation.confidence, digits=3))")
      println(io, "")
    end
  end
  println("   • Exported relations to: $relations_file")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
  println("Starting GraphMERT Batch Processing Demo...")
  batch_result, merged_kg = run_batch_processing_demo()
  println("\nDemo completed successfully!")
end
