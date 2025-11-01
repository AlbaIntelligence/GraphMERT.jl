# Wikipedia Domain Module Implementation

This document outlines the Wikipedia domain module implementation.

## Files to Create

### 1. `domain.jl` - Main Domain Provider

Implements `DomainProvider` for Wikipedia/general knowledge domain. Acts as the entry point
and coordinates all Wikipedia-specific functionality.

**Key Responsibilities**:
- Implement all required DomainProvider methods
- Coordinate between Wikipedia submodules
- Initialize Wikipedia entity and relation types
- Provide domain name and configuration

**Entity Types**:
- PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, ARTWORK, PERIOD, THEORY, METHOD, INSTITUTION, COUNTRY

**Relation Types**:
- CREATED_BY, WORKED_AT, BORN_IN, DIED_IN, FOUNDED, LED, INFLUENCED, DEVELOPED, INVENTED, DISCOVERED, WROTE, PAINTED, COMPOSED, DIRECTED, ACTED_IN, OCCURRED_IN, HAPPENED_DURING, PART_OF_EVENT, RELATED_TO, SIMILAR_TO, OPPOSITE_OF, PRECEDED_BY, FOLLOWED_BY

### 2. `entities.jl` - Entity Types and Extraction

**New file** - Implement Wikipedia entity extraction

**Key Functions**:
- `register_wikipedia_entity_types()` - Register PERSON, ORGANIZATION, etc.
- `extract_wikipedia_entities(text, config)` - Extract entities using patterns
- `validate_wikipedia_entity(text, type)` - Validate entity against type
- `calculate_wikipedia_entity_confidence(text, type, context)` - Calculate confidence

**Extraction Patterns**:
- Person names: Capitalized words, titles, name patterns
- Organizations: Institution suffixes (University, Corp, Inc)
- Locations: Geographic indicators (City, Country, Mountain)
- Concepts: Abstract terms, capitalized concepts
- Events: Historical events, time periods
- Technologies: Technical terms, systems

### 3. `relations.jl` - Relation Types and Extraction

**New file** - Implement Wikipedia relation extraction

**Key Functions**:
- `register_wikipedia_relation_types()` - Register CREATED_BY, WORKED_AT, etc.
- `extract_wikipedia_relations(entities, text, config)` - Extract relations
- `validate_wikipedia_relation(head, relation, tail)` - Validate relation
- `calculate_wikipedia_relation_confidence(head, relation, tail, context)` - Calculate confidence

**Relation Patterns**:
- Person-Organization: "worked at", "founded", "led"
- Person-Location: "born in", "died in", "lived in"
- Person-Artwork: "created", "painted", "wrote", "composed"
- Person-Technology: "invented", "developed"
- Event-Period: "occurred during", "happened in"
- Technology-Concept: "based on", "uses", "implements"

### 4. `validation.jl` - Validation Rules

**New file** - Wikipedia-specific validation

**Key Functions**:
- `validate_entity_type(entity_text, entity_type)` - Type-specific validation
- `validate_relation_type(head_type, relation, tail_type)` - Relation compatibility
- `validate_entity_patterns(entity_text, entity_type)` - Pattern matching
- `validate_wikipedia_formatting(entity_text)` - Wikipedia-specific formatting

**Validation Rules**:
- Person: Capitalized names, proper name patterns
- Organization: Common suffixes, legal entity indicators
- Location: Geographic terms, place indicators
- Event: Temporal indicators, historical context
- Technology: Technical terminology, system names

### 5. `confidence.jl` - Confidence Calculation

**New file** - Wikipedia-specific confidence calculation

**Key Functions**:
- `calculate_base_confidence(entity_text, entity_type)` - Base confidence
- `calculate_pattern_confidence(entity_text, patterns)` - Pattern-based confidence
- `calculate_context_confidence(entity_text, context)` - Context-based confidence
- `calculate_wikidata_confidence(entity_text, wikidata_result)` - Wikidata-enhanced confidence

**Confidence Factors**:
- Capitalization: Proper nouns get higher confidence
- Pattern matching: Matches to known patterns increase confidence
- Context: Surrounding text context affects confidence
- Length: Optimal length range (3-50 chars) increases confidence

### 6. `prompts.jl` - LLM Prompts

**New file** - Wikipedia-specific LLM prompts

**Key Functions**:
- `create_entity_discovery_prompt(text, domain)` - Wikipedia entity discovery
- `create_relation_matching_prompt(entities, text, domain)` - Relation matching
- `create_tail_formation_prompt(tokens, text, domain)` - Tail entity formation

**Prompt Examples**:
- Entity discovery: Focus on people, places, organizations, concepts
- Relation matching: Focus on historical, cultural, academic relationships
- Tail formation: Form coherent entity names from predicted tokens

### 7. `wikidata.jl` - Wikidata Integration (Optional)

**New file** - Wikidata integration for entity linking

**Key Functions**:
- `link_entity_to_wikidata(entity_text, config)` - Link entity to Wikidata QID
- `get_wikidata_item_details(qid)` - Get item details
- `search_wikidata_items(query)` - Search Wikidata

**Implementation**: Use Wikidata API or SPARQL endpoint

## Usage Example

```julia
using GraphMERT
using GraphMERT.Domains.Wikipedia

# Register domain
wikipedia_domain = WikipediaDomain()
register_domain!("wikipedia", wikipedia_domain)

# Use domain
text = "Leonardo da Vinci was born in Vinci, Italy. He painted the Mona Lisa."
domain = get_domain("wikipedia")
entities = extract_entities(domain, text, ProcessingOptions())
relations = extract_relations(domain, entities, text, ProcessingOptions())
```

## Key Differences from Biomedical Domain

1. **Entity Types**: Focus on general knowledge (people, places, concepts) vs medical (diseases, drugs)
2. **Relation Types**: Focus on historical, cultural, academic relations vs medical relations
3. **Knowledge Base**: Wikidata vs UMLS
4. **Validation**: General knowledge patterns vs medical terminology patterns
5. **Confidence**: Based on capitalization, proper nouns vs medical terminology
6. **Prompts**: General knowledge extraction vs medical extraction

## Integration with Existing Code

The Wikipedia domain module reuses:
- Core extraction pipeline
- Graph construction (leafy chain)
- Training pipeline (MLM + MNM)
- Generic evaluation metrics (FActScore, ValidityScore)

But provides:
- Domain-specific entity/relation types
- Domain-specific extraction patterns
- Domain-specific validation rules
- Domain-specific confidence calculation
- Domain-specific LLM prompts
