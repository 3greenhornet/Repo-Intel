# Architecture

See README.md (top-level) for the phase diagram and the component map.
Component interfaces follow §3.2 of the implementation documentation:
IngestionComponent, ParsingComponent, AnalysisComponent,
KnowledgeBaseComponent, RetrievalComponent, AgentComponent,
EvidenceComponent, ReasoningComponent — each with a concrete class in
backend/app/core/**.

Adapter layers (embedded default → production switch):

- graph: NetworkXGraphStore → Neo4jGraphStore (GRAPH_STORE=neo4j)
- relational: SQLiteDatabase → PostgreSQL (RELATIONAL_STORE=postgres, §15.1 DDL)
- cache: InMemoryTokenBucket → RedisTokenBucket (CACHE_BACKEND=redis)
- vector: NumpyFlatIndex → FaissFlatIndex / Pinecone (VECTOR_STORE)
- LLM: ZaiProvider → OpenAIProvider / AnthropicProvider (LLM_PROVIDER)
