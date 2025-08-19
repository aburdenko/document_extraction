# gemini_playground
Run . .scripts/configure.sh first.

You can ground Gemini with either a RAG Corpus or a Vector Search index.

If you want the fastest, most managed path to ground Gemini on your documents, use the RAG Engine and create a RAG Corpus.
If you have a custom data pipeline or already have your vector embeddings, use Vector Search directly and point the Gemini model to your index.
The tools in your project are well-designed to support both modern approaches.

Summary: Which "Search" to Use When?
Use Vertex AI Search (and its RAG Engine feature) when your primary goal is to quickly ground an LLM on your unstructured documents (like PDFs, TXT files, etc.) without worrying about the details of the data pipeline. This is the most direct and managed approach. Your create_vector_store_from_gcs.py script is the perfect example of this.

Use Vector Search directly when you have more advanced needs, such as:

You already have your own vector embeddings generated from a custom model.
You need a specific, custom document chunking strategy.
You are building a similarity search application that doesn't involve an LLM (e.g., product recommendation, image search).
In short, Vertex AI Search is the product, RAG Engine is the feature that makes RAG easy, and Vector Search is the powerful engine under the hood that makes it all possible.

./.scripts/create_vector_store_from_gcs.py \
  --corpus_display_name "agentspace_hcls_demo-corpus" \
  --source_gcs_bucket "agentspace_hcls_demo"

