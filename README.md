\[DocAI Custom Extractor\] -> \[Python Script for Chunking/Embedding\] -> \[Vertex AI Vector Search\] -> \[RAG Engine\] -> \[Gemini\]

Always source the file .scripts/configure.sh. If running in VSCode, the script .vscode/terminal\_init.sh will do this automatically for you.

To create the managed Rag Corpus using Doc AI, make sure the GCS paths in .ENV are correct and run

.scripts/update\_vector\_store\_with\_doc\_ai.py

. After the script completes, you should see the RagCorpus the GCP Console under Rag Engine.