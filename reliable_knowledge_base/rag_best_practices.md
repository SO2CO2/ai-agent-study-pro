# RAG Best Practices

RAG means Retrieval-Augmented Generation. A RAG Agent should retrieve relevant evidence before answering instead of relying only on the model's internal knowledge.

A good RAG answer should be grounded in retrieved evidence. If the retrieved evidence is insufficient, the Agent should say that the provided knowledge base does not contain enough information.

Source citation is important. The program should attach source names and chunk identifiers so a user can inspect where an answer came from.

Documents should be split into chunks with metadata such as source and chunk_index. Retrieval should return the Top K most relevant chunks. A fallback retrieval path can be useful when the primary retriever returns no evidence.

RAG quality can be evaluated by checking whether retrieval hits expected sources, whether evidence contains expected keywords, and whether the final answer stays faithful to the evidence.
