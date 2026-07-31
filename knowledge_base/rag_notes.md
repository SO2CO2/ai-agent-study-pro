# RAG Notes

RAG means Retrieval-Augmented Generation. It answers a question by first retrieving relevant evidence from an external knowledge base and then asking a language model to answer from that evidence.

The index stage loads documents, splits them into chunks, generates an embedding for every chunk, and stores chunk content together with metadata such as source and chunk index. Documents should not be embedded as one huge block because a question usually needs only a small part of a document.

Chunk overlap repeats a small amount of text across neighboring chunks. It reduces the chance that an explanation is separated from its context at a chunk boundary.

At query time, RAG embeds the user question, retrieves Top K relevant chunks, and injects only those chunks into the answer prompt. The answer should state when the retrieved evidence is insufficient. The program should attach source and chunk identifiers so an answer can be checked against the original material.
