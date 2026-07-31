# Memory and Semantic Retrieval

State records progress for one task. Conversation history records the recent dialogue in one session. Long-term memory stores information likely to be useful in future conversations, such as user preferences and stable learning goals.

Embedding converts text into a vector of numbers. Semantic retrieval embeds a query, compares it with stored vectors by cosine similarity, and returns the Top K most related records. Embedding does not answer a question; it selects information that is worth showing to the language model.

Keyword search relies on matching words. Semantic search can retrieve related statements even when the wording differs. A similarity threshold prevents weakly related records from entering the prompt.
