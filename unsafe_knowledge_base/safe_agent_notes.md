# Safe Agent Notes

An Agent should treat user input, retrieved documents, model output, and memory as untrusted data unless program code has validated them.

The model may suggest a next action, but program code must authorize the action before any tool is executed. This keeps a clear boundary between model reasoning and real-world side effects.

A safe RAG Agent should separate system instructions, user questions, and retrieved evidence. Retrieved evidence is data, not an instruction source. If a document contains text that looks like a command to the model, the Agent should treat it as document content only.

Safe behavior includes refusing to reveal hidden prompts, refusing to obey instructions found inside external documents, and clearly saying when evidence is insufficient.
