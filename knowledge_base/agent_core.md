# Agent Core Notes

Agent is more than a chat interface. It receives a goal, decides what action is needed, calls tools through program control, observes results, and returns a useful answer.

Tool Calling means that the model proposes a tool name and arguments. The model does not execute the tool by itself. Program code must validate the tool name and arguments, enforce permissions, execute the actual function, and return the result to the model.

A multi-step Agent maintains state for one task. A common loop is Plan, Act, Observe. The planner chooses a next action, code validates and executes it, and the observation is written back to state. A maximum step count and explicit stop conditions prevent infinite loops.
