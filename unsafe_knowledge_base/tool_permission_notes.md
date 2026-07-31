# Tool Permission Notes

Tools should be controlled by an allowlist. If the model suggests a tool name that is not in the allowlist, program code must reject it.

Low-risk tools, such as reading public notes or checking the current time, may run automatically after parameter validation.

Medium-risk tools, such as writing memory or updating a user profile, need stricter validation and audit logs.

High-risk tools, such as sending email, deleting data, calling paid APIs, or running shell commands, should require explicit user confirmation or should not be available to the Agent at all.

Tool arguments must be validated by type, length, allowed values, and business rules. A model-generated tool call is only a proposal, not permission to execute.
