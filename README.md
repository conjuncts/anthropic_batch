# Batch Anthropic Compat

Archived due to https://docs.anthropic.com/en/docs/build-with-claude/batch-processing and https://www.anthropic.com/news/message-batches-api.


On Apr 15, 2024, OpenAI released their batch API with a 50% cost reduction. Their batch files are pretty convenient ways to bookkeep large number of requests and completions. This short script seeks to make already-constructed OpenAI-style batch files compatible and ingestible for the Anthropic models.

Note: an additional header line is provided in the jsonl which is not given in the original OpenAI batch API. This is configurable.

## Support

Currently, only simple synchronous text is supported
NOT supported: 
- vision
- tools
- streaming


## Alternatives

LiteLLM, etc.
But external libraries are not used here to keep dependencies minimal.
