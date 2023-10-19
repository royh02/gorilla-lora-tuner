# API Data Bank

#### In this dataset, we store some example prompts used to perform few-shot prompting on GPT-4 to generate synthetic instruction-API pairs that will be fed into the finetuning process.

## Examples

The examples inside `prompts/examples.json` will be used as in-context examples. They are split into 3 big categories as of now:

1. ML APIs (Huggingface, Torchhub, Tensorflow, etc.)
2. Rest APIs (AWS)
3. Command Line commands (conda, git, etc.)