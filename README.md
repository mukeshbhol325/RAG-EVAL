# RAG-EVAL

An evaluation framework for Retrieval-Augmented Generation (RAG) applications. It measures retrieval and answer quality against a golden dataset using RAGAS, DeepEval and G-Eval, with LangSmith tracing.

## Why this project

RAG apps can look fine in a demo and still hallucinate or retrieve the wrong context. This project treats a RAG pipeline like any other system under test: a fixed golden dataset, repeatable metrics, and traceable runs.

## Tech stack

Python, LangChain, ChromaDB, RAGAS, DeepEval, G-Eval, LangSmith, uv

## What it evaluates

| Area | Metrics |
|---|---|
| Retrieval | Context precision, context recall |
| Generation | Faithfulness, answer relevancy |
| Custom criteria | G-Eval (LLM-as-a-judge) rubrics, e.g. correctness, completeness |

TODO: adjust this table to the metrics you actually run.

## Project structure

```
RAG-EVAL/
├── src/            # RAG pipeline (ingestion, retrieval, generation)
├── RAGAS/          # RAGAS evaluation scripts
├── evals/          # DeepEval and G-Eval test cases
├── golden/         # Golden question-answer dataset
├── data/           # Source documents for the knowledge base
├── chroma_store/   # Local ChromaDB vector store
├── resources/      # Supporting assets
├── main.py         # Entry point
└── pyproject.toml  # Dependencies (managed with uv)
```

## Getting started

**Prerequisites:** Python (see `.python-version`), [uv](https://docs.astral.sh/uv/), an OpenAI API key (TODO: or your LLM provider), and a LangSmith API key.

```bash
git clone https://github.com/mukeshbhol325/RAG-EVAL.git
cd RAG-EVAL
uv sync
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=rag-eval
```

## Usage

```bash
# Run the RAG pipeline
uv run python main.py

# Run DeepEval / G-Eval tests
uv run deepeval test run evals/   # TODO: confirm file names

# Run RAGAS evaluation
uv run python RAGAS/<script>.py   # TODO: confirm script name
```

## Sample results

TODO: paste a small table of your scores, for example:

| Metric | Score |
|---|---|
| Faithfulness | 0.00 |
| Answer relevancy | 0.00 |
| Context precision | 0.00 |
| Context recall | 0.00 |

## Roadmap

- Run evals automatically in CI (Jenkins / GitHub Actions) and fail the build below a score threshold
- Add more golden test cases and adversarial prompts
- Compare retrievers and chunking strategies

## Author

**Mukesh Bhol**, Test Automation Engineer | LLM and RAG evaluation
[GitHub](https://github.com/mukeshbhol325) · [LinkedIn](TODO)
