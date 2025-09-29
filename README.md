# Data Science Pro

AI-agentic, LLM-augmented data science pipeline for CSV datasets. It performs EDA, preprocessing, modeling, evaluation, and reporting with minimal user code.

## Install

```bash
pip install -e .
# or when released
# pip install data-science-pro
```

Verify:

```python
import data_science_pro
from data_science_pro import DataSciencePro
```

## Quickstart (Python API)

Minimal end-to-end run that generates a markdown report:

```python
from data_science_pro import DataSciencePro

dsp = DataSciencePro(api_key="YOUR_OPENAI_KEY")  # or "" to run without LLM/RAG embeddings

# Optional guidance to the agent
dsp.controller.goal = "Predict churn with strong accuracy"
dsp.controller.user_target = "churn"  # optional; if omitted the agent will infer

final = dsp.run("/path/to/data.csv")
print(final.get("report", "No report produced"))
```

What happens:
- Loads your CSV, performs EDA and indexes summaries (RAG) for retrieval
- Plans and suggests next steps
- Selects/validates the target
- Iterates: preprocess → train → evaluate → critique → improve
- Stops on target metric or max iterations and produces a professional report

Notes:
- When `api_key` is empty or a test value, RAG uses a local mode and still runs.
- Set `dsp.controller.goal` (string) to guide the agent. Set `dsp.controller.user_target` (string) to force the target column.

## Beginner-Friendly Example

```python
from data_science_pro import DataSciencePro

dsp = DataSciencePro(api_key="")  # run without external LLM (RAG in local mode)
final = dsp.run("titanic.csv")
print(final["report"])  # Executive summary, EDA, preprocessing, modeling, metrics, recommendations
```

## Advanced Usage (Experienced DS)

Control the agent through state:

```python
from data_science_pro import DataSciencePro

dsp = DataSciencePro(api_key="YOUR_OPENAI_KEY")

# Configure agent objectives
dsp.controller.goal = "Reach accuracy >= 0.90 with interpretable model"
dsp.controller.user_target = None  # let the agent choose from candidates

# (Optional) adjust loop parameters directly by editing controller.run() defaults if needed
state = dsp.run("/data/training.csv")

# Inspect artifacts
print(state.get("analysis"))           # EDA summary
print(state.get("retrieved_context"))  # RAG snippets
print(state.get("evaluation"))         # metrics dict
print(state.get("history"))            # actions taken across iterations
```

## Key Concepts

- Agentic LangGraph
  - Orchestrator decides next step (analyze, preprocess, train, evaluate, report)
  - Planner proposes a concise plan; Critic flags quality issues
  - TargetSelector chooses target via user hint > LLM > heuristic

- RAG over EDA
  - Indexes column and dataset summaries (missingness, cardinality, basic stats)
  - Retrieves relevant context to ground suggestions and reports
  - Uses OpenAI embeddings when `api_key` is valid; falls back to local mode otherwise

- Reporting
  - Structured markdown report with executive summary, EDA highlights, preprocessing, modeling, metrics, and recommendations

## Troubleshooting

- No OpenAI key: set `api_key=""` to run without external calls; RAG runs in local mode.
- Target not detected: set `dsp.controller.user_target = "your_target"`.
- Healthcheck: run a local end-to-end check without network calls:

```bash
python corrected_imports.py
```

## FAQ

- Does it handle large CSVs? Yes, but consider sampling for initial iterations.
- Can I change the model? The agent selects sensible defaults; you can modify `Trainer` to register alternatives.
- Where is the final report? Returned in `final_state["report"]`.

## License

MIT

