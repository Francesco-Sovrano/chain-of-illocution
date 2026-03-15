# Illocutionary Explanation Planning for Source-Faithful Explanations in Retrieval-Augmented Language Models

This replication package accompanies the paper **"Illocutionary Explanation Planning for Source-Faithful Explanations in Retrieval-Augmented Language Models"** by **Francesco Sovrano** and **Alberto Bacchelli**.

The paper studies **source-faithful natural-language explanations** in retrieval-augmented generation (RAG) for programming education. Using **90 Stack Overflow questions** grounded in **three programming textbooks** (Java, Python, and Pharo), it benchmarks six LLMs, measures textbook adherence with **FActScore** and related metrics, and evaluates a retrieval-time query-expansion strategy called **chain-of-illocution prompting (CoI)**. Across models, CoI improves source adherence substantially while preserving user-facing quality in a controlled user study.

## What is in this package

This package contains the code, data, and study material used to reproduce the three research questions in the paper.

### RQ1 — Source-faithfulness benchmark
The package includes the scripts used to evaluate how closely RAG-generated explanations adhere to the reference textbooks, as well as the corresponding non-RAG baselines.

### RQ2 — Chain-of-illocution prompting
The package includes the code used to build the implicit explanatory-question scaffold and to compare standard RAG against **RAG+CoI**.

### RQ3 — User study
The package includes the web interface used for the user study, the analysis script, and the generated PDF summaries for the GPT-3.5-turbo and GPT-4o study runs.

## Paper at a glance

The paper makes three main claims:

1. **Baseline source adherence is low.** Non-RAG models have a median source adherence of 0%, and standard RAG systems still show only modest textbook adherence.
2. **Illocutionary planning helps.** CoI retrieves evidence not only for the explicit question, but also for implicit explanatory questions, improving source faithfulness.
3. **User experience is preserved.** The user study found no statistically significant decrease in satisfaction, relevance, or perceived correctness for RAG+CoI compared with standard RAG.

## Repository structure

- `README.md`  
  This file.

- `setup.sh`  
  Top-level setup script that creates the virtual environments used by the experiment and user-study components.

- `explanation_analysis/`  
  Scripts and utilities for reproducing the RQ1/RQ2 experiments.
  - `ask_stackoverflow_questions_per_book.py`: main experiment driver.
  - `combine_plots.py`, `combine_plots_and_llms.py`: utilities to aggregate outputs and generate plots.
  - `config.py`: experiment configuration.
  - `lib/`: helper modules for chunking, prompting, linguistic processing, and plotting.

- `knowpy/`  
  Local Python package used by the experiment pipeline for knowledge extraction, retrieval, and supporting utilities.

- `stackoverflow_posts/`  
  Source datasets containing the top Stack Overflow questions used in the study.
  - `java.csv`
  - `python.csv`
  - `pharo.csv`

- `textbooks/`  
  Source textbooks used as the authoritative evidence base.
  - `textbooks/java/[2022]Introduction to Programming Using Java.pdf`
  - `textbooks/python/[2015]Think Python - How to Think Like a Computer Scientist.pdf`
  - `textbooks/pharo/[2018]Pharo By Example 5.pdf`

- `user_study/`  
  Material for reproducing or inspecting the user-study component.
  - `server.py`: local web server for the study interface.
  - `analyze_results.py`: analysis and plotting script.
  - `static/`: HTML/CSS/JS front-end and bundled CSV files used by the interface.
  - `results/`: destination directory for collected user-study responses.
  - `user_study_results_gpt-3.5-turbo.pdf`
  - `user_study_results_gpt-4o.pdf`

## System requirements

The package was prepared for **Linux** and **macOS** environments.

Recommended baseline:

- Python **3.10**
- A POSIX shell environment
- Enough disk space and RAM for textbook processing, embedding, and local-model execution

## External dependencies and services

Reproducing the full set of experiments requires access to:

- an **OpenAI API key**, used for embeddings and for the OpenAI-hosted models;
- a local **Ollama** installation if you want to reproduce the non-OpenAI model runs (`mistral`, `mixtral`, `llama3`, `llama3:70b`).

The setup scripts install Python dependencies, download the required spaCy model, and download the NLTK resources used by the pipeline.

## Setup

From the repository root, run:

```bash
./setup.sh
```

This creates separate virtual environments for:

- `explanation_analysis/.env`
- `user_study/.env`

## Configure credentials

Before running the experiment pipeline, export your OpenAI API key in your shell session:

```bash
export OPENAI_API_KEY="<your_api_key>"
```

If you plan to reproduce the local-model runs, make sure your Ollama server is running and the required models are available locally.

## Reproducing RQ1 and RQ2

The most direct entry point is:

```bash
cd explanation_analysis
source .env/bin/activate
python ask_stackoverflow_questions_per_book.py <book> question 5 0.7 <model>
```

where:

- `<book>` is one of `java`, `python`, or `pharo`
- `<model>` is one of:
  - `gpt-3.5-turbo`
  - `gpt-4o`
  - `llama3:instruct`
  - `llama3:70b-instruct`
  - `mistral:instruct`
  - `mixtral:instruct`

Example:

```bash
cd explanation_analysis
source .env/bin/activate
python ask_stackoverflow_questions_per_book.py java question 5 0.7 gpt-4o
```

The two key fixed parameters used in the paper are:

- `5` implicit explanatory questions for CoI
- `0.7` as the source-adherence similarity threshold

After individual runs, plots can be generated with:

```bash
python combine_plots.py question 5 0.7 gpt-4o
python combine_plots_and_llms.py question 5 0.7
```

A convenience batch script is also included at `explanation_analysis/run_experiments.sh`. For transparent reproduction, the per-book command above is the clearest starting point.

### Output locations

During execution, the experiment pipeline creates cache, log, and result files inside `explanation_analysis/`.

In particular, generated summaries and plots are written under the corresponding experiment output directories created by the scripts.

## Reproducing RQ3 (user study)

To launch the user-study interface locally:

```bash
cd user_study
source .env/bin/activate
python server.py <port> <model>
```

For example:

```bash
cd user_study
source .env/bin/activate
python server.py 8010 gpt-4o
```

Then open `http://localhost:<port>` in your browser.

Supported model arguments for the interface are:

- `gpt-4o`
- `gpt-3.5-turbo`

Collected study responses are written to `user_study/results/`.

To analyze collected responses and regenerate the study plots:

```bash
cd user_study
source .env/bin/activate
python analyze_results.py
```

The package also includes the generated report PDFs:

- `user_study/user_study_results_gpt-3.5-turbo.pdf`
- `user_study/user_study_results_gpt-4o.pdf`

## Notes on reproducibility

- The package contains the textbooks and Stack Overflow source data used in the study.
- The experiment code builds its own caches and derived artifacts when executed.
- Full reproduction requires the same family of models used in the paper and corresponding access to OpenAI and/or local Ollama models.
- Because LLM outputs are inherently non-deterministic, exact generations may vary, even though the paper uses constrained decoding settings.

## Citation

If you use this replication package, please cite the paper as:

**Sovrano, F., & Bacchelli, A. (2026). _Illocutionary Explanation Planning for Source-Faithful Explanations in Retrieval-Augmented Language Models_. In _Proceedings of the 4th World Conference on eXplainable Artificial Intelligence (xAI 2026)_. Springer, Communications in Computer and Information Science (CCIS).**

BibTeX:

```bibtex
@inproceedings{sovrano2026illocutionary,
  author    = {Francesco Sovrano and Alberto Bacchelli},
  title     = {Illocutionary Explanation Planning for Source-Faithful Explanations in Retrieval-Augmented Language Models},
  booktitle = {Proceedings of the 4th World Conference on eXplainable Artificial Intelligence (xAI 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

Conference and proceedings information for xAI 2026 is available on the official conference website: <https://xaiworldconference.com/2026/>.
