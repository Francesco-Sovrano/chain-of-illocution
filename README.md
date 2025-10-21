# Chain-of-Illocution Prompting Elicits Textbook Adherence in Retrieval-Augmented Generation

Welcome to the replication package of the paper titled "Chain-of-Illocution Prompting Elicits Textbook Adherence in Retrieval-Augmented Generation".

## Abstract

Well-crafted curricula are key to effective learning, with textbooks serving as the backbone of modern education by providing stable, peer-reviewed knowledge. Yet, generative AI is rapidly transforming software education with personalized, on-demand responses. Even with Retrieval-Augmented Generative AI (RAG) directly incorporating textbook content, maintaining strict adherence to these sources remains challenging. We found that state-of-the-art RAG systems too often fail to maintain the integrity of textbook material, potentially compromising educational quality. To address this, we introduce a chain-of-illocution prompting strategy based on Achinstein’s theory of explanations to enhance RAG's adherence. Our method shifts macro-planning control to the retrieval system, ensuring AI-generated answers remain faithful to the textbook. Experiments with leading RAG models (GPT-4o, Llama3, Mixtral) show up to 63% increase in textbook adherence (p < 0.05) across 90 popular Stack Overflow questions on Python, Java, and Pharo. A user study with 200+ developers also confirms that our approach improves adherence without sacrificing output quality. 


## Repository Contents

This repository comprises various tools, scripts, and data essential for replicating the findings of our ICSE 2025 paper. Here's a detailed breakdown:

1. **explanation_analysis**
   - Scripts for running experiments the experiments of RQ1 & RQ2.

2. **knowpy**:
   - Python package used by the experiments of RQ1 & RQ2.

3. **stackoverflow_posts**:
   - CSV files with the top-200 most viewed Stack Overflow posts with tags: `java`, `pharo`, `python`.

4. **textbooks**:
   - Folders containing the textbooks:
		- [Introduction to Programming Using Java](textbooks/java/[2022]Introduction to Programming Using Java.pdf)
		- [Think Python: How to Think Like a Computer Scientist](textbooks/python/[2015]Think Python - How to Think Like a Computer Scientist.pdf)
		- [Pharo By Example 5](textbooks/pharo/[2018]Pharo By Example 5.pdf)

5. **user_study**:
   - Scripts for running the user interface of the study.
   - Results of the user studies, including detailed PDF reports (`user_study_results_gpt-3.5-turbo.pdf`, `user_study_results_gpt-4o.pdf`).


## System Specifications

This repository is tested and recommended on:

- OS: Linux (Debian 5.10.179 or newer) and macOS (14.4.1 Sonoma or newer)
- Python version: 3.9


## Environment Setup

In order to run the automated assessments, you need to install a proper virtual environment, running the following script:

```bash
./setup.sh
```

## Installation of OpenAI Keys

To use this package, you must set up two environment variables: `OPENAI_ORGANIZATION` and `OPENAI_API_KEY`. These variables represent your OpenAI organization identifier and your API key respectively.

On UNIX-like Operating Systems (Linux, MacOS):
1. Open your terminal.
2. To set the `OPENAI_ORGANIZATION` variable, run:
   ```bash
   export OPENAI_ORGANIZATION='your_organization_id'
   ```
3. To set the `OPENAI_API_KEY` variable, run:
   ```bash
   export OPENAI_API_KEY='your_api_key'
   ```
4. These commands will set the environment variables for your current session. If you want to make them permanent, you can add the above lines to your shell profile (`~/.bashrc`, `~/.bash_profile`, `~/.zshrc`, etc.)

To ensure you've set up the environment variables correctly:

1. In your terminal or command prompt, run:
   ```bash
   echo $OPENAI_ORGANIZATION
   ```
   This should display your organization ID.
   
2. Similarly, verify the API key:
   ```bash
   echo $OPENAI_API_KEY
   ```

Ensure that both values match what you've set.


## RQ1 & RQ2: Run the Experiments

After setting up the environment, you can run the experiments of RQ1 and RQ2 using:

```bash
cd explanation_analysis
./run_experiments.sh
cd ..
```

In our study, we evaluated six state-of-the-art, RAG-enhanced LLMs: GPT-3.5-turbo, GPT-4o, LLama3 8B and 70B, Mistral, and Mixtral 8x7B. 

The experiment will be automatically run on all the textbooks:
- [Introduction to Programming Using Java](textbooks/java/[2022]Introduction to Programming Using Java.pdf)
- [Think Python: How to Think Like a Computer Scientist](textbooks/python/[2015]Think Python - How to Think Like a Computer Scientist.pdf)
- [Pharo By Example 5](textbooks/pharo/[2018]Pharo By Example 5.pdf)

Since we [cached](explanation_analysis/cache) all the outputs of all the six considered LLMs, running the script won't take much time.

Upon completion of the script, you will find the assessment results in the directory [explanation_analysis/results](explanation_analysis/results). This directory contains CSV and PDF files detailing the outcomes of the assessments.

**Note:** To increase script verbosity and manage log output, please remove the comments from lines 23 to 29 in the file located at [explanation_analysis/ask_stackoverflow_questions_per_book.py](explanation_analysis/ask_stackoverflow_questions_per_book.py).


## RQ3: User Study

After setting up the environment, you can run the user interface of the study we conducted to answer RQ3:

```bash
cd user_study
. .env/bin/activate
python server.py [port] [model]
cd ..
```
The user interface will then be accessible with a browser on `localhost:[port]`.

Replace `[port]` with the port number (e.g., `8010`) and `[model]` with either `gpt-4o` or `gpt-3.5-turbo`.

The data collected from the users can be found inside the directory [user_study/results](user_study/results).

The logic of the user interface instead can be found inside the directory [user_study/static](user_study/static).

By default the user interface show the explanations generated by GPT-4o only. Edit the file [user_study/static/js/app.js](user_study/static/js/app.js) to make the user interface show the explanations generated by GPT-3.5-turbo instead. Go to line 34 and change the string `combined_data_gpt-4o_question_5_0.7.csv` with `combined_data_gpt-3.5-turbo_question_5_0.7.csv`.

The script [user_study/analyze_results.py](user_study/analyze_results.py) can be ran to analyze the results in [user_study/results](user_study/results) and generate the box plots which can be found inside: [user_study/user_study_results_gpt-3.5-turbo.pdf](user_study/user_study_results_gpt-3.5-turbo.pdf) and [user_study/user_study_results_gpt-4o.pdf](user_study/user_study_results_gpt-4o.pdf)

