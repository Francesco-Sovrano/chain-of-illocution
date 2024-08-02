#!/bin/bash

# Activate virtual environment
source .env/bin/activate

FACT_SCORE_SIMILARITY_THRESHOLD=0.7
EXTRA_QUESTIONS_TO_CONSIDER=5

#################
### Mistral
python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mistral:instruct

#################
### Mixtral
python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mixtral:instruct

#################
### Llama3
python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:instruct

#################
### Llama3 70B
python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:70b-instruct

#################
### GPT-3.5
python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-3.5-turbo

#################
### GPT-4o
python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-4o

#################
### All
python3 combine_plots_and_llms.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD}
