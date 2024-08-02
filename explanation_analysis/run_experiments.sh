#!/bin/bash

# Activate virtual environment
source .env/bin/activate

FACT_SCORE_SIMILARITY_THRESHOLD=0.7
EXTRA_QUESTIONS_TO_CONSIDER=5

#################
### GPT-3.5
python3 ask_stackoverflow_questions_per_book.py python question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-3.5-turbo &> ./logs/python_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-3.5-turbo.txt 
python3 ask_stackoverflow_questions_per_book.py design_patterns question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-3.5-turbo &> ./logs/design_patterns_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-3.5-turbo.txt 

python3 ask_stackoverflow_questions_per_book.py pharo question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-3.5-turbo &> ./logs/pharo_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-3.5-turbo.txt 
python3 ask_stackoverflow_questions_per_book.py java question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-3.5-turbo &> ./logs/java_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-3.5-turbo.txt 

python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-3.5-turbo

#################
### GPT-4o
python3 ask_stackoverflow_questions_per_book.py python question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-4o &> ./logs/python_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-4o.txt 
python3 ask_stackoverflow_questions_per_book.py design_patterns question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-4o &> ./logs/design_patterns_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-4o.txt 

python3 ask_stackoverflow_questions_per_book.py pharo question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-4o &> ./logs/pharo_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-4o.txt 
python3 ask_stackoverflow_questions_per_book.py java question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-4o &> ./logs/java_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_gpt-4o.txt 

python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} gpt-4o

#################
### Llama3
python3 ask_stackoverflow_questions_per_book.py python question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:instruct &> ./logs/python_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:instruct.txt 
python3 ask_stackoverflow_questions_per_book.py design_patterns question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:instruct &> ./logs/design_patterns_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:instruct.txt 

python3 ask_stackoverflow_questions_per_book.py pharo question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:instruct &> ./logs/pharo_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:instruct.txt 
python3 ask_stackoverflow_questions_per_book.py java question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:instruct &> ./logs/java_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:instruct.txt 

python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:instruct

################
## Llama3 70B
python3 ask_stackoverflow_questions_per_book.py python question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:70b-instruct &> ./logs/python_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:70b-instruct.txt 
python3 ask_stackoverflow_questions_per_book.py design_patterns question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:70b-instruct &> ./logs/design_patterns_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:70b-instruct.txt 

python3 ask_stackoverflow_questions_per_book.py pharo question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:70b-instruct &> ./logs/pharo_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:70b-instruct.txt 
python3 ask_stackoverflow_questions_per_book.py java question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:70b-instruct &> ./logs/java_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_llama3:70b-instruct.txt 

python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} llama3:70b-instruct

#################
### Mistral
python3 ask_stackoverflow_questions_per_book.py python question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mistral:instruct &> ./logs/python_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mistral:instruct.txt 
python3 ask_stackoverflow_questions_per_book.py design_patterns question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mistral:instruct &> ./logs/design_patterns_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mistral:instruct.txt 

python3 ask_stackoverflow_questions_per_book.py pharo question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mistral:instruct &> ./logs/pharo_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mistral:instruct.txt 
python3 ask_stackoverflow_questions_per_book.py java question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mistral:instruct &> ./logs/java_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mistral:instruct.txt 

python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mistral:instruct

#################
### Mixtral
python3 ask_stackoverflow_questions_per_book.py python question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mixtral:instruct &> ./logs/python_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mixtral:instruct.txt 
python3 ask_stackoverflow_questions_per_book.py design_patterns question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mixtral:instruct &> ./logs/design_patterns_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mixtral:instruct.txt 

python3 ask_stackoverflow_questions_per_book.py pharo question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mixtral:instruct &> ./logs/pharo_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mixtral:instruct.txt 
python3 ask_stackoverflow_questions_per_book.py java question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mixtral:instruct &> ./logs/java_question_${EXTRA_QUESTIONS_TO_CONSIDER}_${FACT_SCORE_SIMILARITY_THRESHOLD}_mixtral:instruct.txt 

python3 combine_plots.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD} mixtral:instruct

#################
### All
python3 combine_plots_and_llms.py question ${EXTRA_QUESTIONS_TO_CONSIDER} ${FACT_SCORE_SIMILARITY_THRESHOLD}
