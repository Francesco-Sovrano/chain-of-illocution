import os
import pandas as pd
import sys
import json

from lib.plot_n_stats import *

_, output_type, extra_questions_to_consider, fact_score_similarity_threshold = sys.argv
fact_score_similarity_threshold = float(fact_score_similarity_threshold)

topic_list = [
	'design_patterns',
	'java',
	'pharo',
	'python',
]

model_list = [
	'mistral:instruct',
	'mixtral:instruct',
	# 'llama3',
	'llama3:instruct',
	'llama3:70b-instruct',
	'gpt-3.5-turbo',
	'gpt-4o',
]

model_name_dict = {
	# f'norag_{output_type}': 'GenAI',
	f'yai_{0}_clause_{output_type}': 'RAG',
	f'yai_{extra_questions_to_consider}_clause_{output_type}': 'RAG+CoI',
}
# if 'gpt' in model:
# 	model_name_dict.update({
# 		# f'openai_assistants_all_{output_type}': 'RAG_old (OpenAI)',
# 		f'openai_assistants_{output_type}': 'RAG (OpenAI)',
# 	})

scores_to_consider_for_boxplot = [
	# 'faithfulness_fact_score',
	# 'faithfulness_fact_score_fuzzy',
	# 'faithfulness_supporting_facts',

	'fact_score',
	'fact_score_fuzzy',
	'supporting_facts',

	# 'dox_score',
	
	'explanation_words',
]

###########################

all_dataframe_list = []
for model in model_list:
	file_path_list = [
		os.path.join('results/',topic,model.replace(':','-'),f'data_{topic}_{output_type}_{extra_questions_to_consider}_{fact_score_similarity_threshold}.csv')
		for topic in topic_list
	]
	# Read each CSV file and append to the list
	dataframe_list = [
		pd.read_csv(file_path)
		for file_path in file_path_list
	]

	for topic, dataframe in zip(topic_list, dataframe_list):
		dataframe['topic'] = topic
		dataframe['llm'] = model.replace(':70b-instruct','-70b').replace(':instruct','')
	all_dataframe_list += dataframe_list

# Concatenate all dataframe_list
merged_df = pd.concat(all_dataframe_list, ignore_index=True)

output_file = os.path.join('results/',f'combined_data_all_{output_type}_{extra_questions_to_consider}_{fact_score_similarity_threshold}.csv')
# Write the merged dataframe to the output file
merged_df.to_csv(output_file, index=False)

# df = pd.read_csv(data_path)
data = [row.to_dict() for _, row in merged_df.iterrows()]

plots_path = os.path.join('results/',f'combined_plot_all_{output_type}_{extra_questions_to_consider}_{fact_score_similarity_threshold}.pdf')
make_boxplots(data, model_name_dict, scores_to_consider_for_boxplot, fact_score_similarity_threshold, plots_path, main_exp_label=model_name_dict[f'yai_{extra_questions_to_consider}_clause_{output_type}'])
