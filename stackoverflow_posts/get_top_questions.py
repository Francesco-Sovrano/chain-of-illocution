import math
import pandas as pd
import sys
import json

def top_viewed_titles(top_k, csv_file_path):
	# Load the CSV file into a DataFrame
	data = pd.read_csv(csv_file_path)

	# Ensure the 'ViewCount' column is treated as integers for correct sorting
	data['ViewCount'] = pd.to_numeric(data['ViewCount'], errors='coerce')

	# Sort the DataFrame by the 'ViewCount' column in descending order
	sorted_data = data.sort_values(by='ViewCount', ascending=False)

	# Get the top 35 titles with the highest 'ViewCount'
	top_titles = sorted_data['Title'].head(top_k)
	top_bodies = sorted_data['Body'].head(top_k)
	top_answers = sorted_data['AnswerBody'].head(top_k)

	# Convert the top titles from a Series to a list
	top_titles_list = top_titles.tolist()
	top_bodies_list = top_bodies.tolist()
	top_answers_list = top_answers.tolist()

	result_dict = {}
	for title, body, answer in zip(top_titles_list, top_bodies_list, top_answers_list):
		if not pd.notna(answer):  # Properly checks for NaN
			continue
		result_dict[title] = {
			'body': body,
			'answer': answer
		}
		# if title in valid_questions:  # Ensure title is in valid_questions
		# 	result_dict[title] = {
		# 		'body': body,
		# 		'answer': answer
		# 	}
		# else: print('Discarding:', title)
		# if title == valid_questions[-1]:
		# 	break
	return result_dict

# Replace 'your_file.csv' with the path to your CSV file
top_k = int(sys.argv[1])
csv_file_path = sys.argv[2]

# Print the top 35 titles
print(csv_file_path, json.dumps(top_viewed_titles(top_k, csv_file_path), indent=4))
