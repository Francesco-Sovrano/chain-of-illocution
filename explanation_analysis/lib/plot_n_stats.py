import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import textwrap  # Import textwrap for wrapping text
import numpy as np
import json

plt.rcParams.update({'font.size': 14})  # You can adjust the size as needed

def cliffs_delta(x, y):
	n = len(x)
	m = len(y)
	ranksum = sum([sum([1 for j in y if j < i]) - sum([1 for j in y if j > i]) for i in x])
	return ranksum / (n * m)

def cohens_dz(z, x, y, zero_method='wilcox'):
	if zero_method == 'wilcox':
		n = len(x)              # total number of pairs, including zeros
	else:
		# Compute number of nonzero differences
		diffs = x - y
		n = np.count_nonzero(diffs)   # Pratt’s zeros are automatically excluded

	# Compute dz
	dz = z / np.sqrt(n)
	return dz

def confidence_intervals(zstatistic, x, y):
	# Analytical CI for Cohen's dz
	dz = cohens_dz(zstatistic, x, y)
	n = len(x)
	se_dz = np.sqrt(1/n + dz**2/(2*n))
	zcrit = stats.norm.ppf(0.975)
	return (dz - zcrit * se_dz, dz + zcrit * se_dz)

# Calculate matched-pairs rank-biserial correlation
def rank_biserial_correlation(U, n1, n2):
	return 1 - (2*U/(n1*n2))

# Function to prepare DataFrame for a given model and score type
def prepare_data_for_model(data, model_name, score_to_consider):
	return pd.DataFrame([
		{
			'QTitle': data_dict['title'],
			'Score': data_dict[score_to_consider],
		}
		for data_dict in data
		if data_dict['tool'] == model_name
	])

def draw_bracket(ax, x1, x2, y, h, text):
	"""Draws a curly bracket to show statistical comparison."""
	ax.annotate('', xy=(x1, y+h), xytext=(x2, y+h), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.5'.format(abs(x2-x1)/2), lw=1.5, color='black'))
	ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', color='black')

def make_boxplots(data, model_name_dict, scores_to_consider_for_boxplot, fact_score_similarity_threshold, plot_path, main_exp_label='RAG+CoR'):
	grouped_data = {llm: [row.to_dict() for _, row in group.iterrows()] for llm, group in pd.DataFrame(data).groupby('llm')}
	
	num_llms = len(grouped_data)
	plt.figure(figsize=(3 * len(scores_to_consider_for_boxplot), 2.3 * num_llms))

	# Load the default color palette
	palette = sns.color_palette()
	
	for llm_index, (llm, llm_data) in enumerate(grouped_data.items()):
		for score_index, score in enumerate(scores_to_consider_for_boxplot):
			ax = plt.subplot(num_llms, len(scores_to_consider_for_boxplot), llm_index * len(scores_to_consider_for_boxplot) + score_index + 1)
			scores_dict = {v: prepare_data_for_model(llm_data, v, score) for v in model_name_dict.values()}
			comparison_df = pd.DataFrame({k: v['Score'].values for k, v in scores_dict.items()}, index=scores_dict[next(iter(model_name_dict.values()))]['QTitle'])
			
			sns.boxplot(data=comparison_df, palette=palette)

			if llm_index==0:
				plt.title(score.replace('fact_score_fuzzy','Semantic Similarity').replace('fact_score','Adherence Precision').replace('supporting_facts','# Adherent Clauses').replace('explanation_words','# Explanation Words').replace('_', ' ').replace('dox ','DoX '))
			
			if llm_index==num_llms-1:
				# Rotate x-labels for better visibility
				# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

				# Alternatively, auto-wrap text
				ax.set_xticklabels([textwrap.fill(label.get_text(), width=10) for label in ax.get_xticklabels()])
			else:
				ax.set_xticklabels([])

			if score_index == 0:
				ax.set_ylabel(llm+'\n'+ax.get_ylabel())  # Set llm label as y-axis label on the first column only


			offset_increment = 0.02    # Additional space for each new line of annotation
			
			# Annotate medians
			medians = comparison_df.median()
			for i, median in enumerate(medians):
				plt.text(i, median + offset_increment, f"{median:.3f}".rstrip('0').rstrip('.'), horizontalalignment='center', size='small', color='black', weight='semibold')
			
			# Adjust bracket height
			if score == 'explanation_words':
				continue
			annotation_y_offset = 0.75  # Starting offset just above the subplot

			# # is_normally_distributed = True
			# for model in model_name_dict.values():
			# 	print(f"Testing for {score}:")
			# 	# Testing normality for each group
			# 	stat, p_value = stats.shapiro(comparison_df[model])
			# 	print(f"{model} - Shapiro Test p-value: {p_value:.4f}")
			# 	# if p_value < 0.05:
			# 	# 	is_normally_distributed = False

			for i,model in enumerate(filter(lambda x: x!=main_exp_label, model_name_dict.values())):
				if model == 'GenAI':
					continue
				x = comparison_df[main_exp_label]
				y = comparison_df[model]

				differences = np.array(x) - np.array(y)
				abs_differences = np.abs(differences)

				ranks = stats.rankdata(abs_differences)
				pos_rank_sum = np.sum(ranks[differences > 0])
				neg_rank_sum = np.sum(ranks[differences < 0])

				t = min(pos_rank_sum, neg_rank_sum)

				statistic, p_value = stats.wilcoxon(x, y, alternative='greater') #if score != 'fact_score_fuzzy' else stats.ttest_rel(x, y, alternative='greater')
				zstatistic = abs(stats.norm.ppf(p_value/2))

				effect_size = {
					'rank_biserial_correlation': rank_biserial_correlation(statistic, len(x), len(y)),
					'cliffs_delta': cliffs_delta(x, y),
					'cohens_dz': cohens_dz(zstatistic, x, y),
					'confidence_intervals': confidence_intervals(zstatistic, x, y),
				}

				print(f"{main_exp_label} > {model} comparison for {score}:")
				print(f"    Wilcoxon signed-rank test statistic (T): {statistic}")
				print(f"    Smaller Rank Sum (t): {t}")
				print(f"    P-value: {p_value}")
				print(f"    Effect size: {json.dumps(effect_size, indent=4)}")

				# Format p_value based on its magnitude
				if p_value < 0.001:
					p_text = "p<0.001"
				elif p_value > 0.999:
					p_text = "p>0.999"
				else:
					p_text = f"p={p_value:.3f}"

				# Make p_value bold if it's less than 0.05
				if p_value < 0.05:
					p_text = f"$\\bf{{{p_text}}}$"  # Use LaTeX for bold

				# Format the full annotation text
				# annotation_text = f"{p_text}\nT={statistic:.0f}\nΔ={effect_size['cliffs_delta']:.3f}"
				ci_low, ci_high = effect_size['confidence_intervals']
				annotation_text = f"{p_text}\nCI=[{ci_low:.2f}, {ci_high:.1f}]\ndz={effect_size['cohens_dz']:.2f}"
				# annotation_text = f"{main_exp_label} vs. {model}: T={statistic:.0f}, p={p_value:.3f}, Δ={effect_size['cliffs_delta']:.3f}"
				# Position text at the top of each subplot, incrementing vertically for each model
				plt.text(0.5, annotation_y_offset, annotation_text, ha='center', va='bottom', size='small', transform=ax.transAxes, bbox=dict(
					facecolor='white',   # background color
					alpha=0.75,           # transparency, 0.0 (invisible) to 1.0 (opaque)
					edgecolor='none'     # no border; set to e.g. 'black' if you want an outline
				))
				annotation_y_offset += offset_increment  # Move up for the next line

	plt.tight_layout()
	plt.subplots_adjust(wspace=0.3, hspace=0.1)  # adjust the horizontal and vertical spaces
	plt.savefig(plot_path)

def make_boxplot(data, model_name_dict, scores_to_consider_for_boxplot, fact_score_similarity_threshold, plot_path, main_exp_label='RAG+CoR'):

	plt.figure(figsize=(10 * len(scores_to_consider_for_boxplot), 10))

	for index, score in enumerate(scores_to_consider_for_boxplot):
		# Create horizontal subplots
		plt.subplot(1, len(scores_to_consider_for_boxplot), index + 1)

		scores_dict = {
			v: prepare_data_for_model(data, v, score)
			for v in model_name_dict.values()
		}
		comparison_df = pd.DataFrame({
			k: v['Score'].values
			for k,v in scores_dict.items()
		}, index=scores_dict[main_exp_label]['QTitle'])
		
		ax = sns.boxplot(data=comparison_df)
		# plt.title(f'Summary statistics of {score.replace("_", " ")} for {main_exp_label} vs ChatGPT')
		plt.ylabel(score.replace('fact_score_fuzzy','Semantic Similarity').replace('fact_score','Source Adherence').replace('supporting_facts','# Adherent Clauses').replace('_', ' ').replace('dox ','DoX ') + (f' (t={fact_score_similarity_threshold:.2f})' if score == 'fact_score' or score == 'supporting_facts' else ''))

		# Rotate x-labels for better visibility
		# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

		# Alternatively, auto-wrap text
		ax.set_xticklabels([textwrap.fill(label.get_text(), width=10) for label in ax.get_xticklabels()])


		offset_increment = 0.04    # Additional space for each new line of annotation
		
		# Annotate medians
		medians = comparison_df.median()
		for i, median in enumerate(medians):
			plt.text(i, median + offset_increment, f"{median:.3f}".rstrip('0').rstrip('.'), horizontalalignment='center', size='small', color='black', weight='semibold')

		# # is_normally_distributed = True
		# for model in model_name_dict.values():
		# 	print(f"Testing for {score}:")
		# 	# Testing normality for each group
		# 	stat, p_value = stats.shapiro(comparison_df[model])
		# 	print(f"{model} - Shapiro Test p-value: {p_value:.4f}")
		# 	# if p_value < 0.05:
		# 	# 	is_normally_distributed = False
		
		# Adjust bracket height
		annotation_y_offset = 1.02  # Starting offset just above the subplot
		for i,model in enumerate(filter(lambda x: x!=main_exp_label, model_name_dict.values())):
			x = comparison_df[main_exp_label]
			y = comparison_df[model]

			differences = np.array(x) - np.array(y)
			abs_differences = np.abs(differences)

			ranks = stats.rankdata(abs_differences)
			pos_rank_sum = np.sum(ranks[differences > 0])
			neg_rank_sum = np.sum(ranks[differences < 0])

			t = min(pos_rank_sum, neg_rank_sum)

			statistic, p_value = stats.wilcoxon(x, y, alternative='greater') #if score != 'fact_score_fuzzy' else stats.ttest_rel(x, y, alternative='greater')
			zstatistic = abs(stats.norm.ppf(p_value/2))

			effect_size = {
				'rank_biserial_correlation': rank_biserial_correlation(statistic, len(x), len(y)),
				'cliffs_delta': cliffs_delta(x, y),
				'cohens_dz': cohens_dz(zstatistic, x, y),
				'confidence_intervals': confidence_intervals(zstatistic, x, y),
			}

			print(f"{main_exp_label} > {model} comparison for {score}:")
			print(f"    Wilcoxon signed-rank test statistic (T): {statistic}")
			print(f"    Smaller Rank Sum (t): {t}")
			print(f"    P-value: {p_value}")
			print(f"    Effect size: {json.dumps(effect_size, indent=4)}")

			# Format p_value based on its magnitude
			if p_value < 0.001:
				p_text = "p<0.001"
			elif p_value > 0.999:
				p_text = "p>0.999"
			else:
				p_text = f"p={p_value:.3f}"

			# Make p_value bold if it's less than 0.05
			if p_value < 0.05:
				p_text = f"$\\bf{{{p_text}}}$"  # Use LaTeX for bold

			# Format the full annotation text
			ci_low, ci_high = effect_size['confidence_intervals']
			annotation_text = f"{main_exp_label} > {model}: T={statistic:.0f}, {p_text}, CI=[{ci_low:.2f}, {ci_high:.1f}], dz={effect_size['cohens_dz']:.2f}"
			# annotation_text = f"{main_exp_label} vs. {model}: T={statistic:.0f}, p={p_value:.3f}, Δ={effect_size['cliffs_delta']:.3f}"
			# Position text at the top of each subplot, incrementing vertically for each model
			plt.text(0.5, annotation_y_offset, annotation_text, ha='center', va='bottom', size='small', transform=ax.transAxes, bbox=dict(
				facecolor='white',   # background color
				alpha=0.75,           # transparency, 0.0 (invisible) to 1.0 (opaque)
				edgecolor='none'     # no border; set to e.g. 'black' if you want an outline
			))
			annotation_y_offset += offset_increment  # Move up for the next line

	plt.tight_layout()
	plt.savefig(plot_path)
