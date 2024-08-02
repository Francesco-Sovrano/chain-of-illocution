import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, kruskal, mannwhitneyu, shapiro
import seaborn as sns
import textwrap  # Import textwrap for wrapping text
import sys

plt.rcParams.update({'font.size': 14})  # You can adjust the size as needed

min_experiment_duration = 9*60 # seconds
max_ratio_of_all_ten = 1
# min_seconds_per_question = 0

topic_list = [
    'java',
    'python',
    'pharo',
    'design_patterns',
]

strategy_list = [
    'GenAI',
    'RAG', 
    'RAG+CoI', 
]

model_list = [
    'gpt-3.5-turbo',
    'gpt-4o',
]

metrics = ['Net Promoter Score', 'Relevance', 'Correctness']

# Load JSON files from the directory
def load_data(directory):
    data = []
    for filename in os.listdir(directory):
        if not filename.endswith('.json'): #and '_' not in filename:
            continue
        
        with open(os.path.join(directory, filename), 'r') as file:
            json_data = json.load(file)
        if "qualitative_feedback" not in json_data:
            continue
        
        evaluation_list = json_data['evaluation_list']
        evaluation_list = [e for e in evaluation_list if e['topic'] in topic_list]
        if not np.sum([d['elapsed_seconds'] for d in evaluation_list]) >= min_experiment_duration:
            continue 
        # if not all(d['elapsed_seconds'] >= min_seconds_per_question for d in evaluation_list):
        #     continue
            
        # Check how many ratings are equal to 10
        all_ten = np.sum([
            all(v==10 for v in sub_d.values())
            for d in evaluation_list
            for sub_d in d['evaluation_dict'].values()
        ]) >= len(evaluation_list)*max_ratio_of_all_ten
        # all_ten = np.mean([
        #     v
        #     for d in evaluation_list
        #     for sub_d in d['evaluation_dict'].values()
        #     for v in sub_d.values()
        # ]) >= 9
        if all_ten:
            continue
            
        data += evaluation_list
    return data

# def map_likert(value):
#     if value >= 8:
#         return 1
#     elif value >= 6:
#         return 0.5
#     else:
#         return 0

# Filter and process the data
def process_data(data):
    # data = [entry for entry in data if entry['elapsed_seconds'] >= 180]
    # data = [entry for entry in data if entry['java_knowledge'] >= 4 or entry['topic']!='java']
    # data = [entry for entry in data if entry['python_knowledge'] >= 4 or entry['topic']!='python']
    results = {metric: [] for metric in metrics}
    
    for entry in data:
        for metric in metrics:
            result_dict = {
                m: entry['evaluation_dict'][metric][m] 
                for m in strategy_list
            }
            result_dict['question'] = entry['question']
            results[metric].append(result_dict)

    df_results = {}
    for metric, entries in results.items():
        df = pd.DataFrame(entries)
        df = df.groupby('question').mean()
        df = df.reset_index() # Reset the index to make 'question' a column
        df = df.sort_values('question') # Sort the DataFrame by the 'question' column
        # df['GenAI'] = df['GenAI'].apply(map_likert)
        # df['RAG'] = df['RAG'].apply(map_likert)
        # df['RAG+CoI'] = df['RAG+CoI'].apply(map_likert)
        df_results[metric] = df
            
    return df_results

# Perform statistical tests
def statistical_tests(data):
    test_results = {metric: {} for metric in metrics}

    for metric, df in data.items():
        print(f"Testing for {metric}:")
        # Testing normality for each group
        for column in strategy_list:
            stat, p_value = shapiro(df[column])
            print(f"{column} - Shapiro Test p-value: {p_value:.4f}")

    for metric, df in data.items():
        for i,col1 in enumerate(strategy_list):
            for col2 in strategy_list[i+1:]:
                if col1 == col2:
                    continue
                if df[col1].median() > df[col2].median():
                    alternative = 'greater'
                    symbol = '>'
                elif df[col1].median() < df[col2].median():
                    alternative = 'less'
                    symbol = '<'
                else:
                    alternative = 'two-sided'
                    symbol = '!='
                stat, p = wilcoxon(df[col1], df[col2], alternative=alternative)
                test_results[metric][f'{col1} {symbol} {col2}'] = (p, stat, len(df[col1]), len(df[col2]))
                
    return test_results

# Plotting the results with median annotations and improved title positioning
def plot_results(data, test_results, figure_path):
    fig, axes = plt.subplots(1, 3, figsize=(4 * len(metrics), 4))

    # Load the default color palette
    palette = sns.color_palette()
    if len(strategy_list) == 2:
        palette = palette[1:]
    
    for i, (metric, df) in enumerate(data.items()):
        df_melted = df.melt(id_vars=['question'], var_name='Strategy', value_name=metric)
        ax = sns.boxplot(x='Strategy', y=metric, hue='Strategy', data=df_melted, ax=axes[i], palette=palette)
        # ax.set_title(metric)  # Adjust title padding
        ax.set_xlabel('')
        # ax.set_ylabel(metric.replace('Net Promoter Score','Satisfaction').strip(), labelpad=0)
        ax.set_title(metric.replace('Net Promoter Score','Satisfaction').strip()) #, y=1.06)  # Adjust the y value to move the title
        ax.set_ylabel('')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Show median values on the plot
        median_values = df_melted.groupby('Strategy')[metric].median()
        medians_formatted = [f'{median:.2f}' for median in median_values]
        positions = range(len(median_values))

        offset_increment = 0.04    # Additional space for each new line of annotation

        # Add text annotations for median values
        for tick, label in zip(positions, axes[i].get_xticklabels()):
            median = median_values[label.get_text()]
            ax.text(tick, median+offset_increment, f"{median:.3f}".rstrip('0').rstrip('.'), horizontalalignment='center', size='small', color='black', weight='semibold')

        ax.set_xticklabels([textwrap.fill(label.get_text(), width=10) for label in ax.get_xticklabels()])

        # # Annotating p-values
        # annotation_y_offset = 1.02  # Starting offset just above the subplot
        # for key, value in test_results[metric].items():
        #     p_value, statistic, _, _ = value
        #     # Format p-value
        #     if p_value < 0.001:
        #         p_text = "p<0.001"
        #     elif p_value > 0.999:
        #         p_text = "p>0.999"
        #     else:
        #         p_text = f"p={p_value:.3f}"

        #     # Make p_value bold if it's less than 0.05
        #     if p_value < 0.05:
        #         if f"{p_value:.3f}" == '0.05':
        #             p_text = "p=0.049"
        #         p_text = f"$\\bf{{{p_text}}}$"  # Use LaTeX for bold

        #     # Format the full annotation text
        #     # annotation_text = f"{key}: T={statistic:.0f}, {p_text}"
        #     annotation_text = f"{key}: {p_text}"
        #     plt.text(0.5, annotation_y_offset, annotation_text, ha='center', va='bottom', size='small', transform=ax.transAxes)
        #     annotation_y_offset += offset_increment  # Move up for the next line
    
    plt.subplots_adjust()  # Increase the width space between subplots
    plt.tight_layout()  # Adjust the layout to fit better
    plt.savefig(figure_path)
    plt.show()

# Additional function to perform the Wilcoxon test
def compare_models(data1, data2):
    comparison_results = {}
    for metric in metrics:
        comparison_results[metric] = {}
        for column in strategy_list:
            model1_data = data1[metric][column]
            model2_data = data2[metric][column]
            # print(len(model1_data))
            # print(len(model2_data))
        
            # Perform Wilcoxon signed-rank test
            try:
                stat, p_value = wilcoxon(model1_data, model2_data, alternative='greater')
                comparison_results[metric][column] = (stat, p_value)
            except:
                pass
    
    return comparison_results


# Main function to run the program
def main():
    model_data = {}
    for model in model_list:
        raw_data = load_data(f'./results/{model}')
        processed_data = process_data(raw_data)
        model_data[model] = processed_data
        test_results = statistical_tests(processed_data)
        print("Statistical Test Results:")
        print(test_results)
        plot_results(processed_data, test_results, f'./user_study_results_{model}.pdf')

    # Compare GPT-4o with GPT-3.5-turbo
    comparison_results = compare_models(model_data['gpt-4o'], model_data['gpt-3.5-turbo'])
    print("\nComparison Results between GPT-4o and GPT-3.5-turbo:")
    print(json.dumps(comparison_results, indent=4))


if __name__ == "__main__":
    main()
