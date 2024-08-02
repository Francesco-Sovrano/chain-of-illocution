from knowpy.models.retrieval.word_retriever import WordRetriever
from knowpy.models.retrieval import is_not_wh_word
from knowpy.misc.doc_reader import DocParser
from knowpy.misc.jsonld_lib import *

from itertools import islice
from collections import defaultdict, namedtuple
from more_itertools import unique_everseen

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

ArchetypePertinence = namedtuple('ArchetypePertinence',['archetype','pertinence'])

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def init_word_retriever(word_retriever, document_path):
	concept_dict_iter = word_retriever.concept_extractor.get_concept_list(
		DocParser().set_documents_path(document_path), 
		remove_source_paragraph=False, 
		remove_idx=True, 
		remove_span=True, 
		remove_pronouns=False,
		parallel_extraction=False, 
	)
	concept_description_dict = defaultdict(list)
	for c in concept_dict_iter:
		concept_description_dict[c['concept']['lemma']].append(c['concept']['text'].strip())
	concept_description_dict = {
		uri: list(unique_everseen(
			filter(
				lambda x: len(x) > 2 and not is_number(x),
				label_list
			), 
			key=lambda x: x.casefold()
		))
		for uri, label_list in concept_description_dict.items()
	}
	concept_description_dict = {
		uri: label_list
		for uri, label_list in concept_description_dict.items()
		if label_list
	}
	word_retriever.set_concept_description_dict(concept_description_dict)

def get_related_concepts(word_retriever, question_list, query_concept_similarity_threshold=0.55, ignore_numbers=True, ignore_stopwords=False, lemmatized=False, keep_the_n_most_similar_concepts=None, concept_label_filter=is_not_wh_word, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, filter_fn=None, answer_horizon=None, minimise=False, ignore_pronouns=True, **args):
	concepts_dict = word_retriever.get_word_dict(
		doc_parser=DocParser().set_content_list(question_list),
		similarity_threshold=query_concept_similarity_threshold, 
		remove_numbers=ignore_numbers, 
		remove_stopwords=ignore_stopwords, 
		lemmatized=lemmatized,
		concept_label_filter=concept_label_filter,
		top_k=keep_the_n_most_similar_concepts,
		remove_pronouns=ignore_pronouns,
		with_cache=False, # caching arbitrary inputs from users might cause a quick memory exhaustion
	)
	# Group queries by concept_uri
	concept_query_uri_dict = defaultdict(set)
	# Determine the maximum number of similar concepts to keep, defaulting to None if not specified.
	max_similar = max(1, keep_the_n_most_similar_concepts) if keep_the_n_most_similar_concepts else None
	# Iterate over each concept and its associated counts in the concepts dictionary.
	for concept_label, concept_count_dict in concepts_dict.items():
		# Create an iterator over unique 'similar_to' entries, ensuring each is seen only once based on its 'id'.
		concept_similarity_dict_iter = unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"])
		# If a limit on the number of similar concepts to keep is specified, apply it to the iterator.
		if keep_the_n_most_similar_concepts:
			concept_similarity_dict_iter = islice(concept_similarity_dict_iter, max_similar)
		# Iterate over each similar concept.
		for concept_similarity_dict in concept_similarity_dict_iter:
			concept_uri_label = (concept_similarity_dict["id"], concept_similarity_dict["doc"])
			# Update the query dictionary with paragraph texts from the source list of the current concept.
			for sent_dict in concept_count_dict["source_list"]:
				concept_query_uri_dict[sent_dict["paragraph_text"]].add(concept_uri_label)
		# Construct the URI for the current concept label.
		concept_uri = CONCEPT_PREFIX+get_uri_from_txt(urify(concept_label))
		# Update the URI with paragraph texts.
		for sent_dict in concept_count_dict["source_list"]:
			concept_query_uri_dict[sent_dict["paragraph_text"]].add((concept_uri, concept_label))
	
	total_concept_uri_set = []
	for concept_query, concept_uri_set in concept_query_uri_dict.items():
		total_concept_uri_set += concept_uri_set
	return list(unique_everseen(total_concept_uri_set, key=lambda x: x[0]))

def contains_verb(sentence):
	# Tokenize the sentence into words
	words = word_tokenize(sentence)
	# Tag each word with part of speech
	tagged_words = pos_tag(words)
	
	# Check if any word is a verb
	for word, tag in tagged_words:
		if tag.startswith('VB'):  # VB, VBD, VBG, VBN, VBP, VBZ are tags for verbs
			return True
	return False

def remove_redundant_facts_from_fact_dict(data_dict):
	# Create a sorted list of keys based on the length of the keys
	sorted_keys = sorted(data_dict.keys(), key=len)
	
	# Create a copy of the dictionary
	updated_dict = dict(data_dict)
	
	# Iterate over the sorted keys list
	for i, key1 in enumerate(sorted_keys[:-1]):
		# Find a key that includes key1 as a substring
		super_key = next((key2 for key2 in sorted_keys[i+1:] if key1 in key2), None)
		
		# If such a key exists, compare and possibly update the value in the dictionary
		if super_key:
			# updated_dict[super_key] = max(updated_dict[key1], updated_dict[super_key])
			del updated_dict[key1]  # Remove the smaller key
	
	# Return the updated dictionary
	return updated_dict

def get_most_similar_questions(sentence_retriever, qa_dict_iter, input_question, top_k=10, max_question_length=None, valid_question_type_set=None, question_to_input_min_similarity_threshold=None, question_to_question_max_similarity_threshold=None, with_cache=False, **args):
	# Filtering questions by their type if a set of valid types is provided.
	if valid_question_type_set:
		qa_dict_iter = filter(lambda x: x['type'][0] in valid_question_type_set, qa_dict_iter)
	# Adjusting the maximum question length based on the length of the concept's label.
	if max_question_length:
		qa_dict_iter = filter(lambda x: len(x['question']) <= max_question_length, qa_dict_iter)
	# Setting up the sentence_retriever
	question_list = list(unique_everseen(map(lambda x: x['question'], qa_dict_iter)))
	if not question_list:
		return []
	# Further processing to eliminate similar questions.
	if question_to_question_max_similarity_threshold:
		question_list = sorted(
			question_list, 
			key=len, 
			# reverse=True,
		)
		question_list = sentence_retriever.remove_similar_labels(
			question_list,
			threshold=question_to_question_max_similarity_threshold, 
			# key=lambda x: x,
			without_context=True,
			with_cache=with_cache, # caching arbitrary inputs from users might cause a quick memory exhaustion
		)
	# Retrieve top-k questions
	sentence_retriever.set_documents(question_list, question_list)
	retrieved_question_dict_gen = tuple(sentence_retriever.retrieve(
		query_list=[input_question], 
		similarity_threshold=question_to_input_min_similarity_threshold, 
		without_context=True, 
		top_k=top_k,
		with_cache=with_cache, # caching arbitrary inputs from users might cause a quick memory exhaustion
	))[0]
	return list(unique_everseen(map(lambda x: x['doc'], retrieved_question_dict_gen)))

def get_answer_question_pertinence_dict(question_answer_dict, update_answers=False):
	answer_question_pertinence_dict = defaultdict(list)
	for question,answers in question_answer_dict.items():
		for a in answers:
			answer_question_pertinence_dict[a['sentence']].append(ArchetypePertinence(question, a['confidence']))
	if update_answers:
		for question,answers in question_answer_dict.items():
			for a in answers:
				a['question_pertinence_set'] = answer_question_pertinence_dict[a['sentence']]
	return answer_question_pertinence_dict

def minimise_question_answer_dict(question_answer_dict):
	# remove duplicated answers
	answer_question_dict = get_answer_question_pertinence_dict(question_answer_dict)
	get_best_answer_archetype = lambda a: max(answer_question_dict[a['sentence']], key=lambda y: y.pertinence).archetype
	return {
		question: list(filter(lambda x: get_best_answer_archetype(x)==question, answers))
		for question,answers in question_answer_dict.items()
	}

def merge_duplicated_answers(question_answer_dict):
	"""
	For each question, remove duplicate answers, favoring the longest answer.
	This is done by first sorting answers by their length and then ensuring
	only unique, longest answers are kept for each question.
	
	Args:
	question_answer_dict (dict): A dictionary where keys are questions and values are lists of answers. Each answer is a dictionary with a 'sentence' key.
	
	Returns:
	dict: The updated dictionary with duplicates removed based on the 'sentence' key.
	"""
	# Initialize a cache to track processed sentences
	sentence_cache = {}
	# Sort answers by the length of the sentence, in descending order
	sorted_answers = sorted(
		unique_everseen(map(lambda answer: answer['sentence'], flatten(question_answer_dict.values()))),
		key=len,
		reverse=True
	)
	# Process each question and its answers
	for question, answers in question_answer_dict.items():
		# Update sentences in answers to keep only the longest, unique ones
		for answer in answers:
			sentence = answer['sentence']
			cached_sentence = sentence_cache.get(sentence,None)
			if cached_sentence is not None:
				# If already processed, reuse the longest sentence, this will save some time
				answer['sentence'] = cached_sentence
			else:
				# Find the longest sentence that contains the current one and update
				longest_sentence = next(filter(lambda s: sentence in s, sorted_answers))
				sentence_cache[sentence] = answer['sentence'] = longest_sentence
	# Remove duplicate answers based on updated sentences
	for question in question_answer_dict:
		question_answer_dict[question] = list(unique_everseen(question_answer_dict[question], key=lambda answer: answer['sentence']))
	return question_answer_dict