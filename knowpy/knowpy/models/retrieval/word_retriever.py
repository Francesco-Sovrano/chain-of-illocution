from knowpy.misc.utils import *
from knowpy.misc.doc_reader import DocParser
from knowpy.models.retrieval.sentence_retriever import SentenceRetriever
from knowpy.models.knowledge_extraction.concept_extractor import ConceptExtractor
from more_itertools import unique_everseen
from collections import Counter
try:
	from nltk.corpus import stopwords
except Exception as e:
	import nltk
	print('Downloading nltk::stopwords\n'
		"(don't worry, this will only happen once)")
	nltk.download('stopwords')
	from nltk.corpus import stopwords
import re
# import json
import itertools

class WordRetriever(SentenceRetriever):
	def __init__(self, model_options, **argv):
		# self.logger.info('Initialising WordRetriever')
		super().__init__(model_options, **argv)
		self.concept_extractor = ConceptExtractor(model_options)

	def set_concept_description_dict(self, concept_description_dict, **args):
		id_list, doc_list = zip(*(
			(key,description) 
			for key, value in concept_description_dict.items() 
			for description in value
			if description
		))
		self.set_documents(id_list, doc_list, **args)
		return self
	
	def get_word_dict(self, doc_parser: DocParser, concept_counter_dict=None, similarity_threshold=None, remove_numbers=True, remove_stopwords=True, lemmatized=False, concept_label_filter=None, concept_id_filter=None, parallel_extraction=False, top_k=None, remove_pronouns=True, **args):
		if concept_counter_dict is None:
			concept_counter_dict = {}
		# Extract concept dict list
		concept_dict_iter = self.concept_extractor.get_concept_list(doc_parser, 
			remove_source_paragraph=False, 
			remove_idx=True, 
			remove_span=True, 
			remove_pronouns=remove_pronouns,
			parallel_extraction=parallel_extraction, 
		)
		get_concept_label = lambda x: x['concept']['lemma' if lemmatized else 'text']
		# Remove unwanted concepts
		if concept_label_filter:
			concept_dict_iter = filter(
				lambda x: concept_label_filter(get_concept_label(x)), 
				concept_dict_iter
			)
		if remove_numbers:
			concept_dict_iter = filter(
				# lambda x: re.search(r'\d', get_concept_label(x)) is None, 
				lambda x: not is_number(get_concept_label(x)),
				concept_dict_iter
			)
		if remove_stopwords:
			try:
				get_stopwords = lambda x: stopwords.words(language_map.get(x['source']['language_id'], self.model_options.get('language','English')).lower())
				concept_dict_iter = filter(
					lambda x: get_concept_label(x).lower() not in get_stopwords(x), 
					concept_dict_iter
				)
			except Exception as e:
				self.logger.error('WordRetriever::get_word_dict:', e)
		if not isinstance(concept_dict_iter, (list,tuple)):
			concept_dict_iter = list(concept_dict_iter)
		# Extract concept_counter
		concept_iter = map(get_concept_label, concept_dict_iter)
		concept_list = tuple(concept_iter)
		concept_counter = Counter(concept_list)
		# Merge input concept_counter_dict with concept_counter
		for concept,count in concept_counter.items():
			if concept not in concept_counter_dict:
				concept_counter_dict[concept] = {
					'count': count, 
					'source_list': [],
					'similar_to': []
				}
			else:
				concept_counter_dict[concept]['count'] += count
		# Add sources
		for concept, cdict in zip(concept_list, concept_dict_iter):
			# concept_counter_dict[concept]['span'] = cdict['concept']['span']
			concept_counter_dict[concept]['source_list'].append(cdict['source'])
		# Add similarities
		if not concept_counter_dict:
			return {}
		text_list, cdict_list = zip(*concept_counter_dict.items())
		# formatted_text_list = tuple(map(lambda x: x['span'], cdict_list))
		index_of_most_similar_documents_list = self.retrieve(
			text_list, 
			similarity_threshold=similarity_threshold, 
			top_k=top_k, 
			**args
		)
		for concept, index_of_most_similar_documents in zip(text_list, index_of_most_similar_documents_list):
			if concept_id_filter is not None:
				index_of_most_similar_documents = filter(lambda x: concept_id_filter(x['id']), index_of_most_similar_documents)
			if concept_label_filter is not None:
				index_of_most_similar_documents = filter(lambda x: concept_label_filter(x['doc']), index_of_most_similar_documents)
			concept_counter_dict[concept]['similar_to'] = tuple(index_of_most_similar_documents)
			concept_counter_dict[concept]['source_list'] = tuple(unique_everseen(concept_counter_dict[concept]['source_list'], key=lambda x:x['sentence_text']))
		return concept_counter_dict

	@staticmethod
	def get_missing_concepts_counter(concept_dict):
		return {
			concept: value['count']
			for concept, value in concept_dict.items()
			if len(value['similar_to'])==0
		}

	def annotate(self, doc_parser: DocParser, similarity_threshold=None, max_concepts_per_alignment=1, concept_id_filter=None, is_preprocessed_content=False, top_k=None, parallel_extraction=True, remove_pronouns=True, **args):
		self.logger.info(f'Annotating {"preprocessed" if is_preprocessed_content else "new"} documents..')
		if not is_preprocessed_content:
			concept_dict = self.get_word_dict(
				doc_parser, 
				similarity_threshold= similarity_threshold, 
				remove_numbers= True, 
				lemmatized= False,
				remove_stopwords= True,
				concept_id_filter=concept_id_filter,
				parallel_extraction=parallel_extraction,
				top_k=max_concepts_per_alignment,
				remove_pronouns=remove_pronouns,
				**args
			)
			annotation_iter = (
				{
					'text': concept_label,
					'annotation': concept_uri_dict['id'],
					'similarity': concept_uri_dict['similarity'],
					'syntactic_similarity': concept_uri_dict['syntactic_similarity'],
					'semantic_similarity': concept_uri_dict['semantic_similarity'],
				}
				for concept_label,similarity_dict in concept_dict.items()
				for concept_uri_dict in similarity_dict['similar_to']
			)
			annotation_iter = unique_everseen(annotation_iter, key=lambda x: x['text'])
		else:
			content_iter = doc_parser.get_content_iter()
			content_txt = ' '.join(unique_everseen(content_iter)).casefold()
			annotation_iter = (
				{
					'text': concept_label,
					'annotation': concept_id,
					# 'similarity': concept_uri_dict['similarity'],
					'similarity': 1,
					# 'syntactic_similarity': concept_uri_dict['syntactic_similarity'],
					'syntactic_similarity': 1,
					# 'semantic_similarity': concept_uri_dict['semantic_similarity'],
					'semantic_similarity': 0,
				}
				for concept_id, concept_label in zip(self.ids, self.documents)
				if concept_id_filter(concept_id) 
				and len(concept_label) > 1
				and concept_label.casefold() in content_txt
			)

		annotation_list = list(annotation_iter)
		# print(json.dumps(annotation_list, indent=4))
		return annotation_list
