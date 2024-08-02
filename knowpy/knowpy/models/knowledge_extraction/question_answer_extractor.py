import sys
import json
import numpy as np
import re
from more_itertools import unique_everseen
import math
import os
from collections import defaultdict

from knowpy.models.model_manager import ModelManager
from knowpy.misc.levenshtein_lib import labels_are_contained
from knowpy.misc.adjacency_list import AdjacencyList
from knowpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from knowpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from knowpy.models.knowledge_extraction.concept_extractor import ConceptExtractor
# from knowpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences
from knowpy.misc.graph_builder import tuplefy, get_subject_set, get_object_set
# from knowpy.misc.doc_reader import clean_content
from knowpy.misc.cache_lib import load_or_create_cache
from knowpy.misc.utils import *

from knowpy.misc.jsonld_lib import *
from knowpy.models.knowledge_extraction.knowledge_graph_manager import singlefy

class QuestionAnswerExtractor(ModelManager):
	amr_interrogative_particles = [
		'what',
		'who',
		'why',
		'how',
		'how much',
		'where',
		'when',
		'who by',
		'which',
		'whose',
	]
	amr_interrogative_particles_str = ', '.join(amr_interrogative_particles)
	
	edu_interrogative_particles = [
		'despite what',
		'after what',
		'in what manner',
		'while what',
		'in what case',
		'before what',
		'since when',
		'until when',
		'instead of what',
		'except when',
		'unless what',
	]
	edu_interrogative_particles_str = ', '.join(edu_interrogative_particles)

	amr_edu_interrogative_particles_str = ', '.join([amr_interrogative_particles_str,edu_interrogative_particles_str])

	qa_extraction_prompt = """Analyse the {language} paragraph below to generate a comprehensive list of Q&As in English, capturing: {archetypal_questions}. Answers must succinctly reflect the paragraph's content without repeating the question's wording. Q&As must use precise and direct language, avoiding vague terms and generalizations, clearly specifying the context and subjects involved without assuming prior knowledge.

Example Paragraph: Alice, an experienced hiker, explores the Rocky Mountains despite rain. She packs her gear early in the morning.

Expected Output:
- Who is Alice? An experienced hiker.
- What did Alice do? Explored the Rocky Mountains.
- Despite what did Alice decide to explore the Rocky Mountains? Rain.
- What did she pack? Waterproof gear.
- When did she pack? Early in the morning.

Paragraph for Analysis:
{sentence}"""
# 	qa_extraction_prompt = """Analyse the paragraph below to generate a succinct but comprehensive list of different questions and answers in English, capturing various aspects of the content like: {archetypal_questions}. Answers must succinctly reflect the paragraph's content without repeating the question's wording. Answers and questions must use precise and direct language, avoiding vague terms and generalizations, clearly specifying the context and subjects involved without assuming prior knowledge.

# Example Paragraph: Alice, an experienced hiker, explores the Rocky Mountains despite rain. She packs her gear early in the morning.

# Expected Output:
# - Who is Alice? An experienced hiker.
# - What did Alice do? Explored the Rocky Mountains.
# - Despite what did Alice decide to explore the Rocky Mountains? Rain.
# - What did she pack? Waterproof gear.
# - When did she pack? Early in the morning.

# Paragraph for Analysis:
# {sentence}"""

	def __init__(self, model_options):
		super().__init__(model_options)
		self.disable_spacy_component = ["ner", "textcat"]
		self.generative_ai_options = model_options.get('generative_ai_options',{'model': 'mistral:instruct'})
		model_options.get('generative_ai_options',{'model': 'mistral:instruct'})
		self.min_sentence_len = model_options.get('min_sentence_len',1)
		self.max_sentence_len = model_options.get('max_sentence_len',float('inf'))
		self.min_paragraph_len = model_options.get('min_paragraph_len',1)
		self.max_paragraph_len = model_options.get('max_paragraph_len',float('inf'))
		

	def extract_aligned_graph_from_qa_dict_list(self, kg_manager, qa_dict_list, graph_builder_options, graph_extraction_options, qa_type_to_use=None, use_paragraph_text=False, **graph_cleaning_options):
		if qa_type_to_use is not None: # disco or qaamr
			qa_dict_list = list(filter(lambda x: x['type'] in qa_type_to_use, qa_dict_list))

		# Set correct paragraph_text and doc before building the EDUs graph
		span_source_id_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_SOURCE_ID_PREDICATE)
		source_uri_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_PARAGRAPH_ID_PREDICATE)
		if use_paragraph_text:
			source_to_sentence_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy)
		else:
			source_to_sentence_list = []
			for x,source_span_text_list in kg_manager.adjacency_list.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy).items():
				for source_span_text in source_span_text_list:
					# if 'The law applicable should also govern the question of the capacity to incur liability in tort/delict.' in source_span_text:
					# 	print('ooo',x, source_span_text, source_span_text=='The law applicable should also govern the question of the capacity to incur liability in tort/delict.')
					if x in span_source_id_dict:
						source_to_sentence_list += [
							(source_uri,source_span_text)
							for source_sentence_uri in span_source_id_dict[x]
							for source_uri in source_uri_dict[source_sentence_uri]
						]
					else:
						source_to_sentence_list += [
							(source_uri,source_span_text)
							for source_uri in source_uri_dict[x]
						]
			source_to_sentence_dict = defaultdict(list)
			for source_uri, source_span_text in source_to_sentence_list:
				source_to_sentence_dict[source_uri].append(source_span_text)
			del source_to_sentence_list
		### Clean memory
		del span_source_id_dict
		del source_uri_dict
		################
		
		# Build content-to-source dict
		content_to_source_uri_dict = {
			sentence: source_uri
			for source_uri,sentence_list in source_to_sentence_dict.items()
			for sentence in sentence_list
		}
		del source_to_sentence_dict

		# Build qa2sentence_dict
		qa2sentence_dict = defaultdict(list)
		for qa_dict in qa_dict_list:
			sentence = qa_dict['sentence']
			source_uri = content_to_source_uri_dict.get(sentence, None)
			# assert source_uri, f"Could not find: {sentence}"
			if not source_uri:
				self.logger.debug(f'<extract_aligned_graph_from_qa_dict_list> "{sentence}" is missing')
				continue
			question_answer = qa_dict['abstract'] # QA extractor is not normalising strings, but KnowledgeGraphExtractor will
			qa2sentence_dict[question_answer].append((sentence,source_uri))
		del content_to_source_uri_dict
		# print(json.dumps(qa2sentence_dict, indent=4))

		qa_iter = map(lambda x: x['abstract'], qa_dict_list)
		qa_iter = filter(lambda x: x in qa2sentence_dict, qa_iter)
		qa_list = list(qa_iter)
		assert qa_list, f"No valid QA found in qa_dict_list of length {len(qa_dict_list)}"

		########################
		self.logger.info(f'QuestionAnswerExtractor::extract_aligned_graph_from_qa_dict_list - Processing {len(qa_list)} QAs')
		kg_builder = KnowledgeGraphExtractor(graph_builder_options).set_content_list(qa_list, **graph_cleaning_options)
		self.logger.info(f'QuestionAnswerExtractor::extract_aligned_graph_from_qa_dict_list - Aligning triplets')
		new_triplet_list = []
		triplet_tuple_iter = iter(kg_builder.triplet_tuple)
		triplet_tuple_len = len(kg_builder.triplet_tuple)
		del kg_builder.triplet_tuple # free space
		already_processed_source_uri = set() # to reduce memory usage

		for subj_dict, predicate_dict, obj_dict in self.tqdm(triplet_tuple_iter, total=triplet_tuple_len, desc='Aligned triplets'):
			qa = predicate_dict['source']['paragraph_text']
			predicate_dict['source_text'] = qa # this will force the KnowledgeGraphExtractor to consider the EDUs/AMRs as subj/obj instead of triplets
			# assert len(qa2sentence_dict[qa]) > 0
			for i, (sentence_text, source_uri) in enumerate(qa2sentence_dict[qa]):
				annotation_dict = {
					'root': source_uri, # important to have the same source_uri of the 'source graph', without it graphs alignment might be incomplete
					'content': [],
				}
				if source_uri not in already_processed_source_uri: # get sub-graph only if not extracted yet, otherwise it's just redundant
					annotation_dict['content'] = kg_manager.get_sub_graph(source_uri) # add source_id sub-graph, i.e., the extra annotations extracted during parsing
					already_processed_source_uri.add(source_uri)
				new_predicate_dict = dict(predicate_dict) if i > 0 else predicate_dict # shallow copy only if needed
				new_predicate_dict['source'] = { # no need to add paragraph, these are already in the annotation_dict['content']
					'sentence_text': sentence_text,
					'annotation': annotation_dict,
					'doc_uri': kg_manager.doc_dict[source_uri][0],
				}
				# print(kg_manager.doc_dict[source_uri])
				new_triplet_list.append((subj_dict, new_predicate_dict, obj_dict))
		del already_processed_source_uri
		del qa2sentence_dict
		# assert new_triplet_list, 'new_triplet_list is empty'
		kg_builder.triplet_tuple = tuple(new_triplet_list)
		# Build EDUs graph
		kg_builder.logger.info('QuestionAnswerExtractor::extract_aligned_graph_from_qa_dict_list - Extracting edge_list')
		# edge_list_fn = kg_builder.parallel_get_edge_list if parallel_extraction else kg_builder.get_edge_list
		edu_graph = kg_builder.get_edge_list(kg_builder.triplet_tuple, **graph_extraction_options)
		edu_graph = tuplefy(unique_everseen(edu_graph))
		########################
		# Add useful sub-class relations
		subclass_dict = kg_manager.adjacency_list.get_predicate_dict(SUBCLASSOF_PREDICATE)
		subclass_graph = [
			(a,SUBCLASSOF_PREDICATE,b)
			for a,b_list in subclass_dict.items()
			for b in b_list
		]
		del subclass_dict
		edu_graph += subclass_graph
		edu_graph += filter(lambda x: not triplet_is_clause(x), unique_everseen(flatten(map(kg_manager.get_sub_graph, get_object_set(subclass_graph)))))
		del subclass_graph
		
		edu_graph = list(unique_everseen(edu_graph))
		return edu_graph

	def extract_qa_dict_list(self, sentence_list, cache_path=None):
		qa_dict_list = []
		sentence_list, language_list = zip(*self.detect_language_parallel(sentence_list)) # detect_language_parallel is fetching languages in parallel reshuffling in an unordered fashion, reset sentence_list accordingly
		prompt_list = [
			self.qa_extraction_prompt.format(language=language, archetypal_questions=self.amr_edu_interrogative_particles_str, sentence=sentence)
			# self.qa_extraction_prompt.format(archetypal_questions=self.amr_edu_interrogative_particles_str, sentence=sentence)
			for sentence, language in zip(sentence_list, language_list)
			if language and sentence
		]
		del language_list
		output_list = self.instruct_model(
			prompt_list, 
			**self.generative_ai_options, 
			output_to_input_proportion=2,
			non_influential_prompt_size=len(self.qa_extraction_prompt.format(language='English', archetypal_questions=self.amr_edu_interrogative_particles_str, sentence='').split(' ')),
			n=1, 
			cache_path=cache_path, 
			with_cache=cache_path is not None
		)
		self.logger.debug('Extracted questions:')
		for sentence, qa_list_str in zip(sentence_list, output_list):
			if not qa_list_str:
				self.logger.warning(f'QuestionAnswerExtractor::extract_qa_dict_list - Could not process "{sentence}"')
				continue
			if isinstance(qa_list_str,(list,tuple)):
				qa_list_str = qa_list_str[0]
			# Regular expression pattern to match and remove various types of list or enumeration markers
			pattern = r'^\s*(\d+\.|\(\w+\)|\w\)|\d+\)|\d*\-|\w\-)\s*'
			# Splitting the text by lines
			split_text = qa_list_str.split('\n')
			# Applying the regular expression to each line and filtering out empty lines
			qa_iter = (re.sub(pattern, '', item).strip() for item in split_text if item.strip())
			qa_iter = map(lambda x: x.replace('? -','?'), qa_iter)
			qa_iter = filter(lambda x: x.count('?')==1, qa_iter)
			qa_list = list(qa_iter)
			# self.logger.debug(f'"{sentence}": {json.dumps(qa_list, indent=4)}')
			qa_dict_list += (
				{
					'question': q.strip(),
					'answer': a.strip(),
					'sentence': sentence,
					'type': 'edu' if any(map(lambda x: x in q.lower(), self.edu_interrogative_particles)) else 'amr'
				}
				for q,a in map(lambda x: x.split('?'), qa_list)
			)
		return qa_dict_list

	def extract(self, graph, paraphraser_options=None, add_declarations=False, use_paragraph_text=False, cache_path=None, elements_per_chunk=10000):
		self.logger.info(f'QuestionAnswerExtractor::extract - Extracting QA dict with use_paragraph_text={use_paragraph_text}')
		# Build adjacency matrix from knowledge graph
		adjacency_list = AdjacencyList(
			graph, 
			equivalence_relation_set=set([IS_EQUIVALENT_PREDICATE]),
			is_sorted=True,
		)
		if use_paragraph_text:
			sentence_iter = flatten(adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy).values())
			sentence_iter = filter(lambda x: self.min_paragraph_len <= len(x) <= self.max_paragraph_len, sentence_iter)
		else:
			span_source_id_dict = adjacency_list.get_predicate_dict(HAS_SOURCE_ID_PREDICATE)
			content_dict = adjacency_list.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy)
			content_dict = dict(filter(lambda x: x[0] not in span_source_id_dict, content_dict.items())) # avoid clauses, consider sentences only
			del span_source_id_dict
			sentence_iter = flatten(content_dict.values())
			del content_dict
			sentence_iter = filter(lambda x: self.min_sentence_len <= len(x) <= self.max_sentence_len, sentence_iter)
		sentence_list = tuple(sentence_iter)
		### Clean memory
		del adjacency_list
		################
		# Extract QA dictionary
		qa_dict_list = self.extract_qa_dict_list(sentence_list, cache_path=cache_path)
		del sentence_list
		# print(json.dumps(question_answer_matrix, indent=4))
		# Add manually specified QAs
		known_qa_dict = defaultdict(dict)
		for qa_id, p, o in filter(lambda x: x[1] in KNOWN_QA_PREDICATES, graph):
			known_qa_dict[qa_id][p] = o
		if known_qa_dict:
			self.logger.debug('QuestionAnswerExtractor::extract - Adding manually specified QA (with tag known_qa):')
			self.logger.debug(json.dumps(known_qa_dict, indent=4))
			qa_dict_list += [
				{
					'question': qa_dict[QUESTION_TEMPLATE_PREDICATE],
					'answer': qa_dict[ANSWER_TEMPLATE_PREDICATE],
					'sentence': qa_dict.get(EXPLANATORY_TEMPLATE_PREDICATE, qa_dict[QUESTION_TEMPLATE_PREDICATE]+' '+qa_dict[ANSWER_TEMPLATE_PREDICATE]),
					'type': 'known_qa',
				}
				for qa_dict in known_qa_dict.values()
				if ANSWER_TEMPLATE_PREDICATE in qa_dict and QUESTION_TEMPLATE_PREDICATE in qa_dict
			]
			del known_qa_dict
		
		# Add missing question marks
		for qa_dict in qa_dict_list:
			qa_dict['question'] = qa_dict['question'].strip().strip('?') + '?'
		# # Remove questions with no subjects
		# self.logger.info(f'QuestionAnswerExtractor::extract - Removing questions with no subject')
		# chunks = tuple(get_chunks(qa_dict_list, elements_per_chunk=elements_per_chunk)) if elements_per_chunk else [qa_dict_list]
		# qa_dict_list = []
		# for chunk in self.tqdm(chunks):
		# 	# has_subj = lambda x: next(filter(lambda y: 'subj' in y, map(lambda x: x.dep_, self.nlp([x])[0])), None) is not None
		# 	question_list = tuple(unique_everseen(map(lambda x: x['question'], chunk)))
		# 	question_nlp_dict = dict(zip(question_list, self.nlp(question_list)))
		# 	del question_list
		# 	has_subj = lambda x: next(filter(lambda y: 'subj' in y, map(lambda x: x.dep_, question_nlp_dict[x])), None) is not None
		# 	qa_dict_list.extend(filter(lambda x: has_subj(x['question']), chunk))
		# 	del question_nlp_dict

		# Add missing abstracts
		for qa_dict in qa_dict_list:
			qa_dict['abstract'] = qa_dict['question'] + ' ' + qa_dict['answer']
		# Add declarations
		if add_declarations:
			abstract_list = list(map(lambda x: x['abstract'], qa_dict_list))
			declaration_list = self.convert_interrogative_to_declarative_sentence_list(abstract_list, paraphraser_options)
			del abstract_list
			for declaration, qa_dict in zip(declaration_list, qa_dict_list):
				qa_dict['declaration'] = declaration
			del declaration_list
		self.logger.info(f'QuestionAnswerExtractor::extract - QA dict extracted')
		return qa_dict_list
