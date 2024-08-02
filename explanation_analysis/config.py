import sched, time
import json
import math
from os import mkdir, path as os_path
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


from knowpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from knowpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from knowpy.models.retrieval.sentence_retriever import SentenceRetriever
from knowpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from knowpy.misc.doc_reader import load_or_create_cache, DocParser, get_document_list
from knowpy.misc.graph_builder import save_graphml
from knowpy.misc.utils import *
from more_itertools import unique_everseen

import sys
import logging
logger = logging.getLogger('knowpy')
logger.setLevel(logging.INFO)
# logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler(sys.stdout))

EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES = True # set it to True to reduce the number of considered Q/A and memory footprint
AVOID_JUMPS = True

KG_MANAGER_OPTIONS = {
	'spacy_model': 'en_core_web_md',
	# 'n_threads': 1,
	'use_gpu': False,
	'with_cache': False,
	'with_tqdm': False,

	# 'min_triplet_len': 0,
	# 'max_triplet_len': float('inf'),
	'min_sentence_len': 100,
	# 'max_sentence_len': 2000,
	# 'min_paragraph_len': 100,
	'max_paragraph_len': 2000,
}

OQA_OPTIONS = {
	'answer_horizon': 10,
	######################
	## AnswerRetriever stuff
	'answer_pertinence_threshold': 0.55, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.95,
	'top_k': 100,
	# 'filter_fn': lambda a: not ('....' in a['sentence'] or a['sentence'].startswith('*') or a['sentence'].casefold().startswith('figure')),
	'keep_the_n_most_similar_concepts': 15, 
	'query_concept_similarity_threshold': 0.75, 
	'add_external_definitions': False, 
	'include_super_concepts_graph': True, 
	'include_sub_concepts_graph': True, 
	'consider_incoming_relations': True,
	'depth': None,
	'minimise': True, 
	###########################
	'ignore_stopwords': True,
	'ignore_pronouns': False,
	'ignore_numbers': True,
}

QA_EXTRACTOR_OPTIONS = {
	# 'with_cache': False,
	'with_tqdm': True,
	'use_gpu': True,
	'n_threads': 4,
	'generative_ai_options': {
		'model': 'mistral:instruct',
		"options": { # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
			"seed": 42, # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
			"num_predict": -2, # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
			"top_k": 40, # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
			"top_p": 0.95, # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
			"temperature": 0.7, # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
			"repeat_penalty": 1., # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
			"tfs_z": 1, # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
			"num_ctx": 2**13,  # Sets the size of the context window used to generate the next token. (Default: 2048)
			"repeat_last_n": 64, # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
			# "num_gpu": 0, # The number of layers to send to the GPU(s). Set to 0 to disable.
		},
		'empty_is_missing': True,
	},
	'min_sentence_len': KG_MANAGER_OPTIONS.get('min_paragraph_len',1),
	'max_sentence_len': KG_MANAGER_OPTIONS.get('max_sentence_len',float('inf')),
	'min_paragraph_len': KG_MANAGER_OPTIONS.get('min_paragraph_len',1),
	'max_paragraph_len': KG_MANAGER_OPTIONS.get('max_paragraph_len',float('inf')),
}

GRAPH_EXTRACTION_OPTIONS = {
	'add_verbs': False, 
	'add_predicates_label': False, 
	'add_subclasses': True, 
	'use_wordnet': False,
	'lemmatize_label': False,
	'max_syntagma_length': None,
}

GRAPH_CLEANING_OPTIONS = {
	'avoid_jumps': AVOID_JUMPS,
	'parallel_extraction': False,
	'remove_language_undetectable_sentences': True,
}

GRAPH_BUILDER_OPTIONS = {
	'spacy_model': KG_MANAGER_OPTIONS['spacy_model'],
	'with_cache': True,
	'with_tqdm': True,
}

WORD_RETRIEVER_OPTIONS = {
	# 'n_threads': 1,
	# 'default_batch_size': 20,
	'with_tqdm':True,
	'fast_knn_search_options': {
		'ef_search': 50,
		# 'knn_activation_threshold': 1,
		'max_number_of_connections': 64,
		'ef_construction': 400,
		'knn_activation_threshold': float('inf'),
	},
	'sbert_model': {
		'url': 'sentence-transformers/all-mpnet-base-v2', # other models: https://huggingface.co/models?pipeline_tag=sentence-similarity&library=sentence-transformers&language=en,de,it,fr&sort=trending
		'use_gpu': True,
		'with_cache': True,
	},
	'default_similarity_threshold': OQA_OPTIONS['query_concept_similarity_threshold'],
}

SENTENCE_RETRIEVER_OPTIONS = {
	'n_threads': 25,
	# 'default_batch_size': 100,
	'with_stemmed_tfidf': True,
	'with_tqdm':True,
	'with_cache': True,
	'fast_knn_search_options': {
		'ef_search': int(OQA_OPTIONS['top_k']*1.25),
		'knn_activation_threshold': float('inf'),
	},
	'generative_ai_options': {
		'api_key': os.environ.get('OPENAI_API_KEY'),
		'model': 'gpt-3.5-turbo',
		# 'model': 'gpt-4o',
		'empty_is_missing': True,
		'temperature': 1, # default values
		'top_p': 1, # default values
	},
	'openai_embedding_model': {
		'api_key': os.environ.get('OPENAI_API_KEY'),
		'model': 'text-embedding-3-large',
	}
}

def get_qa_dict_list(cache_path, document_path):
	qa_extractor_cache = os_path.join(cache_path,'qa_extractor.pkl')
	graph_cache = os_path.join(cache_path,f"graph_clauses_lemma-{GRAPH_EXTRACTION_OPTIONS['lemmatize_label']}_avoidjumps-{AVOID_JUMPS}.pkl")
	QA_EXTRACTOR_OPTIONS['default_cache_path'] = qa_extractor_cache
	GRAPH_BUILDER_OPTIONS['default_cache_path'] = qa_extractor_cache+'.kg_builder.pkl'

	########################################################################
	print('Extracting Knowledge Graph')
	graph = load_or_create_cache(graph_cache, lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_document_list(get_document_list(document_path), **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS))
	# save_graphml(graph, 'knowledge_graph')
	
	########################################################################
	def _extract_qa_dict_list():
		qa_extractor = QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS)
		qa_extractor.load_cache(qa_extractor_cache)
		return qa_extractor.extract(graph, cache_path=qa_extractor_cache, use_paragraph_text=EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES)
	return load_or_create_cache(
		os_path.join(cache_path, f"qa_dict_list_paragraphs-{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}.pkl"),
		lambda: _extract_qa_dict_list()
	)

################ Initialise data structures ################
def init(cache_path, document_path):
	information_units = 'clause'
	graph_cache = os_path.join(cache_path,f"graph_clauses_lemma-{GRAPH_EXTRACTION_OPTIONS['lemmatize_label']}_avoidjumps-{AVOID_JUMPS}.pkl")
	qa_cache = os_path.join(cache_path,f'adaptive_qa_embedder-{information_units}.pkl')
	SENTENCE_RETRIEVER_OPTIONS['default_cache_path'] = qa_cache+'.sentence_retriever.pkl'
	WORD_RETRIEVER_OPTIONS['default_cache_path'] = qa_cache+'.word_retriever.pkl'
	GRAPH_BUILDER_OPTIONS['default_cache_path'] = qa_cache+'.kg_builder.pkl'

	########################################################################
	print('Extracting Knowledge Graph')
	graph = load_or_create_cache(graph_cache, lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_document_list(get_document_list(document_path), **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS))
	# save_graphml(graph, 'knowledge_graph')
	
	########################################################################
	kg = list(unique_everseen(graph))
	del graph
	# save_graphml(kg, 'knowledge_graph')
	print('Graph size:', len(kg))
	kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, kg)
	print('Grammatical Clauses:', kg_manager.graph_clauses_count)
	del kg

	sentence_retriever = SentenceRetriever(SENTENCE_RETRIEVER_OPTIONS)
	sentence_retriever.load_cache()
	# document_list, context_list, _, _ = zip(*kg_manager.get_sourced_graph())
	# docs_to_embed = zip(document_list,context_list)
	# sentence_retriever.sentence_embedding_fn(docs_to_embed, without_context=False, with_cache=True, remove_other_values=False)
	print('Done')

	return kg_manager, sentence_retriever
