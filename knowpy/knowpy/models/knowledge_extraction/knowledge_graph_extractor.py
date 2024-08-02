from knowpy.misc.doc_reader import DocParser
from knowpy.models.knowledge_extraction.couple_extractor import CoupleExtractor
from knowpy.models.retrieval.word_retriever import WordRetriever
from knowpy.misc.graph_builder import build_edge_dict, get_biggest_connected_graph, get_subject_set, get_concept_description_dict
from knowpy.misc.jsonld_lib import *
from knowpy.misc.utils import *
from knowpy.misc.cache_lib import create_cache, load_cache

import logging
import re
from more_itertools import unique_everseen
from collections import Counter
import json

class KnowledgeGraphExtractor(CoupleExtractor):
	
	def set_documents_path(self, doc_path, **args):
		doc_parser = DocParser(with_tqdm=self.with_tqdm, add_language_tag=True).set_documents_path(doc_path)
		self.build_triplet_list(doc_parser, **args)
		return self

	def set_document_list(self, doc_list, **args):
		doc_parser = DocParser(with_tqdm=self.with_tqdm, add_language_tag=True).set_document_list(doc_list)
		self.build_triplet_list(doc_parser, **args)
		return self

	def set_content_list(self, content_list, **args):
		doc_parser = DocParser(with_tqdm=self.with_tqdm, add_language_tag=True).set_content_list(content_list)
		self.build_triplet_list(doc_parser, **args)
		return self

	def build_triplet_list(self, doc_parser: DocParser, remove_language_undetectable_sentences=True, avoid_jumps=True, parallel_extraction=False):
		self.logger.info('KnowledgeGraphExtractor::build_triplet_list - Building triplets')
		if remove_language_undetectable_sentences: # Ignore sentences for which we don't know the language
			doc_parser.remove_language_undetectable_sentences()
		triplet_iter = self.get_triplet_list(
			doc_parser, 
			avoid_jumps=avoid_jumps,
			parallel_extraction=parallel_extraction,
			# remove_predicate_cores=True, 
			remove_source_paragraph=False, 
			# remove_idx=True, 
			# remove_span=True,
		)
		self.triplet_tuple = tuple(triplet_iter)
		self.graph_tuple = tuple(doc_parser.get_graph_iter())

	def store_cache(self, cache_name, cache_models=True, remove_predicate_cores=False, remove_idx=False, remove_span=False):
		if cache_models:
			super().store_cache(cache_name)
		cache_dict = {
			'triplet_tuple': tuple(self.clean_couples_from_tokens(
				self.triplet_tuple, 
				remove_predicate_cores=remove_predicate_cores, # minimise memory usage
				remove_source_paragraph=False, # minimise memory usage
				remove_idx=remove_idx, # minimise memory usage
				remove_span=remove_span, # minimise memory usage
			)),
			'graph_tuple': self.graph_tuple,
		}
		create_cache(cache_name+'.kg.pkl', lambda: cache_dict)

	def load_cache(self, cache_name):
		super().load_cache(cache_name)
		loaded_cache = load_cache(cache_name+'.kg.pkl')
		if not loaded_cache:
			return False

		triplet_tuple = loaded_cache.get('triplet_tuple',None)
		if triplet_tuple is not None:
			self.triplet_tuple = triplet_tuple

		graph_tuple = loaded_cache.get('graph_tuple',None)
		if graph_tuple is not None:
			self.graph_tuple = graph_tuple

		return True

	@staticmethod
	def is_valid_syntagm(syntagm, max_syntagma_length):
		if not max_syntagma_length:
			return True
		return syntagm.count(' ') < max_syntagma_length

	@staticmethod
	def get_family_concept_set(graph, concept_set, max_depth=None, current_depth=0):
		if len(concept_set) == 0:
			return set()
		sub_concept_set = get_subject_set(filter(lambda x: x[1]==SUBCLASSOF_PREDICATE and x[0] not in concept_set and x[-1] in concept_set, graph))
		current_depth +=1
		if len(sub_concept_set) == 0 or current_depth==max_depth:
			return set(concept_set)
		return KnowledgeGraphExtractor.get_family_concept_set(graph, concept_set.union(sub_concept_set), max_depth, current_depth)

	def get_edge_list(self, triplet_iter, add_subclasses=True, use_framenet_fe=False, use_wordnet=False, lemmatize_label=False, add_verbs=False, add_predicates_label=False, **args):
		# Format triples
		self.logger.info('KnowledgeGraphExtractor::get_edge_list - Building formatted_edge_list')
		get_concept_label = (lambda c: c['lemma']) if lemmatize_label else (lambda c: c['text'])
		get_concept_id = lambda s: CONCEPT_PREFIX+get_uri_from_txt(s['lemma'])
		# get_concept_doc_idx = lambda c: c['idx']
		get_concept_source_text = lambda c: c['source_text']
		paragraph_dict = {}
		formatted_edge_list = []
		if (use_wordnet or use_framenet_fe or add_subclasses) and not isinstance(triplet_iter, (list,tuple)):
			triplet_iter = tuple(triplet_iter)
		for ii,(s,p,o) in enumerate(self.tqdm(triplet_iter, desc='Building formatted edge list'), 1):
			info_tuple = (s['concept'], p, o['concept'])
			s_cp, p_cp, o_cp = info_tuple
			s_id, p_id, o_id = map(get_concept_id, info_tuple)
			if not uri_is_clause(p_id):
				p_id += '{obj}' # IMPORTANT! This will help recognizing clauses
			s_lb, p_lb, o_lb = map(get_concept_label, info_tuple)
			source_sentence_text = p['source']['sentence_text']
			source_sentence_uri = ANONYMOUS_PREFIX+get_uri_from_txt(source_sentence_text)
			# add triplet and labels
			formatted_edge_list.extend((
				(s_id, p_id, o_id),
				(s_id, HAS_LABEL_PREDICATE, s_lb),
				(o_id, HAS_LABEL_PREDICATE, o_lb),
			))
			if add_predicates_label:
				formatted_edge_list.append((p_id, HAS_LABEL_PREDICATE, p_lb))
			# add verbs
			if add_verbs:
				v_cp = p['verb']
				if v_cp:
					v_id = get_concept_id(v_cp)
					v_source_text = get_concept_source_text(v_cp)
					v_source_paragraph_uri = ANONYMOUS_PREFIX+get_uri_from_txt(v_source_text)
					formatted_edge_list.extend((
						(p_id, HAS_VERB_PREDICATE, v_id),

						(v_id, HAS_LABEL_PREDICATE, get_concept_label(v_cp)),
						(v_id, HAS_SPAN_ID_PREDICATE, v_source_paragraph_uri),

						(v_source_paragraph_uri, HAS_SOURCE_ID_PREDICATE, source_sentence_uri),
						(v_source_paragraph_uri, HAS_SOURCE_LABEL_PREDICATE, v_source_text),
					))
			# add annotations and sources, only once (many triplets may have the same source)
			paragraph = p['source'].get('paragraph_text', None)
			doc_id = p['source'].get('doc', None)
			if doc_id:
				doc_id = doc_id[len(DOC_PREFIX):] if doc_id.startswith(DOC_PREFIX) else get_uri_from_txt(doc_id)
				doc_uri = DOC_PREFIX+doc_id
			else:
				doc_uri = p['source'].get('doc_uri', None)
				doc_id = doc_uri[len(DOC_PREFIX):]
			assert doc_uri, p['source']
			source_key = (doc_id, paragraph if paragraph else source_sentence_text)
			source_paragraph_uri = paragraph_dict.get(source_key, None)
			if source_paragraph_uri is None:
				source_paragraph_uri = f'{ANONYMOUS_PREFIX}{doc_id}_{get_uri_from_txt(source_key[-1])}'
				# add annotations
				annotation = p['source'].get('annotation', None)
				if annotation:
					if annotation.get('root', None):
						annotation_edge_list = (
							(s if s!= annotation['root'] else source_paragraph_uri, p, o if o!= annotation['root'] else source_paragraph_uri)
							for s,p,o in annotation['content']
						)
					else:
						annotation_edge_list = annotation['content']
					formatted_edge_list.extend(annotation_edge_list)
				# add sources
				paragraph_dict[source_key] = source_paragraph_uri
				if paragraph:
					formatted_edge_list.append((source_paragraph_uri, HAS_CONTENT_PREDICATE, paragraph))
				if doc_uri:
					formatted_edge_list.append((source_paragraph_uri, DOC_ID_PREDICATE, doc_uri))
				language_id = p['source'].get('language_id', None)
				if language_id:
					formatted_edge_list.append((source_paragraph_uri, LANGUAGE_ID_PREDICATE, language_id))
			# # connect triples to sources
			# formatted_edge_list.extend((
			# 	(s_id, HAS_PARAGRAPH_ID_PREDICATE, source_paragraph_uri),
			# 	(p_id, HAS_PARAGRAPH_ID_PREDICATE, source_paragraph_uri),
			# 	(o_id, HAS_PARAGRAPH_ID_PREDICATE, source_paragraph_uri),
			# ))
			# add source spans and sentences
			source_span_text = get_concept_source_text(p)
			source_span_uri = ANONYMOUS_PREFIX+get_uri_from_txt(source_span_text)
			formatted_edge_list.extend((
				(s_id, HAS_SPAN_ID_PREDICATE, source_span_uri),
				(p_id, HAS_SPAN_ID_PREDICATE, source_span_uri),
				(o_id, HAS_SPAN_ID_PREDICATE, source_span_uri),
				((s_id,p_id,o_id), HAS_SPAN_ID_PREDICATE, source_span_uri),

				(source_span_uri, HAS_SOURCE_ID_PREDICATE, source_sentence_uri),
				(source_span_uri, HAS_SOURCE_LABEL_PREDICATE, source_span_text),
				
				(source_sentence_uri, HAS_PARAGRAPH_ID_PREDICATE, source_paragraph_uri),
				(source_sentence_uri, HAS_SOURCE_LABEL_PREDICATE, source_sentence_text),
			))
			if ii%100000==0:
				formatted_edge_list = list(unique_everseen(formatted_edge_list))
		del paragraph_dict
		# Abstract triples
		if use_wordnet or use_framenet_fe or add_subclasses:
			# add subclasses
			if add_subclasses:
				concept_dict_list = flatten(((edge[0] for edge in triplet_iter), (edge[-1] for edge in triplet_iter)), as_list=True)
				# print(0, [(c['concept'], c['concept_core']) for c in concept_list])
				for concept_dict in unique_everseen(concept_dict_list, key=lambda x: get_concept_label(x['concept'])):
					if not concept_dict['concept_core']:
						continue
					related_concept_tuple = (concept_dict['concept'],)+concept_dict['concept_core']
					for i,a in enumerate(related_concept_tuple):
						for b in related_concept_tuple[i+1:]:
							if a['lemma'] == b['lemma']: # ignore autoreferential relations
								continue
							if len(a['lemma']) < len(b['lemma']):
								smaller_concept = a
								larger_concept = b
							else:
								smaller_concept = b
								larger_concept = a
							if smaller_concept['lemma'] in larger_concept['lemma']:
								formatted_edge_list.append(
									(get_concept_id(larger_concept), SUBCLASSOF_PREDICATE, get_concept_id(smaller_concept))
								)
				formatted_edge_list.extend((
					(get_concept_id(concept_core), HAS_LABEL_PREDICATE, get_concept_label(concept_core))
					for concept_dict in concept_dict_list
					for concept_core in concept_dict['concept_core']
				))
		# remove duplicates
		formatted_edge_list = list(unique_everseen(formatted_edge_list))
		return formatted_edge_list

	def get_graph_hinge(self, source_graph, source_valid_concept_filter_fn, word_retriever):
		# Get concept_description_dicts
		self.logger.info('Graph Hinge: Get concept_description_dicts')
		source_concept_description_dict = get_concept_description_dict(
			graph= source_graph, 
			label_predicate= HAS_LABEL_PREDICATE, 
			valid_concept_filter_fn= source_valid_concept_filter_fn,
		)
		if not source_concept_description_dict:
			return []
		# Classify source graph labels
		source_uri_list, source_label_list = zip(*(
			(uri,label)
			for uri, label_list in source_concept_description_dict.items()
			for label in label_list
		))
		self.logger.info('Graph Hinge: Classify source graph labels')
		similarity_dict_generator_list = word_retriever.retrieve(source_label_list)
		# Build the graph hinge
		self.logger.info('Graph Hinge: Build the graph hinge')
		graph_hinge = []
		for source_concept_uri, similarity_dict_generator in zip(source_uri_list, similarity_dict_generator_list):
			similarity_dict = next(similarity_dict_generator,None)
			if not similarity_dict:
				continue
			target_concept_uri = similarity_dict['id']
			if target_concept_uri != source_concept_uri:
				graph_hinge.append((target_concept_uri, IN_SYNSET_PREDICATE, source_concept_uri))
		return graph_hinge

	def build(self, triplet_tuple=None, max_syntagma_length=None, **graph_extraction_options):
		if not triplet_tuple and not self.triplet_tuple:
			self.logger.warning('No couples found')
			return []
		if triplet_tuple is None:
			triplet_tuple = self.triplet_tuple
			if max_syntagma_length:
				triplet_tuple = tuple(filter(lambda x: self.is_valid_syntagm(x['concept']['lemma'], max_syntagma_length), triplet_tuple))
		self.logger.info('Getting edge list..')
		# edge_list_fn = self.parallel_get_edge_list if parallel_extraction else self.get_edge_list
		edge_list = self.get_edge_list(triplet_tuple, **graph_extraction_options)
		assert edge_list, 'No edge_list found'
		# external_graph = []
		if self.graph_tuple:
			self.logger.info(f'Getting {len(self.graph_tuple)} graph hinges..')
			edge_list += unique_everseen(flatten(self.graph_tuple))
			# target_concept_description_dict = get_concept_description_dict(
			# 	graph= edge_list, 
			# 	label_predicate= HAS_LABEL_PREDICATE, 
			# 	valid_concept_filter_fn=triplet_is_clause,
			# )
			# # Build concept classifier using the target_graph
			# word_retriever = WordRetriever(self.model_options).set_concept_description_dict(target_concept_description_dict)
			# edge_list += self.get_graph_hinge(
			# 	external_graph, 
			# 	lambda x: not (x[0].startswith(DOC_PREFIX) or x[0].startswith(ANONYMOUS_PREFIX)), 
			# 	word_retriever,
			# )
			# print(json.dumps(external_graph, indent=4))
		return edge_list

