from knowpy.models.model_manager import ModelManager
from knowpy.misc.adjacency_list import AdjacencyList
from knowpy.models.knowledge_extraction.couple_extractor import CoupleExtractor
from knowpy.misc.graph_builder import get_concept_description_dict, get_subject_set
from knowpy.misc.jsonld_lib import *
from knowpy.misc.utils import *

from knowpy.misc.graph_builder import get_concept_set, get_predicate_set, get_object_set, get_ancestors, filter_graph_by_root_set, tuplefy

try:
	from nltk.corpus import wordnet as wn
	from nltk.corpus import brown
	from nltk.corpus import stopwords
except OSError:
	print('Downloading nltk::wordnet\n'
		"(don't worry, this will only happen once)")
	import nltk
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	nltk.download('brown')
	nltk.download('stopwords')
	from nltk.corpus import stopwords
	from nltk.corpus import wordnet as wn
	from nltk.corpus import brown
from nltk import FreqDist
import unidecode


singlefy = lambda s: unidecode.unidecode(s.strip().replace("\n"," "))#.capitalize()
# singlefy = lambda s: s.strip().replace("\n"," ")#.capitalize()
word_frequency_distribution = FreqDist(i.lower() for i in brown.words())
is_common_word = lambda w: word_frequency_distribution.freq(w) >= 1e-4

class KnowledgeGraphManager(ModelManager):

	def __init__(self, model_options, graph):
		super().__init__(model_options)
		self.disable_spacy_component = ["ner", "textcat"]
		ModelManager.logger.info('Initialising KnowledgeGraphManager')
		# self.graph = graph
		# self.clauses_list = tuple(unique_everseen(filter(triplet_is_clause, graph))) # comment this out for saving memory
		self.graph_clauses_count = len(tuple(unique_everseen(filter(triplet_is_clause, graph))))

		# self.adjacency_list = AdjacencyList(
		# 	graph, 
		# 	equivalence_relation_set=set([IN_SYNSET_PREDICATE,IS_EQUIVALENT_PREDICATE]),
		# 	is_sorted=True,
		# )
		self.min_triplet_len = model_options.get('min_triplet_len',1)
		self.max_triplet_len = model_options.get('max_triplet_len',float('inf'))
		self.min_sentence_len = model_options.get('min_sentence_len',1)
		self.max_sentence_len = model_options.get('max_sentence_len',float('inf'))
		self.min_paragraph_len = model_options.get('min_paragraph_len',1)
		self.max_paragraph_len = model_options.get('max_paragraph_len',float('inf'))
		self.adjacency_list = AdjacencyList(
			graph, 
			equivalence_relation_set=set([IS_EQUIVALENT_PREDICATE]),
			is_sorted=False,
		)
		self._content_dict = None
		self._paragraph_dict = None
		self._doc_dict = None
		self._label_dict = None
		self._language_dict = None

		self._source_span_dict = None
		self._source_sentence_dict = None
		self._source_label_dict = None
		# self.verb_dict = self.adjacency_list.get_predicate_dict(HAS_VERB_PREDICATE)

		ModelManager.logger.info('KnowledgeGraphManager initialised')

	@property
	def content_dict(self):
		if not self._content_dict:
			self.logger.info("Building content_dict")
			self._content_dict = self.adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy)
			self.logger.info("Building content_dict: done")
		return self._content_dict

	@property
	def language_dict(self):
		if not self._language_dict:
			self.logger.info("Building language_dict")
			self._language_dict = self.adjacency_list.get_predicate_dict(LANGUAGE_ID_PREDICATE)
			self.logger.info("Building language_dict: done")
		return self._language_dict

	@property
	def paragraph_dict(self):
		if not self._paragraph_dict:
			self.logger.info("Building paragraph_dict")
			self._paragraph_dict = self.adjacency_list.get_predicate_dict(HAS_PARAGRAPH_ID_PREDICATE)
			for v_list in self._paragraph_dict.values():
				v_list.sort(key=len) # sort in-place
			self.logger.info("Building paragraph_dict: done")
		return self._paragraph_dict

	@property
	def doc_dict(self):
		if not self._doc_dict:
			self.logger.info("Building doc_dict")
			self._doc_dict = self.adjacency_list.get_predicate_dict(DOC_ID_PREDICATE)
			self.logger.info("Building doc_dict: done")
		return self._doc_dict

	@property
	def label_dict(self):
		if not self._label_dict:
			self.logger.info("Building label_dict")
			self._label_dict = self.adjacency_list.get_predicate_dict(HAS_LABEL_PREDICATE, singlefy)
			for v_list in self._label_dict.values():
				v_list.sort(key=len) # sort in-place
			self.logger.info("Building label_dict: done")
		return self._label_dict

	@property
	def source_span_dict(self):
		if not self._source_span_dict:
			self.logger.info("Building source_span_dict")
			self._source_span_dict = self.adjacency_list.get_predicate_dict(HAS_SPAN_ID_PREDICATE)
			self.logger.info("Building source_span_dict: done")
		return self._source_span_dict

	@property
	def source_sentence_dict(self):
		if not self._source_sentence_dict:
			self.logger.info("Building source_sentence_dict")
			self._source_sentence_dict = self.adjacency_list.get_predicate_dict(HAS_SOURCE_ID_PREDICATE)
			self.logger.info("Building source_sentence_dict: done")
		return self._source_sentence_dict

	@property
	def source_label_dict(self):
		if not self._source_label_dict:
			self.logger.info("Building source_label_dict")
			self._source_label_dict = self.adjacency_list.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy)
			self.logger.info("Building source_label_dict: done")
		return self._source_label_dict	

	# @staticmethod
	# def build_from_edus_n_clauses(model_options, graph=None, kg_builder_options=None, qa_dict_list=None, qa_extractor_options=None, qa_type_to_use=None, use_only_elementary_discourse_units=False, edu_graph=None):
	# 	ModelManager.logger.info('build_from_edus_n_clauses')
	# 	if edu_graph is None:
	# 		assert qa_extractor_options is not None, 'if no edu_graph is passed, then qa_extractor_options must not be None'
	# 		assert kg_builder_options is not None, 'if no edu_graph is passed, then kg_builder_options must not be None'
	# 		assert graph is not None, 'if no edu_graph is passed, then graph must not be None'
	# 		edu_graph = QuestionAnswerExtractor(qa_extractor_options).extract_aligned_graph_from_qa_dict_list(graph, kg_builder_options, qa_dict_list=qa_dict_list, qa_type_to_use=qa_type_to_use)
	# 	assert graph, 'graph is missing'
	# 	# Remove invalid labels: All the valid labels of EDU-graph are included in the original graph, except for the extra ones coming from questions (i.e., templates). Thus, removing all the labels from EDU-graph will prevent to consider invalid labels as important aspects
	# 	if use_only_elementary_discourse_units:
	# 		edu_graph_label_set = get_subject_set(filter(lambda x: x[1]==HAS_LABEL_PREDICATE, edu_graph))
	# 		graph_label_set = get_subject_set(filter(lambda x: x[1]==HAS_LABEL_PREDICATE, graph))
	# 		valid_label_set = edu_graph_label_set.intersection(graph_label_set)
	# 		edu_graph = list(filter(lambda x: x[1]!=HAS_LABEL_PREDICATE or x[0] in valid_label_set, edu_graph))
	# 		# paragraph_id_graph = list(filter(lambda x: x[1]==HAS_PARAGRAPH_ID_PREDICATE, graph))
	# 		# edu_graph += paragraph_id_graph
	# 		# source_span_uri_set = get_subject_set(paragraph_id_graph)
	# 		# edu_graph += list(filter(lambda x: x[1]==HAS_SOURCE_LABEL_PREDICATE and x[0] in source_span_uri_set, graph))
	# 		# edu_graph += filter(lambda x: x[1]==HAS_CONTENT_PREDICATE, graph)
	# 	else: # Merge with graph
	# 		edu_graph = list(filter(lambda x: x[1]!=HAS_LABEL_PREDICATE, edu_graph))
	# 		edu_graph = list(unique_everseen(edu_graph + graph))
	# 	return KnowledgeGraphManager(model_options, edu_graph)

	@property
	def concept_description_dict(self):
		return {
			uri: list(unique_everseen(filter(lambda x: x.strip(), label_list), key=lambda x: x.lower()))
			for uri, label_list in self.label_dict.items()
			if not uri_is_clause(uri) # no predicates
			and not self.is_stopword(uri) # no stopwords
			and not is_number(self.get_label(uri)) # no numbers
			and len(self.get_label(uri)) > 2 # no syllables
			and not (uri.startswith(DOC_PREFIX) or uri.startswith(ANONYMOUS_PREFIX)) # no files or anonymous entities
		}

	def get_aspect_uri_language(self, aspect_uri):
		span_uri_list = self.source_span_dict.get(aspect_uri,None)
		if not span_uri_list:
			return None
		span_uri = span_uri_list[0]
		sentence_uri_list = self.source_sentence_dict.get(span_uri,None)
		if not sentence_uri_list:
			return None
		sentence_uri = sentence_uri_list[0]
		paragraph_uri_list = self.paragraph_dict.get(sentence_uri,None)
		if not paragraph_uri_list:
			return None
		paragraph_uri = paragraph_uri_list[0]
		language_list = self.language_dict.get(paragraph_uri,None)
		# language_list = list(filter(self.language_is_available, language_list))
		if not language_list:
			return None
		return language_list[0]

	@property
	def aspect_uri_list(self):
		return list(self.concept_description_dict.keys())

	def get_source_paragraph_set(self, uri):
		return set((
			source_paragraph_id
			for source_span_uri in self.source_span_dict.get(uri,[])
			for source_sentence_uri in self.source_sentence_dict[source_span_uri]
			for source_paragraph_id in self.paragraph_dict[source_sentence_uri]
		))

	def get_source_span_set(self, uri):
		return set(self.source_span_dict.get(uri,[]))

	def get_source_span_label_iter(self, uri):
		return (
			source_span_label
			for source_span_uri in self.source_span_dict.get(uri,[])
			for source_span_label in self.source_label_dict[source_span_uri]
		)

	def get_source_span_label_set(self, uri):
		return set(self.get_source_span_label_iter(uri))

	def get_source_span_label(self, uri):
		return next(iter(self.get_source_span_label_iter(uri)), None)

	def get_edge_source_span_label_set(self, s,p,o):
		return self.get_source_span_label_set(p).intersection(self.get_source_span_label_set(s)).intersection(self.get_source_span_label_set(o))

	def get_edge_source_span_label(self, s,p,o):
		edge_source_span_label_set = self.get_edge_source_span_label_set(s,p,o)
		if edge_source_span_label_set:
			return min(edge_source_span_label_set, key=len)
		return None

	def get_label_iter(self, concept_uri, explode_if_none=True):
		for c in self.get_equivalent_concepts(concept_uri):
			if c in self.label_dict:
				for l in self.label_dict[c]:
					yield l
			elif c.startswith(WORDNET_PREFIX):
				for l in map(lambda x: explode_concept_key(x.name()), wn.synset(c[len(WORDNET_PREFIX):]).lemmas()):
					yield l
			elif explode_if_none:
				exploded_key = explode_concept_key(concept_uri)
				if exploded_key:
					yield exploded_key

	def get_label_list(self, concept_uri, explode_if_none=True):
		return list(self.get_label_iter(concept_uri, explode_if_none=explode_if_none))

	def get_label(self, concept_uri, explode_if_none=True):
		return next(self.get_label_iter(concept_uri, explode_if_none=explode_if_none), None) # always the shortest label

	def is_stopword(self, aspect_uri):
		aspect_language = language_map.get(self.get_aspect_uri_language(aspect_uri), self.model_options.get('language','English')).lower()
		if aspect_language not in stopwords.fileids(): # language not available
			return False
		stopwords_set = set(stopwords.words(aspect_language))
		return any(
			label in stopwords_set
			for label in map(lambda x: x.lower(), self.get_label_list(aspect_uri))
		)

	def is_relevant_aspect(self, aspect_uri, ignore_leaves=False):
		# ignore stopwords
		if self.is_stopword(aspect_uri):
			self.logger.debug(f'Removing <{aspect_uri}>: is_stopword')
			return False
		# ignore numbers
		if is_number(self.get_label(aspect_uri)):
			self.logger.debug(f'Removing <{aspect_uri}>: is_number')
			return False
		# concepts without sources are irrelevant
		source_set = self.get_source_paragraph_set(aspect_uri)
		if not source_set:
			self.logger.debug(f'Removing <{aspect_uri}>: has_no_triplets')
			return False
		# concepts with less than 2 sources are leaves with no triplets: safely ignore them, they are included by a super-class that is necessarily explored before them and they are less relevant than it
		if ignore_leaves:
			if len(source_set) <= 1:
				self.logger.debug(f'Removing <{aspect_uri}>: is_leaf')
				return False
		# concepts with the same sources of one of their sub-classes are redundant as those with no sources
		aspect_set = set([aspect_uri])
		subclass_set = self.get_sub_classes(aspect_set, depth=1) - aspect_set
		if not subclass_set:
			return True
		is_relevant = next(filter(lambda x: source_set-self.get_source_paragraph_set(x), subclass_set), None) is not None
		if not is_relevant:
			self.logger.debug(f'Removing <{aspect_uri}>: is_same_as_subclass')
			return False
		return True

	def get_sub_graph(self, uri, depth=None, predicate_filter_fn=lambda x: x != SUBCLASSOF_PREDICATE and not uri_is_clause(x)):
		uri_set = self.adjacency_list.get_predicate_chain(
			set([uri]), 
			direction_set=['out'], 
			depth=depth, 
			predicate_filter_fn=predicate_filter_fn
		)
		return list(unique_everseen((
			(s,p,o)
			for s in uri_set
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(s)
		)))

	def get_equivalent_concepts(self, concept_uri):
		return list(self.adjacency_list.equivalence_matrix.get(concept_uri,[]))+[concept_uri]

	def get_sub_classes(self, concept_set, depth=None):
		return self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
			direction_set = ['in'],
			depth = depth,
		)

	def get_super_classes(self, concept_set, depth=None):
		return self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
			direction_set = ['out'],
			depth = depth,
		)

	def get_aspect_graph(self, concept_uri, add_external_definitions=False, include_super_concepts_graph=False, include_sub_concepts_graph=False, consider_incoming_relations=False, depth=None, filter_fn=lambda x: x):
		concept_set = set(self.get_equivalent_concepts(concept_uri))
		expanded_concept_set = set(concept_set)
		# Get sub-classes
		if include_sub_concepts_graph:
			expanded_concept_set |= self.get_sub_classes(concept_set, depth=depth)
		# Get super-classes
		if include_super_concepts_graph:
			expanded_concept_set |= self.get_super_classes(concept_set, depth=depth)
		# expanded_concept_set = sorted(expanded_concept_set) # this would improve caching, later on
		# Add outcoming relations to concept graph
		expanded_aspect_graph = (
			(s,p,o)
			for s in expanded_concept_set
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(s)
		)
		expanded_aspect_graph = list(filter(filter_fn,expanded_aspect_graph))
		# Add incoming relations to concept graph
		if consider_incoming_relations:
			expanded_aspect_graph += filter(filter_fn, (
				(s,p,o)
				for o in expanded_concept_set
				for p,s in self.adjacency_list.get_incoming_edges_matrix(o)
			))
		# print(concept_uri, json.dumps(expanded_aspect_graph, indent=4))
		# Add external definitions
		if add_external_definitions:
			# Add wordnet's definition
			for equivalent_concept_uri in filter(lambda x: x.startswith(WORDNET_PREFIX), self.adjacency_list.equivalence_matrix.get(concept_uri,[])):
				synset = wn.synset(equivalent_concept_uri[len(WORDNET_PREFIX):]) # remove string WORDNET_PREFIX, 3 chars
				definition = synset.definition()
				expanded_aspect_graph.append((concept_uri,HAS_DEFINITION_PREDICATE,definition))
			# Add wikipedia's (short) definition
			# try:
			# 	definition = wikipedia.summary(
			# 		self.get_label(concept_uri), 
			# 		sentences=1, # short
			# 		chars=0,
			# 		auto_suggest=True, 
			# 		redirect=True
			# 	)
			# 	expanded_aspect_graph.append((concept_uri,HAS_DEFINITION_PREDICATE,definition))
			# except:
			# 	pass
		return list(unique_everseen(expanded_aspect_graph))

	def get_paragraph_text(self, source_paragraph_id):
		paragraph_text_list = self.content_dict.get(source_paragraph_id, None) # check if any paragraph is available
		return paragraph_text_list[0] if paragraph_text_list else None

	def get_clause_set_from_uri(self, uri):
		return set(
			(uri,p,o)
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(uri)
			if uri_is_clause(p)
		).union(
			(s,p,uri)
			for p,s in self.adjacency_list.get_incoming_edges_matrix(uri)
			if uri_is_clause(p)
		)

	def get_paragraph_list_from_uri(self, uri):
		self.logger.info("Running get_paragraph_list_from_uri")
		original_triplet_list = self.get_clause_set_from_uri(uri)
		def paragraph_list_gen():
			source_span_uri_iter = (
				source_span_uri
				for original_triplet in original_triplet_list
				for source_span_uri in self.source_span_dict.get(original_triplet, [])
			)
			source_span_uri_iter = unique_everseen(source_span_uri_iter)
			for source_span_uri in source_span_uri_iter:
				for source_sentence_uri in self.source_sentence_dict[source_span_uri]:
					for source_paragraph_uri in self.paragraph_dict[source_sentence_uri]:
						for paragraph_text in self.content_dict.get(source_paragraph_uri, []):
							if paragraph_text:
								yield paragraph_text
		return list(unique_everseen(paragraph_list_gen()))

	def get_sourced_graph_from_triplet_list(self, original_triplet_list):
		self.logger.info("Running get_sourced_graph_from_triplet_list")
		if self.max_triplet_len <= 0:
			return []
		if self.max_sentence_len <= 0 and self.max_paragraph_len <= 0:
			return []
		def graph_gen():
			source_span_uri_n_triplet_iter = (
				(source_span_uri, original_triplet)
				for original_triplet in self.tqdm(original_triplet_list, desc='Building sourced graph from triplet list')
				for source_span_uri in self.source_span_dict.get(original_triplet, [])
			)
			source_span_uri_n_triplet_iter = unique_everseen(source_span_uri_n_triplet_iter, key=lambda x: x[0])
			for source_span_uri, original_triplet in source_span_uri_n_triplet_iter:
				triplet_text = self.source_label_dict[source_span_uri][0]
				if not (self.min_triplet_len <= len(triplet_text) <= self.max_triplet_len):
					continue
				if self.max_sentence_len > 0:
					for source_sentence_uri in self.source_sentence_dict[source_span_uri]:
						sentence_text_list = self.source_label_dict.get(source_sentence_uri, None)
						sentence_text = sentence_text_list[0] if sentence_text_list else None
						if sentence_text and self.min_sentence_len <= len(sentence_text) <= self.max_sentence_len:
							source_paragraph_uri = self.paragraph_dict[source_sentence_uri][0]
							yield (
								triplet_text,
								sentence_text,
								original_triplet,
								(source_sentence_uri, source_paragraph_uri, self.doc_dict[source_paragraph_uri][0]),
							)
				if self.max_paragraph_len > 0:
					for source_sentence_uri in self.source_sentence_dict[source_span_uri]:
						for source_paragraph_uri in self.paragraph_dict[source_sentence_uri]:
							paragraph_text = self.get_paragraph_text(source_paragraph_uri)
							if paragraph_text and self.min_paragraph_len <= len(paragraph_text) <= self.max_paragraph_len:
								yield (
									triplet_text,
									paragraph_text,
									original_triplet,
									(None, source_paragraph_uri, self.doc_dict[source_paragraph_uri][0]),
								)
		return list(unique_everseen(graph_gen(), key=lambda x: (x[0],x[1])))

	def get_sourced_graph(self):
		return self.get_sourced_graph_from_triplet_list(self.source_span_dict.keys())

	def get_taxonomical_view(self, concept_uri, depth=None, with_internal_definitions=True, concept_id_filter=None):
		if not concept_id_filter:
			concept_id_filter = lambda x: x
		concept_set = set((concept_uri,))
		if depth != 0:
			concept_set |= self.get_sub_classes(concept_set, depth=depth)
			concept_set |= self.get_super_classes(concept_set, depth=depth)
		concept_set = set(filter(concept_id_filter,concept_set))
		# Add subclassof relations
		taxonomical_view = set(
			(s,p,o)
			for s in concept_set
			for p,o in self.adjacency_list.get_outcoming_edges_matrix(s)
			if p == SUBCLASSOF_PREDICATE and concept_id_filter(o)
		).union(
			(s,p,o)
			for o in concept_set
			for p,s in self.adjacency_list.get_incoming_edges_matrix(o)
			if p == SUBCLASSOF_PREDICATE and concept_id_filter(s)
		)
		taxonomical_view = list(taxonomical_view)
		taxonomy_concept_set = get_concept_set(taxonomical_view).union(concept_set)
		# Add labels
		taxonomical_view += (
			(concept, HAS_LABEL_PREDICATE, self.get_label(concept, explode_if_none=False))
			for concept in taxonomy_concept_set
		)
		# # Add sources
		# taxonomical_view += (
		# 	(concept, HAS_PARAGRAPH_ID_PREDICATE, source)
		# 	for concept in taxonomy_concept_set
		# 	for source in self.paragraph_dict.get(concept,())
		# )
		# for concept in taxonomy_concept_set:
		# 	for source in self.paragraph_dict.get(concept,()):
		# 		taxonomical_view += self.get_sub_graph(source)
		# # Add wordnet definitions
		# taxonomical_view += (
		# 	(concept, HAS_DEFINITION_PREDICATE, wn.synset(concept[3:]).definition())
		# 	for concept in filter(lambda x: x.startswith(WORDNET_PREFIX), taxonomy_concept_set)
		# )
		# Add definitions
		if with_internal_definitions:
			taxonomical_view += unique_everseen(
				(concept_uri,p,o)
				for p,o in self.adjacency_list.get_outcoming_edges_matrix(concept_uri)
				if p == HAS_DEFINITION_PREDICATE
			)
		# Add types
		sub_types_set = self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == HAS_TYPE_PREDICATE, 
			direction_set = ['out'],
			depth = 0,
		)
		super_types_set = self.adjacency_list.get_predicate_chain(
			concept_set = concept_set, 
			predicate_filter_fn = lambda x: x == HAS_TYPE_PREDICATE, 
			direction_set = ['in'],
			depth = 0,
		)
		taxonomical_view += (
			(concept_uri,HAS_TYPE_PREDICATE,o)
			for o in sub_types_set - concept_set
		)
		taxonomical_view += (
			(s,HAS_TYPE_PREDICATE,concept_uri)
			for s in super_types_set - concept_set
		)
		taxonomical_view += unique_everseen(
			(s, HAS_LABEL_PREDICATE, self.get_label(s, explode_if_none=False))
			for s in (super_types_set | sub_types_set) - concept_set
		)
		taxonomical_view = filter(lambda x: x[0] and x[1] and x[2], taxonomical_view)
		# print(taxonomical_view)
		return list(taxonomical_view)

	def get_paragraph_id_from_concept_id(self, x):
		span_id = self.source_span_dict.get(x,[None])[0]
		if not span_id:
			return []
		sentence_id = self.source_sentence_dict.get(span_id,[None])[0]
		if not sentence_id:
			return []
		return self.paragraph_dict[sentence_id][0]

	def get_sourced_graph_from_labeled_graph(self, label_graph):
		sourced_natural_language_triples_set = []

		def extract_sourced_graph(label_graph, str_fn):
			result = []
			for labeled_triple, original_triple in label_graph:
				naturalized_triple = str_fn(labeled_triple)
				# print(naturalized_triple, labeled_triple)
				if not naturalized_triple:
					continue
				context_set = self.get_paragraph_id_from_concept_id(original_triple)
				result += (
					(
						naturalized_triple, 
						self.get_paragraph_text(source_paragraph_uri), # source_label
						original_triple,
						(None, source_paragraph_uri, self.doc_dict[source_paragraph_uri][0]),
					)
					for source_paragraph_uri in context_set
				)
			return result
		sourced_natural_language_triples_set += extract_sourced_graph(label_graph, get_string_from_triple)
		sourced_natural_language_triples_set += extract_sourced_graph(label_graph, lambda x: x[0])
		sourced_natural_language_triples_set += extract_sourced_graph(label_graph, lambda x: x[-1])
		sourced_natural_language_triples_set = list(unique_everseen(sourced_natural_language_triples_set))
		return sourced_natural_language_triples_set

	def get_labeled_graph_from_concept_graph(self, concept_graph):
		def labeled_triples_gen():
			# Get labeled triples
			for original_triple in concept_graph:
				s,p,o = original_triple
				# if s == o:
				# 	continue
				# p_is_subclassof = p == SUBCLASSOF_PREDICATE
				# if p_is_subclassof: # remove redundant triples not directly concerning the concept
				# 	if o!=concept_uri and s!=concept_uri:
				# 		continue
				for label_p in self.label_dict.get(p,[p]):
					label_p_context = set(self.get_paragraph_id_from_concept_id(p)) # get label sources
					for label_s in self.label_dict.get(s,[s]):
						if label_p_context: # triple with sources
							label_s_context = self.get_paragraph_id_from_concept_id(s) # get label sources
							label_context = label_p_context.intersection(label_s_context)
							if not label_context: # these two labels do not appear together, skip
								continue
						for label_o in self.label_dict.get(o,[o]):
							if label_p_context: # triple with sources
								label_o_context = self.get_paragraph_id_from_concept_id(o) # get label sources
								if not label_context.intersection(label_o_context): # these labels do not appear together, skip
									continue
							# if p_is_subclassof and labels_are_similar(label_s,label_o):
							# 	continue
							labeled_triple = (label_s,label_p,label_o)
							yield (labeled_triple, original_triple)
		return unique_everseen(labeled_triples_gen())

	def get_subclass_replacer(self, superclass):
		superclass_set = set([superclass])
		subclass_set = self.get_sub_classes(superclass_set, depth=1).difference(superclass_set)
		# print(subclass_set)
		exploded_superclass = explode_concept_key(superclass).strip().lower()
		exploded_subclass_iter = map(lambda x: explode_concept_key(x).strip().lower(), subclass_set)
		exploded_subclass_iter = filter(lambda x: x and not x.startswith(exploded_superclass), exploded_subclass_iter)
		exploded_subclass_list = sorted(exploded_subclass_iter, key=lambda x:len(x), reverse=True)
		# print(exploded_subclass_list)
		if len(exploded_subclass_list) == 0:
			return None
		# print(exploded_superclass, exploded_subclass_list)
		subclass_regexp = re.compile('|'.join(exploded_subclass_list))
		return lambda x,triple: re.sub(subclass_regexp, exploded_superclass, x) if triple[1]!=SUBCLASSOF_PREDICATE else x

	# def get_noun_set(self, graph):
	# 	noun_set = set()
	# 	concept_list = list(get_concept_set(graph))
	# 	concept_label_list = list(map(self.get_label, concept_list))
	# 	span_list = self.nlp(concept_label_list)
	# 	for concept, span in zip(concept_list, span_list):
	# 		# print(concept, span)
	# 		if not span:
	# 			continue
	# 		for token in span:
	# 			if token.pos_ == 'NOUN':
	# 				noun_set.add(concept)
	# 				break
	# 	return noun_set
