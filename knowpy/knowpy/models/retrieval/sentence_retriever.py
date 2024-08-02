import numpy as np # for fast array ops
from knowpy.models.model_manager import ModelManager
import hnswlib
import os

class SentenceRetriever(ModelManager):

	def __init__(self, model_options, has_immutable_db=True):
		super().__init__(model_options)
		self.disable_spacy_component = ["ner", "textcat"]
		self.sentence_embedding_fn = self.get_default_embedder()
		self.similarity_fn = self.get_default_similarity_fn()
		self.default_similarity_threshold = self.model_options.get('default_similarity_threshold', 0)
		
		self.has_immutable_db = has_immutable_db
		fast_knn_search_options = model_options.get('fast_knn_search_options',{})
		self.fast_knn_search_threshold = max(1,fast_knn_search_options.get('knn_activation_threshold',100000))
		self.fast_knn_search = None
		self.space = 'cosine' if self.get_default_similarity_fn_type() == 'cosine_similarity' else 'ip'
		self._index = None
		self._ef_construction = fast_knn_search_options.get('ef_construction',200)
		self._ef_search = fast_knn_search_options.get('ef_search',150)
		self._max_number_of_connections = fast_knn_search_options.get('max_number_of_connections',16)
		self._random_seed = fast_knn_search_options.get('random_seed',41)

		self.logger.info(f'Initialising SentenceRetriever with has_immutable_db={self.has_immutable_db}, fast_knn_search_threshold={self.fast_knn_search_threshold}')

	def store_cache(self, cache_name=None, **args):
		if not cache_name:
			cache_name = self.default_cache_path
		assert cache_name, 'A cache must be set either as a parameter of this function or in the model options during init'
		super().store_cache(cache_name, **args)
		if self.has_immutable_db and self._index:
			self.logger.info(f"Storing hnswlib.index <{cache_name+'.hnswlib.index'}>")
			self._index.save_index(cache_name+'.hnswlib.index')

	def load_cache(self, cache_name=None):
		if not cache_name:
			cache_name = self.default_cache_path
		assert cache_name, 'A cache must be set either as a parameter of this function or in the model options during init'
		if self.has_immutable_db:
			if os.path.exists(cache_name+'.hnswlib.index'):
				self.logger.info(f"Loading hnswlib.index <{cache_name+'.hnswlib.index'}>")
				dim = self.sentence_embedding_fn([('abstract','context')], without_context=False)[0].shape[-1]
				self._index = hnswlib.Index(space=self.space, dim=dim)
				self._index.set_num_threads(self.n_threads) # multithreading in hnswlib seems to cause random crashes
				self._index.load_index(cache_name+'.hnswlib.index')
				self._index.set_ef(self._ef_search) # ef should always be > top_k_hits # Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.
				loaded_cache = True
			else:
				super().load_cache(cache_name)
				loaded_cache = False
		else:
			loaded_cache = super().load_cache(cache_name)
		return loaded_cache

	def set_documents(self, id_list, doc_list, context_list=None, **args):
		assert id_list and doc_list
		assert len(id_list) == len(doc_list)
		self.logger.info(f"SentenceRetriever: setting {len(doc_list)} documents")
		if context_list:
			assert len(context_list) == len(doc_list)
			without_context = False
		else:
			without_context = True
		self.ids, self.documents, self.contexts = id_list, doc_list, context_list
		self.fast_knn_search = len(self.documents) > self.fast_knn_search_threshold
		if self.fast_knn_search:
			self.logger.warning(f"Vectors DB size ({len(self.documents)}) is {len(self.documents)/self.fast_knn_search_threshold:.2f} times larger than fast_knn_search_threshold ({self.fast_knn_search_threshold}) -> using fast KNN search")
			if self.has_immutable_db and self._index:
				return self
		docs_to_embed = self.documents if without_context else zip(self.documents,self.contexts)
		documents_embeddings = self.sentence_embedding_fn(docs_to_embed, without_context=without_context, **args)
		assert documents_embeddings
		if self.fast_knn_search:
			# Initialize and build the hnswlib index
			self.logger.info(f'Initialising an hnswlib index with {len(documents_embeddings)} vectors')
			self._index = hnswlib.Index(space=self.space, dim=documents_embeddings[0].shape[-1])
			self._index.set_num_threads(self.n_threads) # multithreading in hnswlib seems to cause random crashes
			ef_construction = min(self._ef_construction,len(documents_embeddings))
			self._index.init_index(
				max_elements=len(documents_embeddings), 
				ef_construction=ef_construction, 
				M=self._max_number_of_connections,
				random_seed = self._random_seed,
			)
			self._index.add_items(documents_embeddings, range(len(documents_embeddings)))
			self._index.set_ef(self._ef_search) # ef should always be > top_k_hits # Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.
			self.logger.warning("Index for fast KNN search: built")
		else:
			self.documents_embeddings = documents_embeddings
		return self	

	def retrieve(self, query_list, similarity_threshold=None, without_context=False, top_k=None, **args):
		query_embeddings = self.sentence_embedding_fn(query_list, without_context=without_context, **args)
		if self.fast_knn_search:
			assert top_k, 'fast_knn_search requires a top_k'
			if top_k > self._ef_search:
				self.logger.warning(f'Changing EF to {top_k}!')
				self._index.set_ef(top_k) # ef should always be > top_k_hits # Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.
			# print(min(top_k,self._index.max_elements), len(self.documents))
			labels_list, distances_list = self._index.knn_query(query_embeddings, k=min(top_k,self._index.max_elements))
			return (
				(
					{
						'id':self.ids[idx], 
						'doc':self.documents[idx], 
						'index':idx, 
						'similarity':sim,
						'syntactic_similarity':0,
						'semantic_similarity':sim,
						'context': self.contexts[idx] if self.contexts else None
					}
					for idx,sim in zip(document_indices,map(lambda x: 1-x,distances)) # hnswlib is returning negated scores, for some reason
					if sim >= similarity_threshold
				)
				for document_indices,distances in zip(labels_list, distances_list)
			)
		self.logger.debug('SentenceRetriever::retrieve: computing similarities')
		similarity_matrix = self.similarity_fn(query_embeddings, self.documents_embeddings)
		return self.get_index_of_most_similar_documents(
			similarity_matrix, 
			similarity_threshold=similarity_threshold,
			top_k=min(top_k,len(self.documents_embeddings)) if top_k else top_k,
		)
	
	def get_index_of_most_similar_documents(self, similarity_vec, similarity_threshold=None, top_k=None):
		if similarity_threshold is None:
			similarity_threshold = self.default_similarity_threshold
		def get_similarity_dict_generator(sorted_similarities, sorted_indices):
			valid_indices = (sorted_similarities >= similarity_threshold).nonzero()
			similarities = map(float, sorted_similarities[valid_indices])
			document_indices = map(int, sorted_indices[valid_indices])
			return (
				{
					'id':self.ids[idx], 
					'doc':self.documents[idx], 
					'index':idx, 
					'similarity':sim,
					'syntactic_similarity':0,
					'semantic_similarity':sim,
					'context': self.contexts[idx] if self.contexts else None
				}
				for idx,sim in zip(document_indices,similarities)
			)
		self.logger.debug('SentenceRetriever::get_index_of_most_similar_documents: sorting indexes of similarities')
		sorted_indices = np.argsort(-similarity_vec, kind='stable', axis=-1)
		if top_k is not None:
			self.logger.debug('SentenceRetriever::get_index_of_most_similar_documents: slicing similarities')
			sorted_indices = sorted_indices[:, :top_k]
		self.logger.debug('SentenceRetriever::get_index_of_most_similar_documents: getting similarities')
		sorted_similarities = np.take_along_axis(similarity_vec, sorted_indices, axis=-1)
		self.logger.debug('SentenceRetriever::get_index_of_most_similar_documents: building similarity dictionaries')
		return (
			get_similarity_dict_generator(sim_vec,sim_idx)
			for sim_vec,sim_idx in zip(sorted_similarities,sorted_indices)
		)
