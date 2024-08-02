import os
import random
import re
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import multiprocessing
from pathos.multiprocessing import ProcessPool as Pool
import types
import spacy # for natural language processing
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import json

from more_itertools import unique_everseen

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import ollama

from knowpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache
from knowpy.misc.utils import *

import logging
import concurrent.futures
import time
import copy
from collections import defaultdict

from cache_manager import PickleCacheManager, SQLiteCacheManager, ShelveCacheManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# warnings.filterwarnings('ignore')
# logging.getLogger("spacy").setLevel(logging.ERROR)

# set_seed(42)
is_listable = lambda x: type(x) in (list,tuple)

class ModelManager():
	# static members
	__nlp_models = {}
	__sbert_models = {}
	__openai_embedding_models = {}

	logger = logging.getLogger('knowpy')
	# ollama_client = ollama.Client(host='http://localhost:11434')

	def __init__(self, model_options=None):
		if not model_options:
			model_options = {}
		self.model_options = model_options
		self.disable_spacy_component = []
		self.with_tqdm = model_options.get('with_tqdm', False)

		self.__default_batch_size = model_options.get('default_batch_size', 1000)
		self.__with_cache = model_options.get('with_cache', True)
		self.default_cache_path = model_options.get('default_cache_path', model_options.get('default_cache_dir', None))

		# Initialize CacheManager
		self._cache_manager = None
		self._cache_manager_class = model_options.get('cache_manager_type', None)
		if self._cache_manager_class is None:
			self.logger.info(f'Using PickleCacheManager as the cache manager.')
			self._cache_manager_class = PickleCacheManager
		else:
			self.logger.info(f'Using {self._cache_manager_class} as the cache manager.')
			self._cache_manager_class = eval(self._cache_manager_class)

		self.__spacy_model = model_options.get('spacy_model', 'en_core_web_md')
		if isinstance(self.__spacy_model, list):
			self.__spacy_model = tuple(self.__spacy_model)
		self.__n_threads = model_options.get('n_threads', -1)
		if self.__n_threads < 0:
			self.__n_threads = max(1,multiprocessing.cpu_count()-1)
			
		self.__use_gpu = self.model_options.get('use_gpu', False)
		self.__sbert_model_options = model_options.get('sbert_model', {})
		self.__ollama_model_options = model_options.get('ollama_model', {})
		self.__openai_embedding_model_options = model_options.get('openai_embedding_model', {})

		self.__chatgpt_client = None

	@property
	def n_threads(self):
		return self.__n_threads

	@property
	def default_cache_dir(self):
		return self.default_cache_path

	@property
	def default_batch_size(self):
		return self.__default_batch_size

	@property
	def cache_manager(self):
		if not self._cache_manager:
			assert self.default_cache_path, 'A default_cache_path must be set in the model options during init'
			self._cache_manager = self._cache_manager_class(self.default_cache_path, num_shards=self.model_options.get('num_shards', 250))
		return self._cache_manager

	def store_cache(self, cache_name=None):
		# Trigger the store_cache method of CacheManager
		self.cache_manager.store_cache()

	def load_cache(self, cache_name=None):
		if not cache_name:
			cache_name = self.default_cache_path
		assert cache_name, 'A cache must be set either as a parameter of this function or in the model options during init'
		self._cache_manager = self._cache_manager_class(cache_name, num_shards=self.model_options.get('num_shards', 250))
		# self.cache_manager.import_from_cache_dict(load_cache(cache_name))
		return True

	def copy_cache(self, model_manager):
		self._cache_manager = model_manager._cache_manager

	# def reset_cache(self):
	# 	# Reset the cache in CacheManager
	# 	self.cache_manager.reset_cache()

	def get_cached_values(self, value_list, cache_type, fetch_fn, **args):
		return self.cache_manager.get_cached_values(value_list, cache_type, fetch_fn, **args)

	def tqdm(self, it, **args):
		if isinstance(it, (list,tuple)) and len(it) <= 1:
			return it
		if args.get('total', float('inf')) <= 1:
			return it
		if self.with_tqdm:
			return tqdm(it, **args)
		return it

	@staticmethod
	def load_nlp_model(spacy_model, use_gpu):
		ModelManager.logger.info('Loading Spacy model <{}>'.format(spacy_model))
		# go here <https://spacy.io/usage/processing-pipelines> for more information about Language Processing Pipeline (tokenizer, tagger, parser, etc..)
		if use_gpu:
			activated = spacy.prefer_gpu()
		else:
			spacy.require_cpu()
			ModelManager.logger.info('Running spacy on CPU')
		try:
			nlp = spacy.load(spacy_model)
		except OSError:
			ModelManager.logger.warning('Downloading language model for the spaCy POS tagger\n'
				"(don't worry, this will only happen once)")
			spacy.cli.download(spacy_model)
			nlp = spacy.load(spacy_model)
		# nlp.add_pipe("doc_cleaner")
		# nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
		# nlp.add_pipe(nlp.create_pipe("merge_entities"))
		# nlp.add_pipe(nlp.create_pipe("merge_subtokens"))
		#################################
		# nlp.add_pipe(neuralcoref.NeuralCoref(nlp.vocab), name='neuralcoref', last=True) # load NeuralCoref and add it to the pipe of SpaCy's model
		# def remove_unserializable_results(doc): # Workaround for serialising NeuralCoref's clusters
		#   def cluster_as_doc(c):
		#       c.main = c.main.as_doc()
		#       c.mentions = [
		#           m.as_doc()
		#           for m in c.mentions
		#       ]
		#   # doc.user_data = {}
		#   if not getattr(doc,'_',None):
		#       return doc
		#   if not getattr(doc._,'coref_clusters',None):
		#       return doc
		#   for cluster in doc._.coref_clusters:
		#       cluster_as_doc(cluster)
		#   for token in doc:
		#       for cluster in token._.coref_clusters:
		#           cluster_as_doc(cluster)
		#   return doc
		# nlp.add_pipe(remove_unserializable_results, last=True)
		ModelManager.logger.info('Spacy model loaded')
		return nlp

	@staticmethod
	def load_sbert_model(sbert_model_options):
		model_url = sbert_model_options['url']
		use_gpu = sbert_model_options.get('use_gpu',False)
		cache_dir = sbert_model_options.get('cache_dir',None)
		is_qa_model = ModelManager.sbert_is_qa_model(sbert_model_options)
		if is_qa_model:
			ModelManager.logger.info(f"Loading sentence_transformers model <{model_url}> for QA; GPU support {use_gpu}")
		else:
			ModelManager.logger.info(f"Loading sentence_transformers model <{model_url}>; GPU support {use_gpu}")
		sbert_model = SentenceTransformer(
			model_url, 
			# device='cpu' if not use_gpu or not gpu_devices else None, 
			device='cpu' if not use_gpu else None, 
			cache_folder=cache_dir if isinstance(cache_dir,str) and os.path.isdir(cache_dir) else None
		)

		get_input = lambda y: tuple(map(lambda x: x[0] if is_listable(x) else x, y))
		get_context = lambda y: tuple(map(lambda x: x[1] if is_listable(x) else '', y))
		if model_url == 'nq-distilbert-base-v1':
			a_template = lambda x: (x[0],x[1])
			q_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
			a_module = lambda doc,**args: sbert_model.encode(tuple(map(a_template, zip(get_input(doc),get_context(doc)))),**args)
		elif is_qa_model: # elif model_url.startswith('facebook-dpr'):
			if model_url == 'clips/mfaq':
				q_template = lambda x: '<Q>'+x
				a_template = lambda x: f"<A>{x[0]}" + (f" [SEP] {x[1]}" if x[1] else '')
			elif 'multilingual-e5' in model_url:
				q_template = lambda x: 'query: '+x
				a_template = lambda x: f"passage: {x[0]}" + (f" [SEP] {x[1]}" if x[1] else '')
			else:
				q_template = lambda x: x
				a_template = lambda x: x[0] + (f" [SEP] {x[1]}" if x[1] else '')
			ctx_sbert_model = SentenceTransformer(
				model_url.replace('question_encoder','ctx_encoder'), 
				# device='cpu' if not use_gpu or not gpu_devices else None, 
				device='cpu' if not use_gpu else None, 
				cache_folder=cache_dir if isinstance(cache_dir,str) and os.path.isdir(cache_dir) else None
			) if 'question_encoder' in model_url else sbert_model
			q_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
			a_module = lambda doc,**args: ctx_sbert_model.encode(tuple(map(a_template, zip(get_input(doc),get_context(doc)))),**args)
		else:
			q_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
			a_module = lambda doc,**args: sbert_model.encode(get_input(doc),**args)
		ModelManager.logger.info('SBERT model loaded')
		return {
			'question': q_module,
			'answer': a_module
		}

	def language_is_available(self, lang):
		if not is_listable(self.__spacy_model):
			return self.__spacy_model.startswith(lang)
		return any(map(lambda x: x.startswith(lang), self.__spacy_model))

	def get_nlp_model(self,lang=None):
		if not is_listable(self.__spacy_model):
			if lang:
				model = self.__spacy_model if self.__spacy_model.startswith(lang) else None
			else:
				model = self.__spacy_model
		else:
			if lang:
				model = next(filter(lambda x: x.startswith(lang), self.__spacy_model), None)
			else:
				model = self.__spacy_model[0]
		if not model:
			return None
		if ModelManager.__nlp_models.get(model, None) is None:
			ModelManager.__nlp_models[model] = ModelManager.load_nlp_model(model, self.__use_gpu)
		self.logger.info(f'Spacy model {model} loaded for language {lang}')
		return ModelManager.__nlp_models[model]

	@staticmethod
	def sbert_is_qa_model(model_dict):
		model_dict_url = model_dict['url'].lower()
		return 'question_encoder' in model_dict_url or 'qa' in model_dict_url or 'faq' in model_dict_url

	def get_sbert_model(self):
		model_key = self.__sbert_model_options['url']
		if ModelManager.__sbert_models.get(model_key, None) is None:
			ModelManager.__sbert_models[model_key] = ModelManager.load_sbert_model(self.__sbert_model_options)
		return ModelManager.__sbert_models[model_key]

	def get_openai_embedding_model(self):
		model_key = self.__openai_embedding_model_options['model']
		if ModelManager.__openai_embedding_models.get(model_key, None) is None:
			ModelManager.__openai_embedding_models[model_key] = OpenAI(api_key=self.__openai_embedding_model_options['api_key'])
		return ModelManager.__openai_embedding_models[model_key]

	def detect_language_parallel(self, text_list):
		def detect_language_wrapper(text):
			try:
				language = detect_language(text)
				return (text, language, None)
			except Exception as e:
				# Replace the logger warning with a return value that indicates failure
				return (text, None, str(e))
		# Initialize a pool
		self.logger.info(f'ModelManager::detect_language_parallel: starting Pool')
		with Pool(nodes=max(1,self.__n_threads)) as pool:
			for text, language, error_message in self.tqdm(pool.uimap(detect_language_wrapper, text_list), total=len(text_list), desc="Identifying texts' language"):
				if error_message:
					self.logger.warning(f'Cannot find language of: {text}')
				yield text, language

	def nlp(self, text_list, language_list=None, disable=None, n_threads=None, batch_size=None, **args):
		if not disable:
			disable = self.disable_spacy_component
		if not n_threads: # real multi-processing: https://git.readerbench.com/eit/prepdoc/blob/f8e93b6d0a346e9a53dac2e70e5f1712d40d6e1e/examples/parallel_parse.py
			n_threads = self.__n_threads
		if not batch_size:
			batch_size = self.__default_batch_size
		if language_list:
			if not is_listable(text_list):
				text_list = tuple(text_list)
			if not is_listable(language_list):
				language_list = tuple(language_list)
			text_language_dict = dict(zip(text_list,language_list))
		def fetch_fn(missing_texts):
			self.logger.debug(f"Processing {len(missing_texts)} texts with spacy")
			if is_listable(self.__spacy_model):
				language_missing_texts_dict = defaultdict(list)
				# Initialize a pool
				if language_list:
					for text in missing_texts:
						language_missing_texts_dict[text_language_dict[text]].append(text)
				else:
					for text, language in self.detect_language_parallel(missing_texts):
						language_missing_texts_dict[language].append(text)
			else:
				language_missing_texts_dict = {
					self.__spacy_model.split('_')[0]: missing_texts
				}
			for language, missing_items in language_missing_texts_dict.items():
				nlp = self.get_nlp_model(language) if language else None
				if not nlp:
					nlp = self.get_nlp_model()
				output_iter = self.tqdm(
					nlp.pipe(
						missing_items, 
						disable=disable, 
						batch_size=batch_size,
						# n_process=n_threads, # bugged
					),
					total=len(missing_items), 
					desc=f"Processing {language} inputs with Spacy"
				)
				for i,o in zip(missing_items, output_iter):
					yield i,o
				# self.logger.info(f'ModelManager::spacy_nlp: starting ThreadPoolExecutor')
				# def _fetch_fn(x):
				# 	return (x, nlp.pipe(x, 
				# 		disable=disable, 
				# 		batch_size=len(x),
				# 		# n_process=n_threads, # bugged
				# 	))
				# with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_threads)) as executor:
				# 	futures = [
				# 		executor.submit(_fetch_fn, chunk) 
				# 		for chunk in get_chunks(missing_items, elements_per_chunk=batch_size)
				# 	]
				# 	for future in self.tqdm(concurrent.futures.as_completed(futures), total=math.ceil(len(missing_items)/batch_size), desc=f"Processing {language} inputs with Spacy"):
				# 		for i,o in zip(*future.result()):
				# 			yield i,o
		return self.get_cached_values(text_list, 'spacy_cache', fetch_fn, key_fn=lambda x:(x,self.__spacy_model), **args)

	def run_sbert_embedding(self, inputs, cache_every_n_steps=200000, without_context=False, **args):
		sbert_model = self.get_sbert_model()
		batch_size = self.__sbert_model_options.get('batch_size', self.__default_batch_size)
		def fetch_fn(missing_inputs):
			self.logger.debug(f"Processing {len(missing_inputs)} inputs with sbert")
			encoder = sbert_model['question' if without_context else 'answer']
			len_batch_iter = math.ceil(len(missing_inputs)/cache_every_n_steps)
			ModelManager.logger.info(f'SBERT: Modelling {len(missing_inputs)} sentences in {len_batch_iter} batches. Max len: {len(max(missing_inputs,key=len))}, Mean len: {sum(map(len,missing_inputs))/len(missing_inputs)}, Min len: {len(min(missing_inputs,key=len))}')
			for e in self.tqdm(range(len_batch_iter), total=len_batch_iter, desc="Processing inputs with SBERT"):
				batch_queries = missing_inputs[e*cache_every_n_steps:(e+1)*cache_every_n_steps]
				encoded_queries = encoder(
					batch_queries, 
					convert_to_tensor=True, # set to true for a more efficient numpy conversion
					convert_to_numpy=False, # set to false to save RAM memory at runtime
					batch_size=batch_size,
					# precision=self.__sbert_model_options.get('precision', 'float32'),
					show_progress_bar=self.with_tqdm and len(batch_queries) > batch_size,
				).numpy(force=True) # If force is True this is equivalent to calling t.detach().cpu().resolve_conj().resolve_neg().numpy(). If the tensor isn’t on the CPU or the conjugate or negative bit is set, the tensor won’t share its storage with the returned ndarray. Setting force to True can be a useful shorthand.
				for i,o in zip(batch_queries, encoded_queries):
					yield i,o
				del encoded_queries # free memory
			ModelManager.logger.info('SBERT: Finished modelling')
		return self.get_cached_values(
			inputs, 
			'sbert_cache', 
			fetch_fn, 
			key_fn=lambda x:(x,without_context,self.__sbert_model_options['url']), 
			cache_every_n_steps=cache_every_n_steps,
			**args
		)

	def run_openai_embedding(self, inputs, cache_every_n_steps=200000, **args):
		chatgpt_client = self.get_openai_embedding_model()
		def fetch_fn(i):
			o = chatgpt_client.embeddings.create(
				input=i, 
				model=self.__openai_embedding_model_options['model']
			).data[0].embedding
			return i,o
		def parallel_fetch_fn(missing_inputs):
			if self.n_threads <= 1 or len(missing_inputs) <= 1:
				for i,o in self.tqdm(map(fetch_fn, missing_inputs), total=len(missing_inputs), desc="Sending texts to embed to OpenAI"):
					yield i,o
				return
			# Using ThreadPoolExecutor to run queries in parallel with tqdm for progress tracking
			self.logger.info(f'ModelManager::instruct_gpt_model: starting ThreadPoolExecutor')
			with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,self.n_threads)) as executor:
				futures = [executor.submit(fetch_fn, prompt) for prompt in missing_inputs]
				for e,future in enumerate(self.tqdm(concurrent.futures.as_completed(futures), total=len(missing_inputs), desc="Sending texts to embed to OpenAI")):
					i,o=future.result()
					yield i,o
		return self.get_cached_values(
			inputs, 
			'openai_embedding', 
			parallel_fetch_fn, 
			key_fn=lambda x:(x,self.__openai_embedding_model_options['model']), 
			cache_every_n_steps=cache_every_n_steps,
			**args
		)

	def instruct_model(self, prompts, model='mistral:instruct', cache_every_n_steps=2000, **kwargs):
		if model.startswith('gpt'):
			return self.instruct_gpt_model(prompts, model=model, cache_every_n_steps=cache_every_n_steps, **kwargs)
		return self.instruct_ollama_model(prompts, model=model, cache_every_n_steps=cache_every_n_steps, **kwargs)
			
	def instruct_ollama_model(self, prompts, model='mistral:instruct', options=None, empty_is_missing=True, output_to_input_proportion=2, non_influential_prompt_size=0, **args):
		if options is None:
			# For Mistral: https://www.reddit.com/r/LocalLLaMA/comments/16v820a/mistral_7b_temperature_settings/
			options = { # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
				"seed": 42, # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
				"num_predict": -1, # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
				"top_k": 40, # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
				"top_p": 0.95, # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
				"temperature": 0.7, # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
				"repeat_penalty": 1., # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
				"tfs_z": 1, # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
				"num_ctx": 2**13,  # Sets the size of the context window used to generate the next token. (Default: 2048)
				"repeat_last_n": 64, # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
				# "num_gpu": 0, # The number of layers to send to the GPU(s). Set to 0 to disable.
			}
		if self.__n_threads > 1 and "num_thread" not in options:
			options["num_thread"] = 1 # Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores).
		def fetch_fn(missing_prompt):
			_options = copy.deepcopy(options) # required to avoid side-effects
			if _options.get("num_predict",-2) == -2:
				prompt_tokens = 2*(len(missing_prompt.split(' '))-non_influential_prompt_size)
				_options["num_predict"] = int(output_to_input_proportion*prompt_tokens)
			# response = self.ollama_client.generate(
			response = ollama.generate(
				model=model,
				prompt=missing_prompt,
				stream=False,
				options=_options,
				keep_alive=-1,
			)
			# print(missing_prompt, response['response'])
			# return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
			return missing_prompt, response['response']
		def parallel_fetch_fn(missing_prompt_list):
			if self.n_threads <= 1 or len(missing_prompt_list) <= 1:
				for i,o in self.tqdm(map(fetch_fn, missing_prompt_list), total=len(missing_prompt_list), desc="Sending prompts to OpenAI"):
					yield i,o
				return
			# Using ThreadPoolExecutor to run queries in parallel with tqdm for progress tracking
			self.logger.info(f'ModelManager::instruct_ollama_model: starting ThreadPoolExecutor')
			with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,self.n_threads)) as executor:
				futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
				for future in self.tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to Ollama"):
					i,o=future.result()
					yield i,o
		cache_key = json.dumps(options,indent=4)
		return self.get_cached_values(
			prompts, 
			'ollama_cache', 
			parallel_fetch_fn, 
			key_fn=lambda x: (x,model,cache_key), 
			# key_fn=lambda x: (x,model), 
			empty_is_missing=empty_is_missing,
			# force_cache=True,
			**args
		)

	def instruct_gpt_model(self, prompts, model='gpt-4', n=1, temperature=1, top_p=1, frequency_penalty=0, presence_penalty=0, api_key=None, organization=None, empty_is_missing=True, **args):
		if not self.__chatgpt_client:
			self.__chatgpt_client = OpenAI(api_key=api_key, organization=organization)
		max_tokens = None
		adjust_max_tokens = True
		if '32k' in model:
			max_tokens = 32768
		elif '16k' in model:
			max_tokens = 16385
		elif model=='gpt-4o' or 'preview' in model or 'turbo' in model:
			max_tokens = 4096 #128000
			adjust_max_tokens = False
		if not max_tokens:
			if model.startswith('gpt-4'):
				max_tokens = 8192
			else:
				max_tokens = 4096
				adjust_max_tokens = False
		def fetch_fn(missing_prompt):
			messages = [ {"role": "user", "content": missing_prompt} ]
			prompt_max_tokens = max_tokens
			if adjust_max_tokens:
				prompt_max_tokens -= int(3*len(missing_prompt.split(' \n')))
			if prompt_max_tokens < 1:
				return missing_prompt, None
			try:
				response = self.__chatgpt_client.chat.completions.create(model=model,
					messages=messages,
					max_tokens=prompt_max_tokens,
					n=n,
					stop=None,
					temperature=temperature,
					top_p=top_p,
					frequency_penalty=frequency_penalty, 
					presence_penalty=presence_penalty
				)
				result = [
					r.message.content.strip() 
					for r in response.choices 
					if r.message.content != 'Hello! It seems like your message might have been cut off. How can I assist you today?'
				]
				if len(result) == 1:
					result = result[0]
				return missing_prompt, result # return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
			except Exception as e:
				self.logger.warning(f'OpenAI returned this error: {e}')
				return missing_prompt, None
		def parallel_fetch_fn(missing_prompt_list):
			if self.n_threads <= 1 or len(missing_prompt_list) <= 1:
				for i,o in self.tqdm(map(fetch_fn, missing_prompt_list), total=len(missing_prompt_list), desc="Sending prompts to OpenAI"):
					yield i,o
				return
			# Using ThreadPoolExecutor to run queries in parallel with tqdm for progress tracking
			self.logger.info(f'ModelManager::instruct_gpt_model: starting ThreadPoolExecutor')
			with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,self.n_threads)) as executor:
				futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
				for e,future in enumerate(self.tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to OpenAI")):
					i,o=future.result()
					yield i,o
		return self.get_cached_values(
			prompts, 
			'gpt_cache', 
			parallel_fetch_fn, 
			key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty) if temperature!=0.5 or top_p!=0 or frequency_penalty!=0 or presence_penalty!=0 else (x,model,n), 
			# key_fn=lambda x: (x,model,n), 
			empty_is_missing=empty_is_missing,
			# force_cache=True,
			**args
		)

	def get_default_embedder(self):
		if self.__sbert_model_options:
			return self.run_sbert_embedding
		elif self.__openai_embedding_model_options:
			return self.run_openai_embedding
		return None

	def get_default_similarity_fn_type(self):
		if self.__sbert_model_options:
			return self.__sbert_model_options.get('similarity_fn', 'cosine_similarity')
		elif self.__openai_embedding_model_options:
			return self.__openai_embedding_model_options.get('similarity_fn', 'cosine_similarity')
		return None

	def get_default_similarity_fn(self):
		# sim_fn = cosine_similarity if self.get_default_similarity_fn_type() == 'cosine_similarity' else np.inner
		# def similarity_batch_fn(vec1_set, vec2_set, batch_size=100000): # numpy's multiprocessing implementation doesn't handle well a very large vec2_set (e.g., 32kk items), it would crash without batching
		# 	vec1_set = np.asarray(vec1_set)
		# 	vec2_set = np.asarray(vec2_set)
		# 	similarities = np.zeros((vec1_set.shape[0], vec2_set.shape[0]))
		# 	for start_row in range(0, vec2_set.shape[0], batch_size):
		# 		end_row = start_row + batch_size
		# 		vec2_batch = vec2_set[start_row:end_row]
		# 		# Compute dot product between vec1_set and the current vec2_batch
		# 		similarities[:, start_row:end_row] = sim_fn(vec1_set, vec2_batch)
		# 	return similarities
		# return similarity_batch_fn
		return cosine_similarity if self.get_default_similarity_fn_type() == 'cosine_similarity' else np.inner

	def get_default_embedding(self, text_list, without_context=False, with_cache=None):
		embedding_fn = self.get_default_embedder()
		assert embedding_fn is not None, 'cannot find a proper embedding_fn, please specify a TF or SBERT model'
		return embedding_fn(text_list, without_context=without_context, with_cache=with_cache)

	def get_default_similarity(self, a, b):
		similarity_fn = self.get_default_similarity_fn()
		assert similarity_fn is not None, 'cannot find a proper similarity_fn'
		return similarity_fn(a,b)

	def get_element_wise_similarity(self, source_list, target_list, source_without_context=False, target_without_context=False, get_embedding_fn=None, get_similarity_fn=None, with_cache=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		assert len(source_list) == len(target_list), 'len(source_list) != len(target_list)'
		source_embeddings = get_embedding_fn(source_list, without_context=source_without_context, with_cache=with_cache)
		target_embeddings = get_embedding_fn(target_list, without_context=target_without_context, with_cache=with_cache)
		return [
			float(get_similarity_fn([a],[b])[0][0]) if a is not None and b is not None else 0
			for a,b in zip(source_embeddings,target_embeddings)
		]

	def get_similarity_ranking(self, source_text_list, target_text_list, without_context=False, get_embedding_fn=None, get_similarity_fn=None, with_cache=None):
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity
		source_embeddings = get_embedding_fn(source_text_list, without_context=without_context, with_cache=with_cache)
		target_embeddings = get_embedding_fn(target_text_list, without_context=False, with_cache=with_cache)
		similarity_vec = get_similarity_fn(source_embeddings,target_embeddings)
		return np.argsort(similarity_vec, kind='stable', axis=-1), similarity_vec

	def remove_similar_labels(self, tuple_list, threshold=0.97, key=None, without_context=False, get_embedding_fn=None, get_similarity_fn=None, sort_by_conformity=False, with_cache=None):
		# Set the default key function if none is provided.
		# The key function is used to extract the element (label) from each tuple/list to be processed.
		if key is None:
			key = lambda x: x[0] if isinstance(x, (list, tuple)) else x

		# Set the default embedding function if none is provided.
		if get_embedding_fn is None:
			get_embedding_fn = self.get_default_embedding

		# Set the default similarity function if none is provided.
		if get_similarity_fn is None:
			get_similarity_fn = self.get_default_similarity

		# Extract the values (labels) from the input list using the key function.
		value_list = tuple(map(key, tuple_list))

		# Compute the similarity matrix between all pairs of labels.
		# This involves generating embeddings for each label and then calculating the similarity between every pair of embeddings.
		embeddings = get_embedding_fn(value_list, without_context=without_context, with_cache=with_cache)
		similarity_vec = get_similarity_fn(embeddings, embeddings)
		
		# If sorting by conformity is enabled, sort the labels based on their conformity (similarity to other labels).
		if sort_by_conformity:
			# Calculate the sum of similarities for each label and sort the indices based on these sums in descending order.
			sorted_idx_vec = np.argsort(np.sum(similarity_vec, axis=-1), kind='stable', axis=-1)[::-1]
			# Reorder the similarity matrix according to the sorted indices.
			sorted_similarity_vec = np.take(similarity_vec, sorted_idx_vec, axis=0)
			# Filter out labels that are too similar to any preceding label in the sorted list.
			return [
				tuple_list[j]
				for i, j in enumerate(sorted_idx_vec.tolist())
				if not np.any(sorted_similarity_vec[i][:i] >= threshold)
			]

		# If not sorting by conformity, iterate through each label.
		result_list = []
		for i, v in enumerate(tuple_list):
			# Check if the current label is not too similar to any previous label.
			# This is done by ensuring no similarity value before it reaches the threshold.
			if not np.any(similarity_vec[i][:i] >= threshold):
				result_list.append(v)
			else: # For labels too similar to previous ones, ignore them in future comparisons by setting their similarity to 0.
				similarity_vec[:, i] = 0
		return result_list

