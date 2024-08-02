import sqlite3
import pickle
import shelve
import os
import logging
from knowpy.misc.cache_lib import load_or_create_cache, create_cache, load_cache
from knowpy.misc.utils import *

class PickleCacheManager:
	logger = logging.getLogger('knowpy')

	def __init__(self, cache_dir, **args):
		self.logger.info(f'Loading cache <{cache_dir}>')
		self.cache_dir = cache_dir
		self.cache = load_cache(cache_dir)
		if self.cache is None:
			self.cache = {}
		self.__uncached_steps = 0

	def import_from_cache_dict(self, cache_dict):
		pass

	@property
	def size(self):
		return sum(map(len,self.cache.values()))

	def import_from_cache_dict(self, *args, **argv):
		pass

	def store_cache(self):
		if not self.__uncached_steps:
			return
		self.logger.info(f'Storing cache <{self.cache_dir}>')
		create_cache(self.cache_dir, lambda: self.cache)

	def get_cached_values(self, value_list, cache_type, fetch_fn, key_fn=lambda x:x, empty_is_missing=False, with_cache=True, cache_every_n_steps=200000, **args):
		if not value_list:
			return []

		if not isinstance(value_list, (list, tuple)):
			value_list = tuple(value_list)

		# Collect keys and fetch cached values
		key_list = list(map(key_fn, value_list))
		cache = {
			k: self.cache[cache_type][k]
			for k in key_list
			if k in self.cache[cache_type]
		} if cache_type in self.cache else {}

		# Determine missing keys based on empty values and presence in the cache
		if empty_is_missing:
			missing_values = [v for k,v in zip(key_list,value_list) if not cache.get(k,None)]
		else:
			missing_values = [v for k,v in zip(key_list,value_list) if k not in cache]

		# Fetch and cache missing values
		if missing_values:
			for chunk in get_chunks(fetch_fn(missing_values), elements_per_chunk=cache_every_n_steps):
				missing_keys_chunk, missing_values_chunk = zip(*chunk)
				missing_keys_chunk = list(map(key_fn, missing_keys_chunk))
				missing_key_value_dict = {
					k:v 
					for k,v in zip(missing_keys_chunk, missing_values_chunk)
				}
				cache.update(missing_key_value_dict)
				if with_cache:
					if cache_type not in self.cache:
						self.cache[cache_type] = {}
					self.cache[cache_type].update(missing_key_value_dict)
					self.__uncached_steps += len(missing_key_value_dict)
					if self.__uncached_steps >= cache_every_n_steps:
						self.store_cache()
						self.__uncached_steps = 0 # do it after storing the cache

		# Return values in the order they were requested
		return [cache[k] for k in key_list]

class SQLiteCacheManager:
	logger = logging.getLogger('knowpy')

	def __init__(self, cache_dir, **args):
		cache_dir = cache_dir.replace('-','_')
		cache_dir = cache_dir.replace('.','_')
		if not cache_dir.endswith('/'):
			cache_dir += '/'
		# Ensure the directory exists
		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)  # Create the cache directory if it doesn't exist
		###################
		db_path = os.path.join(cache_dir,'cache.db')
		self.logger.info(f'Loading SQLite cache <{db_path}>')
		self.conn = sqlite3.connect(db_path)
		self.conn.row_factory = sqlite3.Row
		self.cur = self.conn.cursor()
		self.cur.execute(f"""
			CREATE TABLE IF NOT EXISTS cache (
				type TEXT,
				key TEXT,
				value BLOB,
				PRIMARY KEY (type, key)
			)
		""")
		# Enable Write-Ahead Logging
		self.cur.execute("PRAGMA journal_mode=WAL;")
		# Increase cache size
		self.cur.execute("PRAGMA size = 10000;")
		# Commit changes
		self.store_cache()
		self.__uncached_steps = 0

	@property
	def size(self):
		# Query to count the number of rows in the cache table
		self.cur.execute(f"SELECT COUNT(*) FROM cache")
		return int(self.cur.fetchone()[0])

	def import_from_cache_dict(self, cache_dict):
		# Insert the data into the SQLite database
		items_to_insert = [
			(cache_type, str(key), pickle.dumps(value))
			for cache_type, cache_values_dict in cache_dict.items()
			for key, value in cache_values_dict.items()
		]
		self.cur.executemany(f"INSERT OR REPLACE INTO cache (type, key, value) VALUES (?, ?, ?)", items_to_insert)
		# Commit changes
		self.store_cache()

	def fetch_cached_values(self, cache_type, key_list):
		MAX_VARIABLES = 900  # Use a safe number below 999
		results = []
		for i in range(0, len(key_list), MAX_VARIABLES):
			chunk_keys = key_list[i:i + MAX_VARIABLES]
			placeholders = ', '.join(['?'] * len(chunk_keys))
			query = f"SELECT key, value FROM cache WHERE type = ? AND key IN ({placeholders})"
			self.cur.execute(query, [cache_type] + chunk_keys)
			results += self.cur.fetchall()
		return results

	def store_cache(self):
		self.conn.commit()

	def get_cached_values(self, value_list, cache_type, fetch_fn, key_fn=lambda x:x, empty_is_missing=False, with_cache=True, cache_every_n_steps=200000, **args):
		if not value_list:
			return []

		# Ensure value list is listable
		if not isinstance(value_list, (list, tuple)):
			value_list = tuple(value_list)

		# Collect keys
		key_list = list(map(str, map(key_fn, value_list)))
		
		# Fetch cached values
		cache = {
			row['key']: pickle.loads(row['value']) 
			for row in self.fetch_cached_values(cache_type, key_list)
		}
		
		# Determine missing keys based on empty values and presence in the cache
		if empty_is_missing:
			missing_values = [v for k,v in zip(key_list,value_list) if not cache.get(k,None)]
		else:
			missing_values = [v for k,v in zip(key_list,value_list) if k not in cache]

		# Fetch and cache missing values
		if missing_values:
			for chunk in get_chunks(fetch_fn(missing_values), elements_per_chunk=cache_every_n_steps):
				missing_keys_chunk, missing_values_chunk = zip(*chunk)
				missing_keys_chunk = list(map(str, map(key_fn, missing_keys_chunk)))
				if with_cache:
					items_to_insert = [
						(cache_type, key, pickle.dumps(value)) 
						for key, value in zip(missing_keys_chunk, missing_values_chunk)
					]
					self.cur.executemany(f"INSERT OR REPLACE INTO cache (type, key, value) VALUES (?, ?, ?)", items_to_insert)
					self.__uncached_steps += len(items_to_insert)
					if self.__uncached_steps >= cache_every_n_steps:
						self.__uncached_steps = 0
						self.store_cache()
				cache.update({k:v for k,v in zip(missing_keys_chunk, missing_values_chunk)})

		# Return values in the order they were requested
		return [cache[k] for k in key_list]

	def __del__(self):
		""" Destructor to ensure the database connection and cursor are properly closed. """
		try:
			self.cur.close()
			self.conn.close()
		except Exception as e:
			print(f"Error closing database connection: {e}")

class ShelveCacheManager:
	logger = logging.getLogger('knowpy')

	def __init__(self, cache_dir, num_shards=250, **args):
		cache_dir = cache_dir.replace('-','_')
		cache_dir = cache_dir.replace('.','_')
		if not cache_dir.endswith('/'):
			cache_dir += '/'
		# Ensure the directory exists
		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)  # Create the cache directory if it doesn't exist
		#######################
		self.logger.info(f'Loading Shelve shards <{cache_dir}>')
		self.num_shards = num_shards
		self.shelve_paths = [os.path.join(cache_dir, f'cache_shard_{shard_id}') for shard_id in range(num_shards)]
		self.stores = [shelve.open(path, writeback=True) for path in self.shelve_paths]
		self.__uncached_steps = 0

	@property
	def size(self):
		return sum(map(lambda x: sum(map(len, x.values())),self.stores))

	def _get_shard(self, key):
		""" Determine the shard by key hash. """
		return self.stores[hash(key) % self.num_shards]

	def import_from_cache_dict(self, cache_dict):
		# Insert the data into the shelve store
		for cache_type, cache_values_dict in cache_dict.items():
			for key, value in cache_values_dict.items():
				store = self._get_shard(key)
				store.setdefault(cache_type, {})[str(key)] = value
			self.store_cache()

	def store_cache(self, store_list=None):
		if store_list is None:
			store_list = self.stores
		for store in store_list:
			store.sync()

	def get_cached_values(self, value_list, cache_type, fetch_fn, key_fn=lambda x:x, empty_is_missing=False, with_cache=True, cache_every_n_steps=200000, **args):
		if not value_list:
			return []

		# Ensure value list is listable
		if not isinstance(value_list, (list, tuple)):
			value_list = tuple(value_list)
		
		# Collect keys
		key_list = list(map(str, map(key_fn, value_list)))
		
		# Fetch cached values
		cache = {}
		for key in key_list:
			store = self._get_shard(key)
			if cache_type in store and key in store[cache_type]:
				cache[key] = store[cache_type][key]
		
		# Determine missing keys based on empty values and presence in the cache
		if empty_is_missing:
			missing_values = [v for k,v in zip(key_list,value_list) if not cache.get(k,None)]
		else:
			missing_values = [v for k,v in zip(key_list,value_list) if k not in cache]

		# Fetch and cache missing values
		if missing_values:
			for chunk in get_chunks(fetch_fn(missing_values), elements_per_chunk=cache_every_n_steps):
				missing_keys_chunk, missing_values_chunk = zip(*chunk)
				missing_keys_chunk = list(map(str, map(key_fn, missing_keys_chunk)))
				if with_cache:
					for key, value in zip(missing_keys_chunk, missing_values_chunk):
						store = self._get_shard(key)
						store.setdefault(cache_type, {})[str(key)] = value
					self.__uncached_steps += len(missing_values_chunk)
					if self.__uncached_steps >= cache_every_n_steps:
						self.__uncached_steps = 0
						self.store_cache()
				cache.update(dict(zip(missing_keys_chunk, missing_values_chunk)))
		# Return values in the order they were requested
		return [cache[k] for k in key_list]

	def __del__(self):
		""" Destructor to ensure the shelve store is properly closed. """
		for store in self.stores:
			try:
				store.close()
			except Exception as e:
				print(f"Error closing shelve store: {e}")

# # Usage example
# def fetch_fn(keys):
#     # Example function that fetches data for given keys
#     return ["Data for " + str(key) for key in keys]

# cache_manager = CacheManager()
# value_list = [1, 2, 3, 4, 5]
# results = cache_manager.get_cached_values(value_list, fetch_fn)
# print(results)
# cache_manager.close()
