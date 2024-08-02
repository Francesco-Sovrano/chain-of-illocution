import itertools
import hashlib
import numpy as np
import math
import unicodedata
import unidecode
import re
from langdetect import detect as detect_language
from langdetect import DetectorFactory
DetectorFactory.seed = 0

import inspect
import functools

is_number = lambda x: bool(re.match(r'^-?\d+(\.\d+)?(e[-+]?\d+)?$', x))

language_map = {
	'af': 'Afrikaans',
	'ar': 'Arabic',
	'bg': 'Bulgarian',
	'bn': 'Bengali',
	'ca': 'Catalan',
	'cs': 'Czech',
	'cy': 'Welsh',
	'da': 'Danish',
	'de': 'German',
	'el': 'Greek',
	'en': 'English',
	'es': 'Spanish',
	'et': 'Estonian',
	'fa': 'Persian',
	'fi': 'Finnish',
	'fr': 'French',
	'gu': 'Gujarati',
	'he': 'Hebrew',
	'hi': 'Hindi',
	'hr': 'Croatian',
	'hu': 'Hungarian',
	'id': 'Indonesian',
	'it': 'Italian',
	'ja': 'Japanese',
	'kn': 'Kannada',
	'ko': 'Korean',
	'lt': 'Lithuanian',
	'lv': 'Latvian',
	'mk': 'Macedonian',
	'ml': 'Malayalam',
	'mr': 'Marathi',
	'ne': 'Nepali',
	'nl': 'Dutch',
	'no': 'Norwegian',
	'pa': 'Punjabi',
	'pl': 'Polish',
	'pt': 'Portuguese',
	'ro': 'Romanian',
	'ru': 'Russian',
	'sk': 'Slovak',
	'sl': 'Slovenian',
	'so': 'Somali',
	'sq': 'Albanian',
	'sv': 'Swedish',
	'sw': 'Swahili',
	'ta': 'Tamil',
	'te': 'Telugu',
	'th': 'Thai',
	'tl': 'Tagalog',
	'tr': 'Turkish',
	'uk': 'Ukrainian',
	'ur': 'Urdu',
	'vi': 'Vietnamese',
	'zh-cn': 'Chinese (Simplified)',
	'zh-tw': 'Chinese (Traditional)',
}
def language_code_to_name(code):
	return language_map.get(code, '')

def get_language_name_from_sentence(sentence):
	return language_code_to_name(detect_language(sentence))

def format_content(content):
	content = unicodedata.normalize("NFKC", content) # normalize content
	content = content.encode('utf-8').decode('utf-8', 'ignore').strip()
	content = unidecode.unidecode(content)
	return content

def get_str_uid(x):
	hex_uid = hashlib.md5(x.encode()).hexdigest()
	return np.base_repr(int(hex_uid,16), 36)

def get_iter_uid(x):
	return get_str_uid(' '.join(x))

def flatten(it, as_list=False):
	flattened_it = (
		y
		for x in it
		for y in x
	)
	# flattened_it = itertools.chain.from_iterable(it)
	if as_list:
		flattened_it = list(flattened_it)
	return flattened_it

def get_chunks(it, elements_per_chunk=None, number_of_chunks=None):
	"""Divide a list of nodes `it` in `n` chunks"""
	# assert elements_per_chunk or number_of_chunks
	if not elements_per_chunk and (not number_of_chunks or number_of_chunks==1):
		return (it,)
	if number_of_chunks:
		if not isinstance(it, (list,tuple)):
			it = tuple(it)
		elements_per_chunk = max(1, len(it) // number_of_chunks)  # Calculate chunk size, ensure it's at least 1
	# The current chunk being collected
	current_chunk = []
	for element in it:
		current_chunk.append(element)
		# When current chunk reaches the desired size, yield it and reset for next chunk
		if len(current_chunk) == elements_per_chunk:
			yield current_chunk
			current_chunk = []
	# If there are remaining elements in the current chunk after the loop, yield them as well
	if current_chunk:
		yield current_chunk

def chunk_paragraph(text, max_chars=5000):
	# Splitting the text into sentences using a simple regex
	# This regex considers ".", "!", and "?" as sentence delimiters.
	sentences = re.split(r'(?<=[.!?]) +', text)
	# Initialize variables
	paragraphs = []
	current_paragraph = ""
	for sentence in sentences:
		# Check if adding the next sentence exceeds the max character limit
		if len(current_paragraph) + len(sentence) > max_chars:
			# If the current paragraph is not empty, add it to the paragraphs list
			if current_paragraph:
				paragraphs.append(current_paragraph.strip())
				current_paragraph = ""
		# Add the sentence to the current paragraph
		current_paragraph += sentence + " "
	# Add the last paragraph if it's not empty
	if current_paragraph:
		paragraphs.append(current_paragraph.strip())
	return paragraphs

