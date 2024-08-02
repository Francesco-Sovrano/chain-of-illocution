from bottle import run, get, post, route, hook, request, response, static_file
from random import random
import sys
import uuid
import json
import urllib.parse
import os

port = int(sys.argv[1])
model = sys.argv[2]

###############################################################
# CORS

@route('/<:re:.*>', method='OPTIONS')
def enable_cors_generic_route():
	"""
	This route takes priority over all others. So any request with an OPTIONS
	method will be handled by this function.

	See: https://github.com/bottlepy/bottle/issues/402

	NOTE: This means we won't 404 any invalid path that is an OPTIONS request.
	"""
	add_cors_headers()

@hook('after_request')
def enable_cors_after_request_hook():
	"""
	This executes after every route. We use it to attach CORS headers when
	applicable.
	"""
	add_cors_headers()

def add_cors_headers():
	try:
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
		response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
	except Exception as e:
		print('Error:',e)

###############################################################
# Static Routes

@get("/favicon.ico")
def favicon():
	return static_file("favicon.ico", root="static/img/")

###################################

@get("/resources/static/component/<filepath:re:.*>")
def docs(filepath):
	return static_file(filepath, root="static/component/")

###################################

@get("/resources/static/<filepath:re:.*\.css>")
def css(filepath):
	return static_file(filepath, root="static/css/")

@get("/resources/static/<filepath:re:.*\.(eot|otf|svg|ttf|woff|woff2?)>")
def font(filepath):
	return static_file(filepath, root="static/css/")

@get("/resources/static/<filepath:re:.*\.(jpg|png|gif|ico|svg)>")
def img(filepath):
	return static_file(filepath, root="static/img/")

@get("/resources/static/<filepath:re:.*\.js>")
def js(filepath):
	return static_file(filepath, root="static/js/")

@get("/resources/static/<filepath:re:.*\.csv>")
def js(filepath):
	return static_file(filepath, root="static/csv/")

@get("/resources/static/<filepath:re:.*\.vue>")
def js(filepath):
	return static_file(filepath, root="static/js/")

@get("/resources/static/<filepath:re:.*\.json>")
def js(filepath):
	return static_file(filepath, root="static/json/")

@get("/<filepath:re:.*\.html>")
def html(filepath):
	print(filepath)
	return static_file(filepath, root="static/html/")

@get("/")
def home():
	resp = static_file("index.html", root="static/html/")
	if not request.get_cookie("uuid"):
		user_uid = str(uuid.uuid4())#.int
		resp.set_cookie("uuid", user_uid)
	# 	print('new',user_uid)
	# else:
	# 	print('old',request.get_cookie("uuid"))
	return resp

@post("/submission")
def submission():
	user_uid = request.get_cookie("uuid")
	results_dict = json.loads(request.forms.get('results_dict'))
	# print(results_dict)
	username = results_dict['username']
	# print(json.dumps(results_dict, indent=4))
	print(f'{username} ({user_uid}) is submitting', results_dict)
	with open(f"results/{model}/{username}.json", 'w') as f:
		json.dump(results_dict, f, indent=4)

@get("/storage")
def storage():
	response.content_type = 'application/json'
	# question = request.forms.get('question') # post
	username = urllib.parse.unquote(request.query.get('username'))

	# Check if username is provided in the query string.
	if not username:
		return json.dumps({'error': 'Username parameter is missing.'})

	# Defines the path to the user-specific results file.
	file_path = f"results/{model}/{username}.json"
	
	# Checks if the file exists and is accessible.
	if not os.path.exists(file_path):
		return json.dumps({'error': 'User data not found.'})

	try:
		# Opens and reads the JSON data file for the given username.
		with open(file_path, 'r') as f:
			results_dict = json.load(f)
		return json.dumps(results_dict)
	except Exception as e:
		# Handles unexpected errors in file reading or JSON parsing.
		return json.dumps({'error': f'An error occurred: {str(e)}'})

if __name__ == "__main__":
	run(server='tornado', host='0.0.0.0', port=port, debug=False)
	# run(server='tornado', host='0.0.0.0', port=port, debug=False)
	