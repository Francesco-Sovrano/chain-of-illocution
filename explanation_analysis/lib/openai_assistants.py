import time

def wait_on_run(client, run, thread):
	while run.status == "queued" or run.status == "in_progress":
		run = client.beta.threads.runs.retrieve(
			thread_id=thread.id,
			run_id=run.id,
		)
		time.sleep(0.5)
	return run

def submit_message(client, assistant_id, thread, user_message, sentence_retriever_options):
	client.beta.threads.messages.create(
		thread_id=thread.id, role="user", content=user_message
	)
	return client.beta.threads.runs.create(
		thread_id=thread.id,
		assistant_id=assistant_id,
		model=sentence_retriever_options['generative_ai_options']['model'],
		temperature=sentence_retriever_options['generative_ai_options']['temperature'],
		top_p=sentence_retriever_options['generative_ai_options']['top_p'],
	)

def get_response(client, thread):
	return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def create_thread_and_run(client, assistant_id, user_input, sentence_retriever_options):
	thread = client.beta.threads.create()
	run = submit_message(client, assistant_id, thread, user_input, sentence_retriever_options)
	# print(run)
	return thread, run