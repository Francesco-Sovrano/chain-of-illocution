async function fetchCsvData(csvUrl) {
	const response = await fetch(csvUrl);
	const csvText = await response.text();
	return new Promise((resolve, reject) => {
		Papa.parse(csvText, {
			header: true,
			complete: function(results) {
				resolve(results.data);
			},
			error: function(err) {
				reject(err);
			}
		});
	});
}

const app = new Vue({
	el: '#app',
	data: {
		index_start_id: 0, // out of 30
		questions_per_type: 3, // out of 30

		stage: 0,
		explanation_options: {
			randomize_explanations: true,
			fast_rating: false,
		},
		timer: {
			time_limit: 0,
			warning_threshold: 5*60,
			alert_threshold: 1*60,
		},
		redirect_link: 'https://app.prolific.com/submissions/complete?cc=CEFBU0VA',
		explanation_csv: '/resources/static/combined_data_gpt-4o_question_5_0.7.csv', // URL of the hosted CSV file
		results_dict: {},
		question_list: [],
		explanation_type_list: ['GenAI','RAG','RAG+CoR'],
		valid_topic_list: [
			'pharo',
			'java',
			'python',
			'design_patterns',
		],
	},
	async created() {
		const data = await fetchCsvData(this.explanation_csv);
		const original_list = data.map(item => ({
			'title': item.title,
			'body': item.body,
			'tool': item.tool,
			'explanation': item.explanation,
			'topic': item.topic,
			'stackoverflow_answer': item.stackoverflow_answer,
		}));

		// Step 1: Convert to the desired intermediate structure
		var intermediate_list = original_list.map(item => {
			// Initialize the explanations object
			let explanations = {};

			// Use a for loop to populate the explanations object based on the tools list
			for (let explanation_type of this.explanation_type_list) {
				explanations[explanation_type] = item.tool === explanation_type ? item.explanation : '';
			}

			return {
				'topic': item.topic,
				'title': item.title,
				'body': item.body,
				'explanations': explanations,
				'stackoverflow_answer': item.stackoverflow_answer,
			};
		});

		// Step 2: Merge items with the same title and body
		const self = this;
		intermediate_list = intermediate_list.reduce((accumulator, current) => {
			let found = accumulator.find(item => item.title === current.title && item.body === current.body);
			if (found) {
				// Update explanations in the found item
				self.explanation_type_list.forEach(key => {
					if (current.explanations[key]) {
						found.explanations[key] = current.explanations[key]; // Update only if current has a non-empty value
					}
				});
			} else {
				// If not found, push the current item as new
				accumulator.push(current);
			}
			return accumulator;
		}, []).filter(item => item.title); // Filter out items with undefined or empty titles

		intermediate_list = intermediate_list.filter(item => this.valid_topic_list.includes(item.topic));

		if (this.questions_per_type > 0) {
			// Group items by topic
			const groupedByTopic = intermediate_list.reduce((acc, item) => {
				acc[item.topic] = acc[item.topic] || [];
				acc[item.topic].push(item);
				return acc;
			}, {});

			// Slice to get x questions starting from the n-th for each topic
			const slicedByTopic = {};
			for (let topic in groupedByTopic) {
				slicedByTopic[topic] = groupedByTopic[topic].slice(this.index_start_id, this.index_start_id+this.questions_per_type); // Slice from 3rd to the 12th item
			}

			// Flatten the object to get a single list of questions
			this.question_list = Object.values(slicedByTopic).flat().filter(item => item.title); // Filter out items with undefined or empty titles
		} else {
			this.question_list = intermediate_list;
		}

		// console.log(this.question_list);
	},
	methods: {
		finish_step_fn: function() {
			this.stage += 1;
			// console.log(this.stage, this.results_dict);
		},
		set_results_fn: function(r) {
			console.log('set_results_fn')
			this.results_dict = Object.assign({}, this.results_dict, r);
		},
		save_results_fn: function(r) {
			console.log('save_results_fn')
			this.set_results_fn(r);
			// this.finish_step_fn(r);
			// this.results_dict['uuid'] = getCookie('uuid');
			const self = this;
			$.ajax({
				url: 'submission', 
				method:'POST',
				data: {'results_dict': JSON.stringify(this.results_dict)},
				contentType: "application/json; charset=utf-8",
				error: x => {
					alert('Something went wrong! Your answer has not been saved.'); 
					self.stage=0;
				},
			});
		},
	},
});