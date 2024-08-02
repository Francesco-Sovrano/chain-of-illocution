function removeTextWithinBrackets(text) {
	// This regex matches any characters within 【】 including the brackets themselves
	const regex_gpt = /【[^】]*】/g;
	const regex_yai = /\([^(]*\)/g;
	return text.replace(regex_gpt, '').replace(regex_yai, '');
}

Vue.component("explanationManager", {
	template: `
		<div>
			<b-navbar toggleable="lg" variant="dark" type="dark" class="rounded">
				<p id="goal" class="text-white text-wrap text-break my-auto py-2 px-1">
					For each question (comprising a title and a body), evaluate the three explanatory answers below in a scale from 0 (very bad) to 10 (very good) stars.
					<br>
					We will use your feedback to understand what the best explanations are.
				</p>

				<div v-if="timer.time_limit" class="ml-auto m-1 text-white">
					<base-timer 
						ref="timer"
						id="timer"
						:time_limit="timer.time_limit" 
						:warning_threshold="timer.warning_threshold" 
						:alert_threshold="timer.alert_threshold"
						:finish_fn="finish_fn"
					></base-timer>
					<b-tooltip target="timer" triggers="hover">
						Time Left
					</b-tooltip>
				</div>

				<div v-if="!timer.time_limit" class="ml-auto m-1" style="color: #88B04B;">
					<span class="font-weight-bold">Questions completed: {{ stage + 1 }} / {{ question_list.length }}</span>
				</div>
			</b-navbar>
			<br v-if="timer.time_limit">
			<br>
			<div class="m-auto"
				v-for="(question, index) in question_list"
				v-if="index==stage"
				:key="index"
			>
				<b-row>
					<b-col cols="10" class="mx-auto border-bottom">
						<div>
							<h2><u>{{titlefy(question.topic.replace('_',' '))}}</u>: {{titlefy(question.title)}}</h2>
						</div>
						<div>
							<div v-html="question.body"></div>
						</div>
					</b-col>
				</b-row>
				<br>
				
				<b-row>
					<b-col cols="1" class="mx-auto text-left">
						<b-button v-if="current_explanation > 1" v-on:click="goto(current_explanation-1)">&laquo; Prev.</b-button>
					</b-col>
					<!-- Empty column for alignment -->
				    <b-col cols="10">
				    </b-col>
				    <b-col cols="1" class="mx-auto text-right">
						<b-button v-if="current_explanation < 3" v-on:click="goto(current_explanation+1)">Next &raquo;</b-button>
					</b-col>

					
					<b-col cols="1">
				    </b-col>
					<b-col cols="10">
						<div v-if="current_explanation==1">
							<div class="btn btn-warning btn-block" style="pointer-events: none;"><b>1st Explanation &darr;</b></div>
							<p class="my-4" v-html="md.render(removeTextWithinBrackets(question.explanations[explanation_type_list[0]]))">
							</p>
						</div>
						<div v-if="current_explanation==2">
							<div class="btn btn-warning btn-block" style="pointer-events: none;"><b>2nd Explanation &darr;</b></div>
							<p class="my-4" v-html="md.render(removeTextWithinBrackets(question.explanations[explanation_type_list[1]]))">
							</p>
						</div>
						<div v-if="current_explanation==3">
							<div class="btn btn-warning btn-block" style="pointer-events: none;"><b>3rd Explanation &darr;</b></div>
							<p class="my-4" v-html="md.render(removeTextWithinBrackets(question.explanations[explanation_type_list[2]]))">
							</p>
						</div>

						<div class="m-1 rounded-0 bg-light border rounded p-2">
							<h4><u>Questionnaire</u></h4>
							<div v-for="(set, setIndex) in evaluation_sets" :key="'set-' + setIndex">
								<b-row
									class="m-auto border-bottom py-2"
									v-for="(evaluation, index) in set.evaluations"
									v-show="current_explanation == index + 1"
									:key="'eval-' + setIndex + '-' + index"
								>
									<div class="my-auto col-5 text-left">
										<label v-html='set.question'></label>
									</div>
									<div class="my-auto col-7">
										<b-form-rating stars="10" show-value show-value-max :variant="evaluation.value != null ? 'success' : 'danger'" v-model="evaluation.value" @change="fast_change_explanation"></b-form-rating>
									</div>
									<br>
									<div v-if="set.with_extra" class="p-2 m-2 rounded" style="background: antiquewhite;">
										<div v-html="question.stackoverflow_answer">
										</div>
									</div>
								</b-row>
							</div>
							<b-row class="m-auto">
								<b-button block variant="success" @click="move_to_next_question" :disabled="total_evaluations !== explanation_type_list.length">
									{{ total_evaluations === explanation_type_list.length ? 'Go to the next question' : 'Rated explanations: ' + total_evaluations + '/' + explanation_type_list.length }}
								</b-button>
							</b-row>
						</div>
					</b-col>
					<b-col cols="1">
				    </b-col>

					<b-col cols="1" class="mx-auto mt-auto text-left">
						<b-button v-if="current_explanation > 1" v-on:click="goto(current_explanation-1)">&laquo; Prev.</b-button>
					</b-col>	
					<!-- Empty column for alignment -->
				    <b-col cols="10">
				    </b-col>
					<b-col cols="1" class="mx-auto mt-auto text-right">
						<b-button v-if="current_explanation < 3" v-on:click="goto(current_explanation+1)">Next &raquo;</b-button>
					</b-col>
				</b-row>
			</div>
		</div> 
	`,
	props: {
		question_list: Array,
		explanation_options: {
			type: Array,
			default: {
				randomize_explanations: false,
				fast_rating: true,
			},
		},
		explanation_type_list: {
			type: Array,
			required: true,
		},
		timer: {
			type: Array,
			default: {
				time_limit: 10*60,
				warning_threshold: 3*60,
				alert_threshold: 1*60,
			},
		},
		finish_fn: {
			type: Function,
			default: function (r) {}
		},
		save_results_fn: {
			type: Function,
			default: function (r) {}
		},
		stage: {
			type: Number,
			default: 0
		},
		result_list: {
			type: Array,
			default: []
		},
	},
	data() {
		return {
			current_explanation: 1,
			stage_timestamp: Math.floor(Date.now() / 1000),
			evaluation_sets: null,
			md: new markdownit(),
		}
	},
	methods: {
		shuffle_array: function(array) {
			for (var i = array.length - 1; i > 0; i--) {
				// Generate random number
				const j = Math.floor(Math.random() * (i + 1));
				const temp = array[i];
				array[i] = array[j];
				array[j] = temp;
			}
			return array;
		},
		move_to_next_question: function() {
			const now = Math.floor(Date.now() / 1000);
			const evaluation_dict = {};
			for (let j = 0; j < this.evaluation_sets.length; j++)
			{
				const _type = this.evaluation_sets[j].type;
				evaluation_dict[_type] = {};
				for (let i = 0; i < this.explanation_type_list.length; i++)
					evaluation_dict[_type][this.explanation_type_list[i]] = this.evaluation_sets[j].evaluations[i].value;
			}
			this.result_list.push({
				'question': this.question_list[this.stage].title,
				'topic': this.question_list[this.stage].topic,
				'evaluation_dict': evaluation_dict,
				'elapsed_seconds': now - this.stage_timestamp,
			});
			this.stage += 1;
			this.reset_evaluation_sets();
			this.current_explanation = 1;
			this.stage_timestamp = now;
			this.save_results_fn({'evaluation_list': this.result_list});
			if (this.stage >= this.question_list.length)
				this.finish_fn();
			else
				window.scrollTo(0, 0);
		},
		fast_change_explanation: function(evaluations) {
			if (this.explanation_options['fast_rating'])
			{
				if (this.evaluation_sets.some(q => q.evaluations[this.current_explanation-1 % this.explanation_type_list.length].value == null))
					return;
				
				if (!this.evaluation_sets.some(q => q.evaluations[this.current_explanation % this.explanation_type_list.length].value == null))
					return;

				if (this.current_explanation==3)
					this.goto(1);
				else
					this.goto(this.current_explanation+1);
				// // alert('Thank you! Please evaluate also the next explanation.')
			}
		},
		goto: function(new_explanation) {
			this.current_explanation = new_explanation;
			window.scrollTo(0, 0);
		},
		reset_evaluation_sets() {
			this.evaluation_sets = [
				{
					type: "Net Promoter Score",
					with_extra: false,
					question: "How likely is it that you would recommend this explanation to a software engineer/developer?",
					evaluations: [
						{ value: null },
						{ value: null },
						{ value: null }
					]
				},
				{
					type: "Relevance",
					with_extra: false,
					question: "What is the degree to which this explanation directly addresses and is appropriate for the given question and topic?",
					evaluations: [
						{ value: null },
						{ value: null },
						{ value: null }
					]
				},
				{
					type: "Correctness",
					with_extra: true,
					question: "Does the explanation correctly answer the question posed? For reference, below you'll find one of many possible examples of the correct answer.",
					evaluations: [
						{ value: null },
						{ value: null },
						{ value: null }
					]
				},
				// {
				// 	type: "Faithfulness/Consistency",
				// 	with_extra: true,
				// 	question: "How well do the following statements (see below) align with the information provided in the explanation above?",
				// 	evaluations: [
				// 		{ value: null },
				// 		{ value: null },
				// 		{ value: null }
				// 	]
				// }
			]
		},
	},
	created: function() {
		this.reset_evaluation_sets();
		if (this.explanation_options['randomize_explanations'])
			this.explanation_type_list = this.shuffle_array(this.explanation_type_list);
	},
	computed: {
		total_evaluations() {
			return Math.floor(this.evaluation_sets.reduce((sum, set) => sum + set.evaluations.filter(eval => eval.value != null).length, 0)/ this.evaluation_sets.length);
		},
	},
});