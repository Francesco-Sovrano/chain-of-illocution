
Vue.component("initialQuestionnaire", {
	template: `
	<div>
		<h1><u>Introduction to the Study</u></h1>

		<b-alert v-model="showDismissibleAlert" variant="warning" dismissible>
			This app is not designed for smartphones or other devices with small screens.
			<br>
			<b>Using a tablet (in landscape mode), a laptop, or a desktop is required</b>.
		</b-alert>

		<br>
		<p>
			Answer the following questions. 
			When finished, you will be able to click the <i>"Start"</i> button.
			<b v-if="time_limit">You will have less than {{time_limit/60}} minutes to complete the study.</b>
		</p>

		<b-row class="my-1">
			<b-col class="my-auto" sm="2">
				<label>What is your <b>Prolific ID</b>?</label>
			</b-col>
			<b-col class="my-auto">
				<b-form-input 
					type="text" 
					v-model="username" 
					:state="username!=null && username!=''"
					@keyup.enter="checkUsername"
					@blur="checkUsername"
				></b-form-input>
			</b-col>
		</b-row>
		
		<div v-if="new_user!=null">
			<div v-if="new_user">
				<hr flex/>
				<br>

				<p>
					For each question provide a score in a scale from 0 (very bad) to 5 (very good) stars.
				</p>
				<b-container fluid>
					<b-row class="my-1" v-for="(qdict,label) in question_dict" :key="label">
						<b-col class="my-auto" sm="3">
							<label v-html="label"></label>
						</b-col>
						<b-col class="my-auto" sm="9">
							<b-form-input v-if="qdict.type=='number' || qdict.type=='range'" :type="qdict.type" v-model="qdict.value" :state="qdict.value!=null && qdict.value!='' && qdict.value>=qdict.min && qdict.value<=qdict.max" :placeholder="qdict.placeholder" :min="qdict.min" :max="qdict.max"></b-form-input>
							<b-form-input v-if="qdict.type!='select' && qdict.type!='number' && qdict.type!='rating'" :type="qdict.type" v-model="qdict.value" :state="qdict.value!=null && qdict.value!=''" :placeholder="qdict.placeholder"></b-form-input>
							<b-form-select v-if="qdict.type=='select'" :type="qdict.type" v-model="qdict.value" :state="qdict.value!=null && qdict.value!=''" :options="qdict.options"></b-form-select>
							<b-form-rating v-if="qdict.type=='rating'" show-value show-value-max :variant="qdict.value!=null?'success':'danger'" v-model="qdict.value"></b-form-rating>
						</b-col>
					</b-row>
				</b-container>
			</div>
			<div v-else>
				<hr flex/>
				<p>
					Welcome back <b>{{username}}</b>! Your last data checkpoint has been loaded.
				</p>
				
			</div>
		</div>

		<hr flex/>
		<br>

		<p>
			Click on the <i>"Start"</i> button to begin the study.
			The study consists of evaluating explanatory answers to <b>{{question_list.length}} different questions</b> related to Python, Java, Pharo, and object-oriented design patterns.
			<br>
			You will be provided with <b>3 different explanations per question</b>.
			<b>Rate the explanations on a scale of 0 (bad) to 10 (good) stars.</b>
			<br>
			We will use your feedback to understand which are the best explanations.
		</p>

		<br>
		<b-button 
			class="float-right mb-2" 
			@click="submit" 
			variant="primary" 
			size="lg"
			:disabled="!can_submit"
		>
			<span class="m-2">Start</span>
		</b-button>
	</div>
	`,
	props: {
		question_list: Array,
		time_limit: {
			type: Number,
			default: 10*60,
		},
		finish_fn: {
			type: Function,
			default: function (r) {}
		},
		save_results_fn: {
			type: Function,
			default: function (r) {}
		},
		set_results_fn: {
			type: Function,
			default: function (r) {}
		}
	},
	data() {
		return {
			question_dict: {
				'How would you rate your knowledge of Pharo?': {
					'type': 'rating',
					'value': null,
				},
				'How would you rate your knowledge of <b>Python</b>?': {
					'type': 'rating',
					'value': null,
				},
				'How would you rate your knowledge of <b>Java</b>?': {
					'type': 'rating',
					'value': null,
				},
				'How would you rate your knowledge of object-oriented design patterns?': {
					'type': 'rating',
					'value': null,
				},
			},
			showDismissibleAlert: true,
			username: null, // Initial state of the username
			new_user: null,
			results_dict: {}
		}
	},
	computed: {
		can_submit: function() {
			if (this.new_user!= null && !this.new_user)
				return true;
			for (var [label,qdict] of Object.entries(this.question_dict))
			{
				if (!qdict.value)
					return false;
			}
			return true;
		},
	},
	methods: {
		checkUsername() {
			if (this.username !== null && this.username !== '') {
				self = this;
				const encodedUsername = encodeURIComponent(this.username); // Encode the username to ensure it is URL-safe.
				$.ajax({
						url: 'storage', // Construct the URL with the query parameter
						type: 'GET',
						data: {'username': encodedUsername},
						dataType: 'json', // Expect JSON response from the server
						success: function(data) {
								if (!data.error) {
										self.new_user = false;
										self.results_dict = data;
										// console.log(self.results_dict);
										self.set_results_fn(self.results_dict);
								} else {
										self.new_user = true;
								}
						},
						error: function(jqXHR, textStatus, errorThrown) {
							self.new_user = true;
						}
				});
			}
		},
		submit: function() {
			this.results_dict['username'] = this.username
			for (var [label,qdict] of Object.entries(this.question_dict))
				this.results_dict[label] = qdict.value;

			if (this.new_user)
				this.save_results_fn(this.results_dict);
			this.finish_fn();
		},
	},
	
});
