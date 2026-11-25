<script lang="ts">
	import { ACCURACY_STEPS } from '#lib/format.js';
</script>

<ol class="steps">
	{#each ACCURACY_STEPS as step}
		<li><strong>{step.title}</strong> {step.text}</li>
	{/each}
</ol>
<p class="muted small">
	This is called <strong>backtesting</strong> (also known as walk-forward validation). For the dataset-wide
	scores, it's done on <strong>test people the models never saw during training</strong>, so the scores are
	honest.
</p>

<style>
	.steps {
		padding-left: 1.2rem;
		margin: 0.25rem 0 0.75rem;
	}
	.steps li {
		margin-bottom: 0.45rem;
	}
</style>
