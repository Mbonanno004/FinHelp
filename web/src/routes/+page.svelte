<script lang="ts">
	const steps = [
		{
			icon: '📚',
			title: 'Learn',
			text: 'A neural network studied 24 months of spending from 350 simulated people and learned patterns like winter utility bills, holiday shopping and cutting back after a job loss.'
		},
		{
			icon: '🔮',
			title: 'Predict',
			text: 'Given your last 6 months, it predicts what you’ll spend next month in 11 categories, and shows how accurate it has been on your own past months.'
		},
		{
			icon: '🎯',
			title: 'Budget',
			text: 'A simple set of rules (not AI) compares that prediction with your income and optional savings goal, and trims flexible categories like dining and shopping until the goal fits.'
		}
	];

	const pages = [
		{
			href: '/planner',
			title: 'Budget Planner',
			text: 'Load someone’s spending (an editable example, a CSV upload, or a person from the datasets). See next month’s prediction, get a recommended budget with an optional savings goal, and check how accurate the model is for that person.'
		},
		{
			href: '/compare',
			title: 'Compare Models',
			text: 'Put the main neural network, your own models and simple baselines side by side, scored on the same people none of them trained on. Compare overall, by category, and for one person.'
		},
		{
			href: '/train',
			title: 'Train Model',
			text: 'Change the neural network’s settings (size, learning rate, history length…), retrain it with a live progress chart, and decide whether to keep the new version.'
		}
	];

	const terms = [
		['Model', 'A program that learned patterns from examples. Here it predicts spending.'],
		['Baseline', 'A simple method with no learning, e.g. “next month = average of the last 3 months”. A model is only worth using if it beats the baselines.'],
		['Training / test people', 'The model learns from training people and is graded on test people it never saw, the same way an exam uses new questions.'],
		['Average miss (MAE)', 'On average, how many dollars a prediction is off for one category in one month.'],
		['Accuracy', '100% minus the total miss as a share of total spending.'],
		['Backtest', 'Checking a model on past months by hiding each month and predicting it from the months before.']
	];
</script>

<div class="page">
	<header class="hero">
		<span class="eyebrow">AI course final project</span>
		<h1>Predict next month’s spending, then budget for your goals.</h1>
		<p class="lead">
			FinHelp uses a PyTorch neural network to forecast spending in 11 categories, tells you how much to
			trust that forecast, and turns it into a budget built around a savings goal.
		</p>
		<div class="row">
			<a class="btn primary" href="/planner">Open the Budget Planner →</a>
			<a class="btn" href="/compare">See how the models compare</a>
		</div>
	</header>

	<div class="grid cols-3">
		{#each steps as s, i}
			<div class="card step">
				<div class="num">{i + 1}</div>
				<h3>{s.icon} {s.title}</h3>
				<p class="muted">{s.text}</p>
			</div>
		{/each}
	</div>

	<h2 class="space">The pages</h2>
	<div class="grid cols-3">
		{#each pages as p}
			<a class="card link-card" href={p.href}>
				<h3>{p.title} <span class="arrow">→</span></h3>
				<p class="muted">{p.text}</p>
			</a>
		{/each}
	</div>

	<h2 class="space">Quick start</h2>
	<div class="card">
		<ol class="quick">
			<li>Open <strong>Budget Planner</strong>. An example person is already loaded (open their table to see or edit it).</li>
			<li>Read <strong>Next month’s prediction</strong>: the total and the prediction for each category.</li>
			<li>Under <strong>Income and savings</strong>, check the income and choose <strong>No savings goal</strong>, <strong>Save a % of income</strong> (20% is a common guideline), or <strong>Save for something</strong> specific (e.g. $500/month for a car).</li>
			<li>Read the <strong>Recommended budget</strong>: your bills (rent, other consistent bills, all bills) and spending groups, plus each category’s <strong>max logical spend</strong> and <strong>recommended spend</strong>.</li>
			<li>Scroll to <strong>How accurate is the model for this person?</strong> to see how far the predictions can be trusted. Then try someone from the dataset.</li>
			<li>Visit <strong>Compare Models</strong> to see how well each model does overall.</li>
		</ol>
	</div>

	<div class="grid cols-2 space">
		<div class="card">
			<h3>Why it’s useful</h3>
			<ul>
				<li><strong>Budgets from what you actually do.</strong> It starts from a realistic forecast instead of a generic 50/30/20 rule.</li>
				<li><strong>Shows the trade-off.</strong> It shows how much each category must shrink to hit a goal, or that the goal isn’t reachable yet.</li>
				<li><strong>Honest about accuracy.</strong> Every prediction comes with a check of how well the model did on that person’s past months.</li>
				<li><strong>A test bench.</strong> You can compare different machine-learning methods on equal footing.</li>
			</ul>
		</div>
		<div class="card">
			<h3>Limits to keep in mind</h3>
			<ul>
				<li>Trained on <strong>simulated</strong> people, so real bank data may behave differently.</li>
				<li>Rare one-off costs (trips, taxes) can’t really be predicted. The model predicts a <em>typical</em> month, so keep a buffer for surprises.</li>
				<li>It needs at least 6 months of history, and 7+ months to check its own accuracy.</li>
				<li>This is a class project, not financial advice.</li>
			</ul>
		</div>
	</div>

	<details class="collapse">
		<summary>Key terms</summary>
		<div class="content">
			<dl>
				{#each terms as [term, def]}
					<dt>{term}</dt>
					<dd class="muted">{def}</dd>
				{/each}
			</dl>
		</div>
	</details>
</div>

<style>
	.hero {
		padding: 1.5rem 0 2rem;
		max-width: 780px;
	}
	.eyebrow {
		display: inline-block;
		font-size: 0.78rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--primary);
		background: var(--primary-soft);
		border-radius: 999px;
		padding: 0.2rem 0.7rem;
		margin-bottom: 0.9rem;
	}
	.hero h1 {
		font-size: 2.4rem;
		line-height: 1.15;
	}
	.lead {
		font-size: 1.08rem;
		color: var(--muted);
		margin-bottom: 1.25rem;
	}
	.hero .btn {
		text-decoration: none;
	}
	.step {
		position: relative;
	}
	.num {
		position: absolute;
		top: 1rem;
		right: 1.1rem;
		font-weight: 700;
		color: var(--border-strong);
		font-size: 1.4rem;
	}
	.space {
		margin-top: 2.25rem;
		margin-bottom: 0.9rem;
	}
	.link-card {
		text-decoration: none;
		color: inherit;
		transition: border-color 0.15s, transform 0.15s;
	}
	.link-card:hover {
		border-color: var(--primary);
		transform: translateY(-2px);
	}
	.arrow {
		color: var(--primary);
	}
	.quick {
		margin: 0;
		padding-left: 1.2rem;
	}
	.quick li,
	ul li {
		margin-bottom: 0.4rem;
	}
	ul {
		padding-left: 1.1rem;
		margin: 0;
	}
	dt {
		font-weight: 600;
		margin-top: 0.6rem;
	}
	dd {
		margin: 0.15rem 0 0;
	}
</style>
