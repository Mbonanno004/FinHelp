<script lang="ts">
	import { onMount } from 'svelte';
	import type { ChartConfiguration } from 'chart.js';
	import { get, post, upload } from '#lib/api.js';
	import { money, pct, signed, SERIES_COLORS } from '#lib/format.js';
	import { app, planner } from '#lib/state.svelte.js';
	import type { Backtest, Budget, Forecast, ModelInfo, MonthRow } from '#lib/types.js';
	import Section from '#lib/components/Section.svelte';
	import Metric from '#lib/components/Metric.svelte';
	import Chart from '#lib/components/Chart.svelte';
	import PersonPicker from '#lib/components/PersonPicker.svelte';
	import AccuracyExplanation from '#lib/components/AccuracyExplanation.svelte';

	const SIMPLE_AVG = '3-month average';

	let models: ModelInfo[] = $state([]);
	let datasetRows: MonthRow[] = $state([]);
	let inTraining = $state(false);
	let error = $state('');
	let forecast = $state<Forecast | null>(null);
	let forecastError = $state('');
	let budget = $state<Budget | null>(null);
	let accuracy = $state<Backtest | null>(null);
	let accuracyError = $state('');
	let loading = $state(false);

	const categories = $derived(app.meta?.categories ?? []);
	const choices = $derived(models.filter((m) => m.kind !== 'Baseline'));
	const flexible = $derived(
		Object.entries(app.meta?.flexibility ?? {})
			.filter(([, v]) => v > 0)
			.map(([c]) => c)
	);

	const rows: MonthRow[] = $derived(
		planner.source === 'example'
			? planner.exampleRows
			: planner.source === 'upload'
				? (planner.uploadPeople.find((p) => p.sample_id === planner.uploadId)?.rows ?? [])
				: datasetRows
	);

	// Which person is loaded; when it changes, the income box resets to their recent income.
	const personKey = $derived(
		planner.source === 'example'
			? 'example'
			: planner.source === 'upload'
				? `upload-${planner.uploadId}`
				: `dataset-${datasetRows[0]?.month}-${datasetRows.length}-${datasetRows[0]?.Income}`
	);
	let incomeFor = '';

	onMount(async () => {
		try {
			const [m, ex] = await Promise.all([
				get<{ models: ModelInfo[] }>('/models'),
				planner.exampleRows.length ? null : get<{ rows: MonthRow[] }>('/example')
			]);
			models = m.models;
			if (!planner.model || !models.some((x) => x.name === planner.model)) {
				planner.model = models.find((x) => x.kind !== 'Baseline')?.name ?? '';
			}
			if (ex) planner.exampleRows = ex.rows;
		} catch (e) {
			error = (e as Error).message;
		}
	});

	// ---- Prediction + accuracy whenever the history or model changes ----
	let historyTimer: ReturnType<typeof setTimeout>;
	$effect(() => {
		const body = { model: planner.model, rows: $state.snapshot(rows) };
		const key = personKey;
		clearTimeout(historyTimer);
		if (!body.model || !body.rows.length) {
			forecast = null;
			accuracy = null;
			return;
		}
		historyTimer = setTimeout(async () => {
			loading = true;
			try {
				forecast = await post<Forecast>('/forecast', body);
				forecastError = '';
				if (incomeFor !== key || planner.income === null) {
					planner.income = Math.round(forecast.recent_income / 10) * 10;
					incomeFor = key;
				}
			} catch (e) {
				forecast = null;
				forecastError = (e as Error).message;
			}
			try {
				accuracy = await post<Backtest>('/backtest', { models: [body.model, SIMPLE_AVG], rows: body.rows });
				accuracyError = '';
			} catch (e) {
				accuracy = null;
				accuracyError = (e as Error).message;
			}
			loading = false;
		}, 250);
	});

	// ---- Savings target in dollars, from whichever savings mode is chosen ----
	const savingsTarget = $derived(
		planner.savingsMode === 'percent'
			? Math.round(((planner.income ?? 0) * planner.savingsRate) / 100)
			: planner.savingsMode === 'specific'
				? (planner.goal ?? 0)
				: 0
	);
	const goalOn = $derived(savingsTarget > 0);
	const budgetTitle = $derived(
		planner.savingsMode === 'percent'
			? `Recommended budget: saving ${planner.savingsRate}% of income`
			: planner.savingsMode === 'specific' && planner.goalName
				? `Recommended budget: ${planner.goalName}`
				: 'Recommended budget'
	);

	// ---- Budget whenever the prediction or goal settings change ----
	let budgetTimer: ReturnType<typeof setTimeout>;
	$effect(() => {
		const body = forecast && {
			predicted: Object.fromEntries(forecast.categories.map((c) => [c.category, c.predicted])),
			typical: forecast.typical,
			income: planner.income ?? 0,
			goal: savingsTarget,
			max_cut: planner.maxCut / 100,
			protected: [...planner.protected],
			target: planner.savingsMode === 'specific' ? planner.target : null
		};
		clearTimeout(budgetTimer);
		if (!body) {
			budget = null;
			return;
		}
		budgetTimer = setTimeout(async () => {
			try {
				budget = await post<Budget>('/budget', body);
			} catch (e) {
				error = (e as Error).message;
			}
		}, 120);
	});

	// ---- Example table editing ----
	function addMonth() {
		const last = planner.exampleRows.at(-1);
		if (!last) return;
		const [y, m] = last.month.split('-').map(Number);
		const next = m === 12 ? `${y + 1}-01` : `${y}-${String(m + 1).padStart(2, '0')}`;
		planner.exampleRows.push({ ...$state.snapshot(last), month: next });
	}

	function removeMonth(i: number) {
		planner.exampleRows.splice(i, 1);
	}

	async function resetExample() {
		planner.exampleRows = (await get<{ rows: MonthRow[] }>('/example')).rows;
	}

	async function onUpload(event: Event) {
		const file = (event.target as HTMLInputElement).files?.[0];
		if (!file) return;
		try {
			const r = await upload<{ people: { sample_id: number; rows: MonthRow[] }[] }>('/upload', file);
			planner.uploadPeople = r.people;
			planner.uploadId = r.people[0]?.sample_id ?? null;
			error = '';
		} catch (e) {
			error = (e as Error).message;
		}
	}

	function toggleProtected(c: string) {
		const i = planner.protected.indexOf(c);
		if (i >= 0) planner.protected.splice(i, 1);
		else planner.protected.push(c);
	}

	// ---- Derived display values ----
	const modelSummary = $derived(accuracy?.summary.find((s) => s.model === planner.model));
	const avgSummary = $derived(accuracy?.summary.find((s) => s.model === SIMPLE_AVG));

	const budgetChart: ChartConfiguration | null = $derived(
		budget && {
			type: 'bar',
			data: {
				labels: budget.table.map((r) => r.category),
				datasets: [
					{ label: 'Predicted', data: budget.table.map((r) => r.predicted), backgroundColor: '#a5b4fc', borderRadius: 4 },
					{ label: 'Recommended', data: budget.table.map((r) => r.recommended), backgroundColor: '#3b5bdb', borderRadius: 4 },
					{ label: 'Max logical spend', data: budget.table.map((r) => r.max), backgroundColor: '#dee2e6', borderRadius: 4 }
				]
			},
			options: {
				indexAxis: 'y',
				maintainAspectRatio: false,
				plugins: {
					legend: { position: 'bottom' },
					tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${money(c.parsed.x as number)}` } }
				},
				scales: { x: { ticks: { callback: (v) => money(Number(v)) } }, y: { grid: { display: false } } }
			}
		}
	);

	const accuracyChart: ChartConfiguration | null = $derived(
		accuracy && {
			type: 'line',
			data: {
				labels: accuracy.per_month.map((m) => m.month),
				datasets: [
					{ label: 'Actual', data: accuracy.per_month.map((m) => m.actual), borderColor: SERIES_COLORS[0], backgroundColor: SERIES_COLORS[0], borderWidth: 3, tension: 0 },
					...[planner.model, SIMPLE_AVG].map((name, i) => ({
						label: name,
						data: accuracy!.per_month.map((m) => m.predicted[name]),
						borderColor: SERIES_COLORS[i + 1],
						backgroundColor: SERIES_COLORS[i + 1],
						borderDash: [6, 4],
						tension: 0
					}))
				]
			},
			options: {
				maintainAspectRatio: false,
				interaction: { mode: 'index', intersect: false },
				plugins: {
					legend: { position: 'bottom' },
					tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${money(c.parsed.y as number)}` } }
				},
				scales: { y: { ticks: { callback: (v) => money(Number(v)) } }, x: { grid: { display: false } } }
			}
		}
	);
</script>

<div class="page">
	<h1>Budget Planner</h1>
	<div class="callout info">
		Give the app someone’s past monthly spending. It <strong>predicts next month</strong>, builds a
		<strong>budget</strong> (optionally around a savings goal), and shows <strong>how accurate</strong> its
		predictions have been for this person. Work top to bottom.
	</div>
	{#if error}<div class="callout error">{error}</div>{/if}

	<!-- ================= History ================= -->
	<Section
		title="Spending history"
		purpose="The model learns about this person from their past months. It needs at least 6 months; with 7 or more it can also check its own accuracy."
	>
		<div class="grid cols-2">
			<label class="field">
				Where should the history come from?
				<select bind:value={planner.source}>
					<option value="example">Example person (editable)</option>
					<option value="upload">Upload a transactions CSV</option>
					<option value="dataset">Pick someone from the dataset</option>
				</select>
			</label>
			<label class="field">
				Prediction model
				<select bind:value={planner.model} title="Your own models appear here once they are added and trained (see Compare Models).">
					{#each choices as m}<option value={m.name}>{m.name}</option>{/each}
				</select>
			</label>
		</div>

		{#if planner.source === 'example'}
			<details class="collapse">
				<summary>Example person’s monthly spending (open to view or edit)</summary>
				<div class="content">
					<p class="muted small">One row per month, oldest first. Click any cell to change it.</p>
					<div class="table-wrap">
						<table class="edit">
							<thead>
								<tr>
									<th>Month</th>
									<th class="num">Income</th>
									{#each categories as c}<th class="num">{c}</th>{/each}
									<th></th>
								</tr>
							</thead>
							<tbody>
								{#each planner.exampleRows as row, i}
									<tr>
										<td><input type="text" bind:value={row.month} class="cell month" /></td>
										<td><input type="number" bind:value={row.Income} class="cell" /></td>
										{#each categories as c}
											<td><input type="number" bind:value={row[c]} class="cell" /></td>
										{/each}
										<td>
											<button class="remove" title="Remove this month" onclick={() => removeMonth(i)}>×</button>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
					<div class="row" style="margin-top: 0.75rem">
						<button class="btn small" onclick={addMonth}>+ Add a month</button>
						<button class="btn small" onclick={resetExample}>Reset example</button>
					</div>
				</div>
			</details>
		{:else if planner.source === 'upload'}
			<p class="muted small upload-help">
				Same columns as <code>data/synthetic_transactions.csv</code>: sample_id, transaction_type, amount,
				merchant_category, balance_after, transaction_date (MM-DD-YYYY).
			</p>
			<div class="grid cols-2">
				<label class="field">
					Transactions CSV
					<input type="file" accept=".csv" onchange={onUpload} />
				</label>
				{#if planner.uploadPeople.length}
					<label class="field">
						Person (sample_id)
						<select bind:value={planner.uploadId}>
							{#each planner.uploadPeople as p}<option value={p.sample_id}>{p.sample_id}</option>{/each}
						</select>
					</label>
				{/if}
			</div>
		{:else}
			<div style="margin-top: 1rem">
				<PersonPicker
					onload={(r, t) => {
						datasetRows = r;
						inTraining = t;
					}}
					onerror={(m) => (error = m)}
				/>
			</div>
		{/if}

		{#if planner.source !== 'example' && rows.length}
			<details class="collapse">
				<summary>See the monthly totals the model reads ({rows.length} months)</summary>
				<div class="content table-wrap">
					<table>
						<thead>
							<tr><th>Month</th><th class="num">Income</th>{#each categories as c}<th class="num">{c}</th>{/each}</tr>
						</thead>
						<tbody>
							{#each rows as r}
								<tr>
									<td>{r.month}</td>
									<td class="num">{money(r.Income)}</td>
									{#each categories as c}<td class="num">{money(Number(r[c]))}</td>{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</details>
		{/if}
	</Section>

	{#if forecastError}
		<div class="callout warning" style="margin-top: 1.5rem">{forecastError}</div>
	{/if}

	{#if forecast}
		<!-- ================= Prediction ================= -->
		<Section
			title="Next month’s prediction ({forecast.month})"
			purpose="What the model expects to be spent next month, based on the months above."
		>
			<div class="grid cols-3">
				<Metric label="Predicted spending" value={money(forecast.predicted_total)} />
				<Metric
					label="Recent average (last 3 months)"
					value={money(forecast.recent_average_total)}
					delta="{money(forecast.predicted_total - forecast.recent_average_total)} predicted vs. average"
				/>
				<Metric label="Recent income (last 3 months)" value={money(forecast.recent_income)} />
			</div>
			<div class="table-wrap" style="margin-top: 1rem">
				<table>
					<thead>
						<tr><th>Category</th><th class="num">Predicted</th><th class="num">Recent average</th><th class="num">Difference</th></tr>
					</thead>
					<tbody>
						{#each forecast.categories as c}
							{@const diff = c.predicted - c.recent_average}
							<tr>
								<td class="label">{c.category}</td>
								<td class="num">{money(c.predicted)}</td>
								<td class="num">{money(c.recent_average)}</td>
								<td class="num" class:pos={diff < -0.5} class:neg={diff > 0.5}>{signed(diff)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</Section>

		<!-- ================= Income and savings ================= -->
		<Section
			title="Income and savings"
			purpose="Set next month’s expected income and choose how much you want to save. The budget below adapts to your choice."
		>
			<div class="grid cols-2">
				<label class="field">
					Expected income next month ($)
					<input type="number" min="0" step="50" bind:value={planner.income} title="Starts at this person’s average income over the last 3 months." />
				</label>
				<div class="field">
					Savings
					<div class="segmented">
						<button class:on={planner.savingsMode === 'none'} onclick={() => (planner.savingsMode = 'none')}>No savings goal</button>
						<button class:on={planner.savingsMode === 'percent'} onclick={() => (planner.savingsMode = 'percent')}>Save a % of income</button>
						<button class:on={planner.savingsMode === 'specific'} onclick={() => (planner.savingsMode = 'specific')}>Save for something</button>
					</div>
				</div>
			</div>

			{#if planner.savingsMode === 'percent'}
				<div class="card goal-box">
					<label class="field">
						<span>Save <strong class="cut">{planner.savingsRate}%</strong> of income = <strong class="cut">{money(savingsTarget)}</strong> per month</span>
						<input type="range" min="5" max="50" step="1" bind:value={planner.savingsRate} />
						<span class="hint">20% is the common “50/30/20” guideline: about 50% of income for needs, 30% for wants and 20% for savings.</span>
					</label>
				</div>
			{:else if planner.savingsMode === 'specific'}
				<div class="card goal-box">
					<div class="grid cols-3">
						<label class="field">
							Saving for
							<input type="text" placeholder="e.g. Vehicle" bind:value={planner.goalName} />
						</label>
						<label class="field">
							Save per month ($)
							<input type="number" min="0" step="25" placeholder="e.g. 500" bind:value={planner.goal} />
						</label>
						<label class="field">
							Total needed ($, optional)
							<input type="number" min="0" step="500" placeholder="e.g. 6000" bind:value={planner.target} title="Used to estimate how many months reaching the goal takes." />
						</label>
					</div>
				</div>
				{#if !planner.goal}
					<div class="callout info">Enter how much you’d like to save per month, and the budget below will be built around it.</div>
				{/if}
			{/if}

			<details class="collapse">
				<summary>Budget options (cut limit, categories not to cut)</summary>
				<div class="content grid cols-2">
					<label class="field">
						<span>Largest cut allowed per category: <strong class="cut">{planner.maxCut}%</strong></span>
						<input type="range" min="0" max="60" step="5" bind:value={planner.maxCut} />
						<span class="hint">No flexible category is reduced by more than this to reach a savings target.</span>
					</label>
					<div class="field">
						Don’t cut these
						<div class="chips">
							{#each flexible as c}
								<button class="chip" class:on={planner.protected.includes(c)} onclick={() => toggleProtected(c)}>{c}</button>
							{/each}
						</div>
						<span class="hint">Protected categories are never trimmed or capped.</span>
					</div>
				</div>
			</details>
		</Section>

		<!-- ================= Budget ================= -->
		{#if budget}
			<Section
				title={budgetTitle}
				purpose="For every category: what the model predicts, the max it’s logical to spend for this income, and what we recommend budgeting."
			>
				<div class="grid {goalOn ? 'cols-4' : 'cols-3'}">
					<Metric label="Income" value={money(budget.income)} />
					<Metric
						label="Recommended spending"
						value={money(budget.spending_total)}
						delta={Math.abs(budget.change_total) >= 0.5 ? `${money(budget.change_total)} vs. predicted` : ''}
						tone="good"
					/>
					{#if goalOn}
						<Metric
							label="Savings"
							value={money(budget.savings)}
							delta={budget.goal_met ? 'target met' : `${money(budget.savings - budget.goal)} short of target`}
							tone={budget.goal_met ? 'good' : 'bad'}
						/>
					{/if}
					<Metric label="Left over" value={money(budget.leftover)} />
				</div>
				{#each budget.messages as msg}
					<div class="callout {msg.includes('above its max') || !budget.goal_met ? 'warning' : 'success'}">{msg}</div>
				{/each}

				<h3 class="sub">Bills and spending groups</h3>
				<div class="grid cols-groups">
					{#each budget.groups as g}
						{@const pctOfIncome = budget.income > 0 ? (g.recommended / budget.income) * 100 : 0}
						{@const fill = g.max > 0 ? Math.min(100, (g.recommended / g.max) * 100) : 100}
						<div class="card group" class:over={g.over} class:total={g.name === 'All bills'}>
							<div class="group-head">
								<span class="group-name">{g.name}</span>
								{#if g.over}<span class="badge bad">Over max</span>{:else}<span class="badge good">Within max</span>{/if}
							</div>
							<div class="group-value">{money(g.recommended)}</div>
							<div class="meter"><div class="meter-fill" style="width: {fill}%"></div></div>
							<div class="group-meta">
								<span>Max {money(g.max)} ({Math.round(g.share_max * 100)}% of income)</span>
								<span>{pctOfIncome.toFixed(0)}% of income</span>
							</div>
							<div class="group-cats">{g.categories.join(' · ')}</div>
						</div>
					{/each}
				</div>

				<h3 class="sub">By category</h3>
				<div class="budget-grid">
					<div class="card">
						{#if budgetChart}<Chart config={budgetChart} height={460} />{/if}
					</div>
					<div class="table-wrap">
						<table>
							<thead>
								<tr><th>Category</th><th class="num">Predicted</th><th class="num">Max</th><th class="num">Recommended</th><th>Status</th></tr>
							</thead>
							<tbody>
								{#each budget.table as r}
									<tr>
										<td class="label">{r.category}<div class="muted tiny">{r.type}</div></td>
										<td class="num">{money(r.predicted)}</td>
										<td class="num muted">{money(r.max)}</td>
										<td class="num"><strong>{money(r.recommended)}</strong></td>
										<td>
											<span class="badge {r.status === 'Over max' ? 'bad' : r.status === 'OK' ? 'neutral' : 'info'}">{r.status}</span>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>

				<details class="collapse">
					<summary>How are the max and recommended amounts calculated?</summary>
					<div class="content">
						<h3>Max logical spend</h3>
						<p>
							Each category’s max is a share of income based on common budgeting rules: the “30% rule” for
							housing and the 50/30/20 budget (about 50% needs, 30% wants, 20% savings):
						</p>
						<div class="chips" style="margin-bottom: 0.75rem">
							{#each Object.entries(app.meta?.guideline_share ?? {}) as [c, v]}
								<span class="chip static">{c} ≤ {Math.round(v * 100)}%</span>
							{/each}
						</div>
						<p>
							For flexible categories the max never drops below the least this person can realistically get by
							on: their typical month (median of the last 6 months) minus the largest allowed cut. That way
							the max fits the person’s real life, not just a rule of thumb. Group maxes: rent / housing ≤ 30%,
							other consistent bills ≤ 20%, all bills ≤ 50%, everyday needs ≤ 25%, wants ≤ 30% of income.
						</p>
						<h3>Recommended spend</h3>
						<ol>
							<li>Start from the model’s prediction.</li>
							<li><strong>Fixed bills</strong> (housing, utilities, bills, family) stay as predicted. If one is above its max it’s flagged, because it can’t be cut within a month.</li>
							<li><strong>Flexible</strong> categories above their max are trimmed down to it.</li>
							<li>
								If there’s still not enough left for the savings target, flexible categories are cut further.
								Each can shrink by at most the cut limit × how flexible it is:
								{Object.entries(app.meta?.flexibility ?? {}).map(([c, v]) => `${c} ${Math.round(v * 100)}%`).join(', ')}.
								The cuts are shared in proportion to how much each category can give.
							</li>
							<li>If that still isn’t enough, the app reports what <em>can</em> be saved.</li>
						</ol>
					</div>
				</details>
			</Section>
		{/if}

		<!-- ================= Accuracy ================= -->
		<Section
			title="How accurate is the model for this person?"
			purpose="We replay this person’s past: each month is hidden, the model predicts it from the months before, and we compare against what really happened."
		>
			{#if accuracyError}
				<div class="callout info">{accuracyError}</div>
			{:else if accuracy && modelSummary && avgSummary}
				{@const diff = modelSummary.accuracy - avgSummary.accuracy}
				<div class="grid cols-3">
					<Metric label="Model accuracy" value={pct(modelSummary.accuracy, 0)} help="100% − (all misses ÷ all actual spending). See the explanation below." />
					<Metric label="Average miss per category" value={money(modelSummary.mae)} help="On average, how many dollars one category’s prediction was off in one month." />
					<Metric
						label="Simple 3-month average"
						value={pct(avgSummary.accuracy, 0)}
						delta="{diff >= 0 ? '+' : ''}{diff.toFixed(1)} points for the model"
						tone={diff > 0 ? 'good' : 'bad'}
					/>
				</div>
				{#if diff > 0}
					<div class="callout success">Over the last {modelSummary.months} months, the model was more accurate for this person than simply averaging their recent months.</div>
				{:else}
					<div class="callout warning">Over the last {modelSummary.months} months, the simple average did as well or better for this person, so treat the prediction with care.</div>
				{/if}
				{#if planner.source === 'dataset' && inTraining}
					<div class="callout warning">
						⚠️ This person was in the <strong>training data</strong>, so the model has seen them before and the
						accuracy here is optimistic. Pick someone from the original dataset for a fair check.
					</div>
				{/if}
				<div class="card" style="margin-top: 0.75rem">
					{#if accuracyChart}<Chart config={accuracyChart} height={300} />{/if}
				</div>
				<details class="collapse">
					<summary>How is accuracy calculated?</summary>
					<div class="content">
						<AccuracyExplanation />
						<h3>Month by month (total spending)</h3>
						<div class="table-wrap">
							<table>
								<thead>
									<tr><th>Month</th><th class="num">Actual</th><th class="num">{planner.model}</th><th class="num">{SIMPLE_AVG}</th></tr>
								</thead>
								<tbody>
									{#each accuracy.per_month as m}
										<tr>
											<td>{m.month}</td>
											<td class="num">{money(m.actual)}</td>
											<td class="num">{money(m.predicted[planner.model])}</td>
											<td class="num">{money(m.predicted[SIMPLE_AVG])}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				</details>
			{/if}
		</Section>
	{:else if loading}
		<p class="muted" style="margin-top: 2rem">Predicting…</p>
	{/if}
</div>

<style>
	.edit td {
		padding: 0.25rem 0.3rem;
	}
	.cell {
		min-width: 88px;
		padding: 0.3rem 0.45rem !important;
		font-size: 0.85rem !important;
		text-align: right;
		border-color: var(--border) !important;
	}
	.cell.month {
		min-width: 92px;
		text-align: left;
	}
	.remove {
		border: none;
		background: none;
		color: var(--faint);
		font-size: 1.2rem;
		cursor: pointer;
		line-height: 1;
	}
	.remove:hover {
		color: var(--danger);
	}
	.upload-help {
		margin-top: 1rem;
	}
	.segmented {
		display: flex;
		border: 1px solid var(--border-strong);
		border-radius: 8px;
		overflow: hidden;
		background: var(--surface);
	}
	.segmented button {
		flex: 1;
		font: inherit;
		font-size: 0.88rem;
		font-weight: 500;
		padding: 0.55rem 0.5rem;
		border: none;
		border-left: 1px solid var(--border);
		background: none;
		color: var(--muted);
		cursor: pointer;
	}
	.segmented button:first-child {
		border-left: none;
	}
	.segmented button.on {
		background: var(--primary);
		color: #fff;
	}
	.hint {
		font-weight: 400;
		color: var(--faint);
		font-size: 0.8rem;
	}
	.sub {
		margin: 1.5rem 0 0.75rem;
	}
	.cols-groups {
		grid-template-columns: repeat(5, minmax(0, 1fr));
	}
	@media (max-width: 1000px) {
		.cols-groups {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	.group {
		padding: 0.9rem 1rem;
	}
	.group.total {
		border-color: var(--primary);
		background: var(--primary-soft);
	}
	.group.over {
		border-color: #ffc9c9;
	}
	.group-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.4rem;
	}
	.group-name {
		font-weight: 600;
		font-size: 0.88rem;
	}
	.group-value {
		font-size: 1.45rem;
		font-weight: 650;
		margin: 0.25rem 0 0.4rem;
		font-variant-numeric: tabular-nums;
	}
	.meter {
		height: 6px;
		border-radius: 999px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		overflow: hidden;
	}
	.meter-fill {
		height: 100%;
		background: var(--primary);
	}
	.group.over .meter-fill {
		background: var(--danger);
	}
	.group-meta {
		display: flex;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 0.25rem;
		font-size: 0.75rem;
		color: var(--muted);
		margin-top: 0.4rem;
	}
	.group-cats {
		font-size: 0.72rem;
		color: var(--faint);
		margin-top: 0.3rem;
	}
	.badge {
		font-size: 0.72rem;
		font-weight: 600;
		border-radius: 999px;
		padding: 0.1rem 0.5rem;
		white-space: nowrap;
	}
	.badge.good {
		background: var(--success-soft);
		color: var(--success);
	}
	.badge.bad {
		background: var(--danger-soft);
		color: var(--danger);
	}
	.badge.info {
		background: var(--primary-soft);
		color: var(--primary);
	}
	.badge.neutral {
		background: var(--surface-2);
		color: var(--muted);
	}
	.tiny {
		font-size: 0.72rem;
		font-weight: 400;
	}
	.chip.static {
		cursor: default;
	}
	.goal-box {
		margin-top: 1rem;
	}
	.cut {
		color: var(--primary);
	}
	.budget-grid :global(td),
	.budget-grid :global(th) {
		padding-left: 0.6rem;
		padding-right: 0.6rem;
	}
	.budget-grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 1rem;
		margin-top: 1rem;
		align-items: start;
	}
	@media (max-width: 860px) {
		.budget-grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
