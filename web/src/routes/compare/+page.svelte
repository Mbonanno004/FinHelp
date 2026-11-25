<script lang="ts">
	import { onMount } from 'svelte';
	import type { ChartConfiguration } from 'chart.js';
	import { get, post } from '#lib/api.js';
	import { money, pct, SERIES_COLORS } from '#lib/format.js';
	import { app } from '#lib/state.svelte.js';
	import type { Backtest, Comparison, ModelInfo, MonthRow } from '#lib/types.js';
	import Section from '#lib/components/Section.svelte';
	import Chart from '#lib/components/Chart.svelte';
	import PersonPicker from '#lib/components/PersonPicker.svelte';
	import AccuracyExplanation from '#lib/components/AccuracyExplanation.svelte';

	type ModelsResponse = {
		models: ModelInfo[];
		untrained: { name: string; module: string }[];
		errors: { module: string; error: string }[];
	};

	let info = $state<ModelsResponse | null>(null);
	let selected = $state<string[]>([]);
	let comparison = $state<Comparison | null>(null);
	let error = $state('');
	let loading = $state(false);
	let training = $state('');
	let trainMessage = $state('');
	let categorySplit = $state<'test' | 'original'>('test');
	let personRows = $state<MonthRow[]>([]);
	let personInTraining = $state(false);
	let person = $state<Backtest | null>(null);
	let personError = $state('');

	async function loadModels() {
		try {
			info = await get<ModelsResponse>('/models');
			const names = info.models.map((m) => m.name);
			const keep = selected.filter((n) => names.includes(n));
			selected = keep.length
				? keep
				: info.models
						.filter((m) => m.kind !== 'Baseline' || ['3-month average', 'Full-history median'].includes(m.name))
						.map((m) => m.name);
		} catch (e) {
			error = (e as Error).message;
		}
	}

	onMount(loadModels);

	// Ordered like the model list, so colors stay stable.
	const ordered = $derived((info?.models ?? []).map((m) => m.name).filter((n) => selected.includes(n)));

	$effect(() => {
		const names = ordered;
		if (!names.length || !app.meta?.generated_available) {
			comparison = null;
			return;
		}
		loading = true;
		get<Comparison>(`/compare?models=${encodeURIComponent(names.join('|'))}`)
			.then((c) => {
				comparison = c;
				error = '';
			})
			.catch((e) => (error = e.message))
			.finally(() => (loading = false));
	});

	$effect(() => {
		const names = ordered;
		const rows = $state.snapshot(personRows);
		if (!names.length || !rows.length) return;
		post<Backtest>('/backtest', { models: names, rows })
			.then((b) => {
				person = b;
				personError = '';
			})
			.catch((e) => {
				person = null;
				personError = e.message;
			});
	});

	function toggle(name: string) {
		selected = selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name];
	}

	async function trainCustom(module: string, name: string) {
		training = name;
		trainMessage = '';
		try {
			await post(`/custom/${module}/train`);
			trainMessage = `Trained “${name}”.`;
			await loadModels();
			selected = [...selected, name];
		} catch (e) {
			error = (e as Error).message;
		}
		training = '';
	}

	const metric = (model: string, split: string) =>
		comparison?.overall.find((o) => o.model === model && o.split === split);

	const best = $derived.by(() => {
		const pick = (split: string, key: 'accuracy' | 'mae', high: boolean) => {
			const vals = (comparison?.overall ?? []).filter((o) => o.split === split).map((o) => o[key]);
			return vals.length ? (high ? Math.max(...vals) : Math.min(...vals)) : null;
		};
		return {
			accTest: pick('test', 'accuracy', true),
			accOrig: pick('original', 'accuracy', true),
			maeTest: pick('test', 'mae', false),
			maeOrig: pick('original', 'mae', false)
		};
	});

	const overallChart: ChartConfiguration | null = $derived.by(() => {
		if (!comparison) return null;
		const lows = comparison.overall.map((o) => o.accuracy);
		const min = Math.max(0, Math.floor(Math.min(...lows) / 5) * 5 - 5);
		return {
			type: 'bar',
			data: {
				labels: ordered,
				datasets: [
					{ label: 'Unseen simulated people', data: ordered.map((n) => metric(n, 'test')?.accuracy ?? null), backgroundColor: '#3b5bdb', borderRadius: 4 },
					{ label: 'Original dataset', data: ordered.map((n) => metric(n, 'original')?.accuracy ?? null), backgroundColor: '#a5b4fc', borderRadius: 4 }
				]
			},
			options: {
				indexAxis: 'y',
				maintainAspectRatio: false,
				plugins: {
					legend: { position: 'bottom' },
					tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.parsed.x as number)}` } }
				},
				scales: { x: { min, max: 100, ticks: { callback: (v) => `${v}%` } }, y: { grid: { display: false } } }
			}
		};
	});

	const personChart: ChartConfiguration | null = $derived(
		person && {
			type: 'line',
			data: {
				labels: person.per_month.map((m) => m.month),
				datasets: [
					{ label: 'Actual', data: person.per_month.map((m) => m.actual), borderColor: SERIES_COLORS[0], backgroundColor: SERIES_COLORS[0], borderWidth: 3, tension: 0 },
					...ordered.map((name, i) => ({
						label: name,
						data: person!.per_month.map((m) => m.predicted[name]),
						borderColor: SERIES_COLORS[(i % 7) + 1],
						backgroundColor: SERIES_COLORS[(i % 7) + 1],
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

	const catRows = $derived(comparison?.categories[categorySplit] ?? null);
</script>

<div class="page">
	<h1>Compare Models</h1>
	<div class="callout info">
		See how the main neural network stacks up against <strong>other models</strong> (like ones you build
		yourselves) and <strong>simple baselines</strong>. Every model is graded on the same people that none of
		them trained on, so the comparison is fair.
	</div>
	{#if error}<div class="callout error">{error}</div>{/if}
	{#if app.meta && !app.meta.generated_available}
		<div class="callout warning">
			The comparison needs the generated test people. Create them on the <a href="/train">Train Model</a> page.
		</div>
	{/if}

	<Section title="Models available" purpose="Everything that can be compared. Your own models show up here automatically. Click a model to add or remove it from the comparison.">
		<div class="table-wrap">
			<table>
				<thead>
					<tr><th>Compare</th><th>Model</th><th>Type</th><th class="num">Months of history</th><th>How it works</th></tr>
				</thead>
				<tbody>
					{#each info?.models ?? [] as m}
						<tr class="clickable" onclick={() => toggle(m.name)}>
							<td><input type="checkbox" checked={selected.includes(m.name)} onclick={(e) => e.stopPropagation()} onchange={() => toggle(m.name)} /></td>
							<td class="label">{m.name}</td>
							<td><span class="tag {m.kind === 'Baseline' ? 'base' : m.kind === 'Custom' ? 'custom' : 'nn'}">{m.kind}</span></td>
							<td class="num">{m.history}</td>
							<td class="muted">{m.description}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		{#each info?.untrained ?? [] as u}
			<div class="callout warning row between">
				<span><strong>{u.name}</strong> is not trained yet.</span>
				<button class="btn small primary" disabled={!!training} onclick={() => trainCustom(u.module, u.name)}>
					{training === u.name ? 'Training…' : 'Train'}
				</button>
			</div>
		{/each}
		{#if trainMessage}<div class="callout success">{trainMessage}</div>{/if}
		{#each info?.errors ?? [] as e}
			<div class="callout error">Could not load <code>src/custom_models/{e.module}.py</code>: {e.error}</div>
		{/each}

		<details class="collapse">
			<summary>How do I add my own model?</summary>
			<div class="content">
				<ol>
					<li>Copy <code>src/custom_models/_template.py</code> to a new file <strong>without</strong> the leading underscore, e.g. <code>src/custom_models/small_nn.py</code>.</li>
					<li>Fill in <code>NAME</code>, <code>DESCRIPTION</code>, <code>HISTORY</code>, and the <code>train()</code> and <code>load()</code> functions. Any method works (PyTorch, NumPy, scikit-learn…) as long as <code>load()</code> returns a function that maps the inputs to predicted dollars for the 11 categories.</li>
					<li>Train only on <code>splits["train"]</code> (or part of it), never on the test people.</li>
					<li>Train it with <code>python -m src.custom_models.small_nn</code> or the <strong>Train</strong> button that appears here, then refresh this page.</li>
				</ol>
				<p class="muted small">See <code>src/custom_models/linear_regression.py</code> for a complete working example.</p>
			</div>
		</details>
	</Section>

	{#if comparison}
		<Section
			title="Overall accuracy"
			purpose="Scored on {comparison.test_predictions.toLocaleString()} predictions for {comparison.test_people} unseen simulated people, and on the original 10-person dataset."
		>
			<div class="card">
				{#if overallChart}<Chart config={overallChart} height={Math.max(180, 70 * ordered.length)} />{/if}
			</div>
			<div class="table-wrap" style="margin-top: 1rem">
				<table>
					<thead>
						<tr>
							<th>Model</th>
							<th class="num">Accuracy · unseen people</th>
							<th class="num">Accuracy · original dataset</th>
							<th class="num">Avg miss / category · unseen</th>
							<th class="num">Avg miss / category · original</th>
						</tr>
					</thead>
					<tbody>
						{#each ordered as name}
							{@const t = metric(name, 'test')}
							{@const o = metric(name, 'original')}
							<tr>
								<td class="label">{name}</td>
								<td class="num" class:best={t?.accuracy === best.accTest}>{t ? pct(t.accuracy) : '–'}</td>
								<td class="num" class:best={o?.accuracy === best.accOrig}>{o ? pct(o.accuracy) : '–'}</td>
								<td class="num" class:best={t?.mae === best.maeTest}>{t ? money(t.mae, 2) : '–'}</td>
								<td class="num" class:best={o?.mae === best.maeOrig}>{o ? money(o.mae, 2) : '–'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="muted small" style="margin-top: 0.5rem">Green = best in each column. Accuracy: higher is better. Misses: lower is better.</p>
			<details class="collapse">
				<summary>How is accuracy calculated?</summary>
				<div class="content"><AccuracyExplanation /></div>
			</details>
		</Section>

		<Section title="Accuracy by category" purpose="Average dollars off per month in each category. This shows where each model wins or loses.">
			<div class="chips" style="margin-bottom: 0.75rem">
				<button class="chip" class:on={categorySplit === 'test'} onclick={() => (categorySplit = 'test')}>Unseen simulated people</button>
				<button class="chip" class:on={categorySplit === 'original'} onclick={() => (categorySplit = 'original')}>Original dataset</button>
			</div>
			{#if catRows}
				<div class="table-wrap">
					<table>
						<thead>
							<tr><th>Category</th><th class="num">Avg actual</th>{#each ordered as n}<th class="num">{n}</th>{/each}</tr>
						</thead>
						<tbody>
							{#each catRows as r}
								{@const low = Math.min(...ordered.map((n) => r.errors[n]))}
								<tr>
									<td class="label">{r.category}</td>
									<td class="num muted">{money(r.avg_actual)}</td>
									{#each ordered as n}<td class="num" class:best={r.errors[n] === low}>{money(r.errors[n], 1)}</td>{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				<p class="muted small" style="margin-top: 0.5rem">Green = the model closest to the truth in that category.</p>
			{:else}
				<div class="callout info">A selected model needs more history than the original dataset’s 12 months allow.</div>
			{/if}
		</Section>

		<Section title="Accuracy for one person" purpose="Replay one person’s past months with every selected model.">
			<PersonPicker
				onload={(rows, t) => {
					personRows = rows;
					personInTraining = t;
				}}
				onerror={(m) => (personError = m)}
			/>
			{#if personError}
				<div class="callout info">{personError}</div>
			{:else if person}
				{#if personInTraining}
					<div class="callout warning">⚠️ This person was in the training data, so the main model’s accuracy here is optimistic.</div>
				{/if}
				<div class="table-wrap" style="margin-top: 1rem">
					<table>
						<thead>
							<tr><th>Model</th><th class="num">Accuracy</th><th class="num">Avg miss / category</th><th class="num">Avg miss / month total</th><th class="num">Months checked</th></tr>
						</thead>
						<tbody>
							{#each person.summary as s}
								<tr>
									<td class="label">{s.model}</td>
									<td class="num" class:best={s.accuracy === Math.max(...person.summary.map((x) => x.accuracy))}>{pct(s.accuracy)}</td>
									<td class="num">{money(s.mae, 2)}</td>
									<td class="num">{money(s.total_mae)}</td>
									<td class="num">{s.months}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				<div class="card" style="margin-top: 1rem">
					{#if personChart}<Chart config={personChart} height={320} />{/if}
				</div>
			{/if}
		</Section>
	{:else if loading}
		<p class="muted" style="margin-top: 2rem">Scoring models…</p>
	{/if}
</div>

<style>
	.clickable {
		cursor: pointer;
	}
	.tag {
		font-size: 0.75rem;
		font-weight: 600;
		border-radius: 999px;
		padding: 0.12rem 0.55rem;
		white-space: nowrap;
	}
	.tag.nn {
		background: var(--primary-soft);
		color: var(--primary);
	}
	.tag.custom {
		background: #f3f0ff;
		color: #6741d9;
	}
	.tag.base {
		background: var(--surface-2);
		color: var(--muted);
		border: 1px solid var(--border);
	}
	.between {
		justify-content: space-between;
	}
	input[type='checkbox'] {
		accent-color: var(--primary);
		width: 16px;
		height: 16px;
	}
</style>
