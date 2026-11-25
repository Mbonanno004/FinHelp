<script lang="ts">
	import type { ChartConfiguration } from 'chart.js';
	import { post, stream } from '#lib/api.js';
	import { money, pct } from '#lib/format.js';
	import { app, loadMeta } from '#lib/state.svelte.js';
	import type { TrainConfig } from '#lib/types.js';
	import Section from '#lib/components/Section.svelte';
	import Chart from '#lib/components/Chart.svelte';

	type Result = { model: string; kind: string; test: number; original: number | null };
	type Done = { run_id: string; seconds: number; epochs_run: number; best_val_mae: number; results: Result[] };

	const HIDDEN = [16, 32, 64, 128, 256, 512];
	const LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01];

	let config = $state<TrainConfig | null>(null);
	$effect(() => {
		if (!config && app.meta) config = { ...app.meta.train_defaults };
	});

	let running = $state(false);
	let epochs = $state<number[]>([]);
	let valMae = $state<number[]>([]);
	let done = $state<Done | null>(null);
	let error = $state('');
	let saved = $state('');

	// Generating data
	let gen = $state({ people: 500, months: 24, seed: 42 });
	let generating = $state(false);
	let genMessage = $state('');

	async function generate() {
		generating = true;
		error = '';
		try {
			genMessage = (await post<{ message: string }>('/generate', gen)).message;
			loadMeta(true);
		} catch (e) {
			error = (e as Error).message;
		}
		generating = false;
	}

	async function runTraining() {
		if (!config) return;
		running = true;
		epochs = [];
		valMae = [];
		done = null;
		error = '';
		saved = '';
		try {
			await stream('/train', $state.snapshot(config), (event) => {
				if (event.type === 'epoch') {
					epochs.push(event.epoch);
					valMae.push(event.val_mae);
				} else if (event.type === 'done') {
					done = event;
				} else if (event.type === 'error') {
					error = event.message;
				}
			});
		} catch (e) {
			error = (e as Error).message;
		}
		running = false;
	}

	async function save() {
		if (!done) return;
		try {
			saved = (await post<{ message: string }>(`/train/${done.run_id}/save`)).message;
		} catch (e) {
			error = (e as Error).message;
		}
	}

	const progress = $derived(config && epochs.length ? epochs.at(-1)! / config.epochs : 0);

	const lossChart: ChartConfiguration | null = $derived(
		epochs.length && config
			? {
					type: 'line',
					data: {
						labels: [...epochs],
						datasets: [
							{ label: 'Validation miss ($)', data: [...valMae], borderColor: '#3b5bdb', backgroundColor: '#3b5bdb', pointRadius: 0, borderWidth: 2, tension: 0.2 }
						]
					},
					options: {
						maintainAspectRatio: false,
						animation: false,
						plugins: {
							legend: { display: false },
							tooltip: { callbacks: { title: (c) => `Pass ${c[0].label}`, label: (c) => money(c.parsed.y as number, 2) } }
						},
						scales: {
							x: { type: 'linear', min: 1, max: config.epochs, title: { display: true, text: 'Pass (epoch)' } },
							y: { title: { display: true, text: 'Average miss ($)' }, ticks: { callback: (v) => money(Number(v)) } }
						}
					}
				}
			: null
	);

	const bestTest = $derived(done ? Math.max(...done.results.map((r) => r.test)) : null);
	const bestOrig = $derived(done ? Math.max(...done.results.map((r) => r.original ?? 0)) : null);
</script>

<div class="page">
	<h1>Train Model</h1>
	<div class="callout info">
		Experiment with the main <strong>neural network</strong>. Change its settings, train it (about 30–60
		seconds), and see whether it beats the current version and the simple baselines. Nothing else in the app
		changes unless you click <strong>Save</strong>.
	</div>
	{#if error}<div class="callout error">{error}</div>{/if}

	{#if app.meta && !app.meta.generated_available}
		<Section title="Create the training data" purpose="Training needs the generated dataset (data/generated_transactions.csv). It takes a few seconds.">
			<div class="card">
				<div class="grid cols-3">
					<label class="field">People <input type="number" min="50" max="5000" step="50" bind:value={gen.people} /></label>
					<label class="field">Months per person <input type="number" min="12" max="60" bind:value={gen.months} /></label>
					<label class="field">Seed <input type="number" bind:value={gen.seed} /></label>
				</div>
				<button class="btn primary" style="margin-top: 1rem" disabled={generating} onclick={generate}>
					{generating ? 'Generating…' : 'Generate training data'}
				</button>
			</div>
		</Section>
	{:else if config}
		{#if genMessage}<div class="callout success">{genMessage}</div>{/if}

		<Section title="Settings" purpose="Each setting changes how the network learns. The grey text under each one explains it in plain English.">
			<div class="card">
				<div class="grid cols-4">
					<label class="field">
						Months of history: <strong>{config.history}</strong>
						<input type="range" min="3" max="9" bind:value={config.history} />
						<span class="hint">How many past months the model looks at to predict the next one.</span>
					</label>
					<label class="field">
						Neurons per layer: <strong>{config.hidden}</strong>
						<input type="range" min="0" max={HIDDEN.length - 1} value={HIDDEN.indexOf(config.hidden)}
							oninput={(e) => (config!.hidden = HIDDEN[+(e.target as HTMLInputElement).value])} />
						<span class="hint">How much “brain power” each layer has. More can learn more, but may memorize.</span>
					</label>
					<label class="field">
						Hidden layers: <strong>{config.layers}</strong>
						<input type="range" min="1" max="4" bind:value={config.layers} />
						<span class="hint">How many layers of pattern-finding are stacked on top of each other.</span>
					</label>
					<label class="field">
						Dropout: <strong>{config.dropout.toFixed(2)}</strong>
						<input type="range" min="0" max="0.5" step="0.05" bind:value={config.dropout} />
						<span class="hint">Share of neurons switched off while practicing, so it can’t just memorize.</span>
					</label>
					<label class="field">
						Learning rate: <strong>{config.lr}</strong>
						<input type="range" min="0" max={LRS.length - 1} value={LRS.indexOf(config.lr)}
							oninput={(e) => (config!.lr = LRS[+(e.target as HTMLInputElement).value])} />
						<span class="hint">Step size for each correction. Too big overshoots; too small is slow.</span>
					</label>
					<label class="field">
						Max epochs: <strong>{config.epochs}</strong>
						<input type="range" min="10" max="400" step="10" bind:value={config.epochs} />
						<span class="hint">Maximum passes through the training data. Stops early if it stops improving.</span>
					</label>
					<label class="field">
						Loss function
						<select bind:value={config.loss}>
							<option value="l1">l1: average miss</option>
							<option value="huber">huber: in between</option>
							<option value="mse">mse: squared miss</option>
						</select>
						<span class="hint">How mistakes are scored while learning. l1 predicts the typical month.</span>
					</label>
					<label class="field">
						Random seed
						<input type="number" bind:value={config.seed} />
						<span class="hint">Changes the random starting point. The test people stay the same.</span>
					</label>
				</div>
				<div class="row" style="margin-top: 1.25rem">
					<button class="btn primary" disabled={running} onclick={runTraining}>{running ? 'Training…' : 'Train'}</button>
					<button class="btn" disabled={running} onclick={() => (config = { ...app.meta!.train_defaults })}>Reset to defaults</button>
				</div>
			</div>
		</Section>

		{#if epochs.length || running}
			<Section title="Training progress" purpose="The line shows the average miss on 75 validation people after each pass. Down is good; a flat line means it has stopped improving.">
				<div class="card">
					<div class="progress"><div class="bar" style="width: {done ? 100 : progress * 100}%"></div></div>
					<p class="muted small status">
						{#if done}
							Done in {done.seconds.toFixed(0)}s after {done.epochs_run} passes. Best validation miss {money(done.best_val_mae, 2)}.
						{:else if epochs.length}
							Pass {epochs.at(-1)} of {config.epochs}: validation miss {money(valMae.at(-1)!, 2)}
						{:else}
							Preparing data…
						{/if}
					</p>
					{#if lossChart}<Chart config={lossChart} height={280} />{/if}
				</div>
			</Section>
		{/if}

		{#if done}
			<Section title="Results" purpose="The new model next to the version the app uses now and the baselines, on people none of them trained on.">
				<div class="table-wrap">
					<table>
						<thead>
							<tr><th>Model</th><th class="num">Accuracy · unseen simulated people</th><th class="num">Accuracy · original dataset</th></tr>
						</thead>
						<tbody>
							{#each done.results as r}
								<tr>
									<td class="label">{r.model}</td>
									<td class="num" class:best={r.test === bestTest}>{pct(r.test)}</td>
									<td class="num" class:best={r.original === bestOrig}>{r.original === null ? '–' : pct(r.original)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				<div class="row" style="margin-top: 1rem">
					<button class="btn primary" disabled={!!saved} onclick={save}>Save as the app’s model</button>
					<span class="muted small">Replaces <code>models/spending_model.pt</code>. Commit it to share with your teammate.</span>
				</div>
				{#if saved}<div class="callout success">{saved}</div>{/if}
			</Section>
		{/if}
	{/if}
</div>

<style>
	.hint {
		font-weight: 400;
		color: var(--faint);
		font-size: 0.78rem;
		line-height: 1.35;
	}
	.field strong {
		color: var(--text);
	}
	.progress {
		height: 8px;
		border-radius: 999px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		overflow: hidden;
	}
	.bar {
		height: 100%;
		background: var(--primary);
		transition: width 0.2s;
	}
	.status {
		margin: 0.6rem 0 0.75rem;
	}
</style>
