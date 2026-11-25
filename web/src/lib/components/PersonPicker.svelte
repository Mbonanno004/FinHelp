<script lang="ts">
	import { get } from '#lib/api.js';
	import type { Dataset, MonthRow } from '#lib/types.js';

	let {
		onload,
		onerror
	}: {
		onload: (rows: MonthRow[], inTraining: boolean) => void;
		onerror?: (message: string) => void;
	} = $props();

	let datasets: Dataset[] = $state([]);
	let datasetKey = $state('original');
	let person = $state<number | null>(null);

	const current = $derived(datasets.find((d) => d.key === datasetKey));

	$effect(() => {
		get<{ datasets: Dataset[] }>('/datasets')
			.then((r) => {
				datasets = r.datasets;
				person = r.datasets[0]?.people[0] ?? null;
			})
			.catch((e) => onerror?.(e.message));
	});

	function changeDataset() {
		person = current?.people[0] ?? null;
	}

	$effect(() => {
		if (person === null) return;
		const key = datasetKey;
		const id = person;
		get<{ rows: MonthRow[]; in_training: boolean }>(`/datasets/${key}/${id}`)
			.then((r) => onload(r.rows, r.in_training))
			.catch((e) => onerror?.(e.message));
	});
</script>

<div class="grid cols-2">
	<label class="field">
		Dataset
		<select bind:value={datasetKey} onchange={changeDataset}>
			{#each datasets as d}<option value={d.key}>{d.label}</option>{/each}
		</select>
	</label>
	<label class="field">
		Person (sample_id)
		<select bind:value={person}>
			{#each current?.people ?? [] as id}<option value={id}>{id}</option>{/each}
		</select>
	</label>
</div>
