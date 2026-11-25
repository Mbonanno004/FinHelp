<script lang="ts">
	import Chart from 'chart.js/auto';
	import type { ChartConfiguration } from 'chart.js';

	let { config, height = 320 }: { config: ChartConfiguration; height?: number } = $props();

	let canvas: HTMLCanvasElement;
	let chart: Chart | undefined;

	Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
	Chart.defaults.color = '#5d6679';

	$effect(() => {
		const cfg = $state.snapshot(config) as ChartConfiguration;
		if (!chart) {
			chart = new Chart(canvas, cfg);
		} else {
			chart.data = cfg.data;
			chart.options = cfg.options ?? {};
			chart.update('none');
		}
	});

	$effect(() => () => {
		chart?.destroy();
		chart = undefined;
	});
</script>

<div class="chart" style="height: {height}px">
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.chart {
		position: relative;
		width: 100%;
	}
</style>
