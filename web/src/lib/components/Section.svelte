<script lang="ts">
	import type { Snippet } from 'svelte';

	let { title, purpose, children }: { title: string; purpose?: string; children: Snippet } = $props();
</script>

<section>
	<header>
		<h2>{title}</h2>
		{#if purpose}<p class="muted small">{purpose}</p>{/if}
	</header>
	{@render children()}
</section>

<style>
	section {
		margin-top: 2.25rem;
	}
	header {
		margin-bottom: 0.9rem;
	}
	header p {
		margin: 0.2rem 0 0;
	}
</style>
