<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { app, loadMeta } from '#lib/state.svelte.js';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	const links = [
		{ href: '/', label: 'Start Here' },
		{ href: '/planner', label: 'Budget Planner' },
		{ href: '/compare', label: 'Compare Models' },
		{ href: '/train', label: 'Train Model' }
	];

	loadMeta();
</script>

<svelte:head>
	<title>FinHelp</title>
	<link rel="icon" href="/favicon.svg" />
</svelte:head>

<nav>
	<div class="inner">
		<a class="brand" href="/">
			<img src="/favicon.svg" alt="" width="26" height="26" />
			FinHelp
		</a>
		<div class="links">
			{#each links as link}
				<a href={link.href} class:active={page.url.pathname === link.href}>{link.label}</a>
			{/each}
		</div>
	</div>
</nav>

{#if app.metaError}
	<div class="page">
		<div class="callout error">
			<strong>Can't reach the Python API.</strong>
			Start it from the project folder with <code>python -m uvicorn api.main:app --port 8000</code>,
			then refresh this page. ({app.metaError})
		</div>
	</div>
{/if}

{@render children()}

<style>
	nav {
		position: sticky;
		top: 0;
		z-index: 10;
		background: rgba(255, 255, 255, 0.92);
		backdrop-filter: blur(8px);
		border-bottom: 1px solid var(--border);
	}
	.inner {
		max-width: 1180px;
		margin: 0 auto;
		padding: 0 1.5rem;
		height: 58px;
		display: flex;
		align-items: center;
		gap: 2rem;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 0.55rem;
		font-weight: 700;
		font-size: 1.1rem;
		color: var(--text);
		text-decoration: none;
	}
	.links {
		display: flex;
		gap: 0.25rem;
		overflow-x: auto;
	}
	.links a {
		color: var(--muted);
		text-decoration: none;
		font-weight: 500;
		font-size: 0.92rem;
		padding: 0.45rem 0.8rem;
		border-radius: 8px;
		white-space: nowrap;
	}
	.links a:hover {
		background: var(--surface-2);
		color: var(--text);
	}
	.links a.active {
		background: var(--primary-soft);
		color: var(--primary);
	}
	@media (max-width: 640px) {
		.inner {
			gap: 0.75rem;
			padding: 0 0.75rem;
		}
		.brand {
			font-size: 0;
		}
	}
</style>
