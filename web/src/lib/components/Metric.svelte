<script lang="ts">
	let {
		label,
		value,
		delta = '',
		tone = 'neutral',
		help = ''
	}: {
		label: string;
		value: string;
		delta?: string;
		tone?: 'good' | 'bad' | 'neutral';
		help?: string;
	} = $props();
</script>

<div class="card metric">
	<div class="label">
		{label}
		{#if help}<span class="help" title={help}>?</span>{/if}
	</div>
	<div class="value">{value}</div>
	{#if delta}<div class="delta {tone}">{delta}</div>{/if}
</div>

<style>
	.metric {
		padding: 1rem 1.1rem;
	}
	.label {
		font-size: 0.82rem;
		color: var(--muted);
		font-weight: 500;
		display: flex;
		align-items: center;
		gap: 0.35rem;
	}
	.value {
		font-size: 1.85rem;
		font-weight: 650;
		letter-spacing: -0.02em;
		font-variant-numeric: tabular-nums;
		margin-top: 0.15rem;
	}
	.delta {
		display: inline-block;
		margin-top: 0.3rem;
		font-size: 0.8rem;
		font-weight: 600;
		border-radius: 999px;
		padding: 0.1rem 0.55rem;
		background: var(--surface-2);
		color: var(--muted);
	}
	.delta.good {
		background: var(--success-soft);
		color: var(--success);
	}
	.delta.bad {
		background: var(--danger-soft);
		color: var(--danger);
	}
	.help {
		display: inline-grid;
		place-items: center;
		width: 15px;
		height: 15px;
		border-radius: 50%;
		border: 1px solid var(--faint);
		color: var(--faint);
		font-size: 0.65rem;
		cursor: help;
	}
</style>
