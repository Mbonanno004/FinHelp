<script lang="ts">
	let { checked = $bindable(false), label }: { checked?: boolean; label: string } = $props();
</script>

<label class="toggle">
	<input type="checkbox" role="switch" bind:checked />
	<span class="track"><span class="thumb"></span></span>
	<span class="text">{label}</span>
</label>

<style>
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 0.6rem;
		cursor: pointer;
		font-weight: 500;
		user-select: none;
	}
	input {
		position: absolute;
		opacity: 0;
		width: 1px;
		height: 1px;
	}
	.track {
		width: 40px;
		height: 22px;
		border-radius: 999px;
		background: var(--border-strong);
		position: relative;
		transition: background 0.15s;
	}
	.thumb {
		position: absolute;
		top: 3px;
		left: 3px;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #fff;
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
		transition: transform 0.15s;
	}
	input:checked + .track {
		background: var(--primary);
	}
	input:checked + .track .thumb {
		transform: translateX(18px);
	}
	input:focus-visible + .track {
		box-shadow: 0 0 0 3px rgba(59, 91, 219, 0.25);
	}
</style>
