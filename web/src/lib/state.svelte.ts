// State shared across pages, so inputs survive switching between tabs.
import { get } from '#lib/api.js';
import type { Meta, MonthRow } from '#lib/types.js';

export const app = $state({
	meta: null as Meta | null,
	metaError: ''
});

let metaRequested = false;

export function loadMeta(force = false) {
	if (metaRequested && !force) return;
	metaRequested = true;
	get<Meta>('/meta')
		.then((m) => {
			app.meta = m;
			app.metaError = '';
		})
		.catch((e) => {
			app.metaError = e.message;
			metaRequested = false;
		});
}

export const planner = $state({
	source: 'example' as 'example' | 'upload' | 'dataset',
	model: '',
	exampleRows: [] as MonthRow[],
	uploadPeople: [] as { sample_id: number; rows: MonthRow[] }[],
	uploadId: null as number | null,
	income: null as number | null,
	hasGoal: false,
	goalName: '',
	goal: null as number | null,
	target: null as number | null,
	maxCut: 30,
	protected: [] as string[]
});
