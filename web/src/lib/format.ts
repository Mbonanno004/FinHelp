export const money = (x: number, digits = 0) =>
	(x < 0 ? '-$' : '$') +
	Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const signed = (x: number) => (x >= 0 ? '+' : '-') + Math.abs(Math.round(x)).toLocaleString('en-US');

export const pct = (x: number, digits = 1) => `${x.toFixed(digits)}%`;

/** Colors used for series in every chart. The first is reserved for "Actual". */
export const SERIES_COLORS = ['#1d2433', '#3b5bdb', '#f08c00', '#2f9e44', '#e03131', '#7048e8', '#0c8599', '#d6336c'];

export const ACCURACY_STEPS = [
	{
		title: 'Hide a month, then predict it.',
		text: 'For every month that has enough earlier history, the model only sees the months before it (for example the 6 months before) and predicts that month’s spending in each of the 11 categories. It never sees the month it is predicting.'
	},
	{
		title: 'Compare with what really happened.',
		text: 'For every category in every month we take the miss: |predicted − actual|.'
	},
	{
		title: 'Summarize the misses.',
		text: 'Average miss per category (MAE) = the average of all those misses, in dollars. Error rate (WAPE) = all misses added up ÷ all actual spending added up. Accuracy = 100% − error rate, so 85% accuracy means the misses add up to 15% of what was actually spent.'
	},
	{
		title: 'Compare with a simple method.',
		text: 'The same check is run on the 3-month average (next month = average of the last 3 months). A model is only useful if it beats that.'
	}
];
