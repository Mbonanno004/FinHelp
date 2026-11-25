// Shapes of the JSON the Python API sends back (see api/main.py).

export type MonthRow = { month: string; Income: number; [category: string]: number | string };

export type ModelInfo = { name: string; kind: 'Neural network' | 'Custom' | 'Baseline'; history: number; description: string };

export type Meta = {
	categories: string[];
	fixed_categories: string[];
	flexibility: Record<string, number>;
	guideline_share: Record<string, number>;
	groups: { name: string; categories: string[]; share_max: number }[];
	train_defaults: TrainConfig;
	generated_available: boolean;
};

export type TrainConfig = {
	history: number;
	hidden: number;
	layers: number;
	dropout: number;
	lr: number;
	epochs: number;
	loss: string;
	seed: number;
};

export type Forecast = {
	month: string;
	categories: { category: string; predicted: number; recent_average: number }[];
	predicted_total: number;
	recent_average_total: number;
	recent_income: number;
	typical: Record<string, number>;
};

export type Budget = {
	table: {
		category: string;
		type: string;
		predicted: number;
		max: number;
		recommended: number;
		change: number;
		status: 'OK' | 'Over max' | 'Capped at max' | 'Cut to save';
	}[];
	groups: BudgetGroup[];
	income: number;
	goal: number;
	savings: number;
	leftover: number;
	goal_met: boolean;
	messages: string[];
	months_to_target: number | null;
	spending_total: number;
	change_total: number;
};

export type Metrics = { accuracy: number; mae: number; total_mae: number; wape: number };

export type Backtest = {
	per_month: { month: string; actual: number; predicted: Record<string, number> }[];
	summary: ({ model: string; kind: string; months: number } & Metrics)[];
};

export type Comparison = {
	test_people: number;
	test_predictions: number;
	overall: ({ model: string; kind: string; split: 'test' | 'original' } & Metrics)[];
	categories: {
		test: CategoryErrors[] | null;
		original: CategoryErrors[] | null;
	};
};

export type CategoryErrors = { category: string; avg_actual: number; errors: Record<string, number> };

export type Dataset = { key: string; label: string; people: number[] };

export type BudgetGroup = {
	name: string;
	categories: string[];
	predicted: number;
	recommended: number;
	max: number;
	share_max: number;
	share_recommended: number | null;
	over: boolean;
};
