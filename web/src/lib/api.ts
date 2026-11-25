// Small helpers for calling the Python API. Every endpoint lives under /api.

export class ApiError extends Error {}

async function handle<T>(res: Response): Promise<T> {
	if (!res.ok) {
		let message = `Request failed (${res.status})`;
		try {
			const body = await res.json();
			if (typeof body.detail === 'string') message = body.detail;
			else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join('; ');
		} catch {
			/* not JSON */
		}
		if (res.status === 502 || res.status === 504) {
			message = 'Cannot reach the Python API. Is it running? (see README: "Running the app")';
		}
		throw new ApiError(message);
	}
	return res.json() as Promise<T>;
}

export async function get<T>(path: string): Promise<T> {
	try {
		return await handle<T>(await fetch(`/api${path}`));
	} catch (e) {
		throw e instanceof ApiError ? e : new ApiError('Cannot reach the Python API. Is it running?');
	}
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
	try {
		return await handle<T>(
			await fetch(`/api${path}`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: body === undefined ? undefined : JSON.stringify(body)
			})
		);
	} catch (e) {
		throw e instanceof ApiError ? e : new ApiError('Cannot reach the Python API. Is it running?');
	}
}

export async function upload<T>(path: string, file: File): Promise<T> {
	const form = new FormData();
	form.append('file', file);
	return handle<T>(await fetch(`/api${path}`, { method: 'POST', body: form }));
}

/** POST and read a newline-delimited JSON stream, calling onEvent for each line. */
export async function stream(path: string, body: unknown, onEvent: (event: any) => void) {
	const res = await fetch(`/api${path}`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok || !res.body) await handle(res);
	const reader = res.body!.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });
		let newline;
		while ((newline = buffer.indexOf('\n')) >= 0) {
			const line = buffer.slice(0, newline).trim();
			buffer = buffer.slice(newline + 1);
			if (line) onEvent(JSON.parse(line));
		}
	}
}
