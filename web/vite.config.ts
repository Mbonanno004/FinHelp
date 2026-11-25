import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// Builds a single-page app into web/build. The Python API serves it in production
			// mode, so the finished app runs from one command (see README).
			adapter: adapter({ fallback: 'index.html' })
		})
	],
	server: {
		port: 5173,
		// During development, forward every /api request to the Python API.
		// Set FINHELP_API to use an API on a different address/port.
		proxy: { '/api': process.env.FINHELP_API ?? 'http://127.0.0.1:8000' }
	}
});
