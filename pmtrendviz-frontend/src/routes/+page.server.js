import { redirect } from '@sveltejs/kit';

/** @type {import('../../.svelte-kit/types/src/routes').LayoutServerLoad} */
export function load() {
    throw redirect(307, '/predict');
}