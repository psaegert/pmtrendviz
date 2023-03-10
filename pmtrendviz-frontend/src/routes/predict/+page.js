import { browser } from '$app/environment';
/** @type {import('./$types').PageLoad} */
export async function load({ fetch }) {
    const res = await fetch('http://localhost:8000/models/saved', {
        method: 'GET'
    });
    const models = await res.json();
    return {
        models: models
    };
}
