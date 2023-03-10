/** @type {import('@sveltejs/kit').HandleFetch} */
export function handleFetch({ request, fetch }) {
  if (request.url.startsWith('http://localhost:8000/')) {
    request = new Request(
      request.url.replace('http://localhost:8000/', 'http://127.0.0.1:8000/'),
      request
    );
  }

  return fetch(request);
}
