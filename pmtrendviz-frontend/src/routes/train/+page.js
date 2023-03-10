import { browser } from '$app/environment';
/** @type {import('./$types').PageLoad} */
export async function load({ fetch }) {
  let res;
  res = await fetch("http://localhost:8000/models/untrained", {
    method: "GET"
  });
  const model_args = await res.json();

  res = await fetch('http://localhost:8000/train/args', {
    method: 'GET'
  });
  const general_args = await res.json();
  //console.log(general_args);
  return {
    model_args: model_args,
    general_args: general_args
  };
}
