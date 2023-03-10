import { writable } from "svelte/store";


export const graph_data = writable([{
  x: [],
  y: [],
  name: ''
}]);
