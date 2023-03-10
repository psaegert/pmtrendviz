<script lang="ts">
    import { createEventDispatcher} from "svelte";

    let timeout: ReturnType<typeof setTimeout>;
    export let delay = 300;
    export let searchText: string;
    export let defaultText: string;
    const dispatch = createEventDispatcher();

    function keyPressed() {
        if (timeout) {
            clearTimeout(timeout);
        }
        timeout = setTimeout(() => {
            reload_search();
            console.log(searchText);
        }, delay);
    }

    function reload_search() {
        dispatch("reloadsearch", searchText);
        console.log("reload_search");
    }
</script>

<div class="w-1/5">
    <div class="input-group relative flex w-full">
        <input bind:value={searchText} placeholder={defaultText} on:keydown={()=> keyPressed()} type="text" class="form-control relative flex-auto block w-full px-3 py-3 text-base font-normal text-gray-700 bg-white bg-clip-padding border border-solid border-gray-300 rounded transition ease-in-out m-0 focus:text-gray-700 focus:bg-white focus:border-gray-400 focus:outline-none">
        <button style="background-color: var(--primary-color); color: #ffffff" class="absolute top-2 right-2 btn flex p-2 text-gray-700 font-medium text-xs leading-tight uppercase rounded focus:outline-none focus:ring-0 transition duration-150 ease-in-out items-center" type="button">
            Search
        </button>
    </div>
</div>