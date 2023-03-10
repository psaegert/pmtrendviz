<script lang="ts">
    import { onMount } from 'svelte';
    import Searchbar from "../../components/Searchbar.svelte";
    import { graph_data} from "../../components/GraphdataStorejs";
    import Nav from "../../components/nav.svelte";
    import Plotly, { Layout, Data } from "plotly.js-dist-min";

    let headerText;

    export let plotHeader = '';

    export let data: Data;

    let plotDiv: HTMLElement;

    let data_plot: [{ x: [], name: string, y: [] }] = [];

    var layout: Partial<Layout> = {
        showlegend: true,
        legend: {
            orientation: "h",
            y: -.4,
            font: {
                family: 'sans-serif',
                size: 12,
                color: '#000'
            },
            bgcolor: '#E2E2E2',
            bordercolor: '#FFFFFF',
            borderwidth: 2
        },
        xaxis: {
            title: {
                text: 'Time',
                font: {
                    family: 'Helvetica, monospace',
                    size: 14,
                    color: '#7f7f7f'
                }
            }
        },
        yaxis: {
            title: {
                text: 'Number of Articles',
                font: {
                    family: 'Helvetica, monospace',
                    size: 14,
                    color: '#7f7f7f'
                }
            }
        }
    }

    onMount(async () => {
        headerText = 'On Mount Called !';
        new Plotly.newPlot(plotDiv, data_plot, layout,  {staticPlot: true});

        graph_data.subscribe(value => {
            console.log(value);
            Plotly.newPlot(plotDiv, value,  layout,  {staticPlot: true});
            console.log('changing data')
        });
    });


    let delay = 500;
    let currentModel: string;
    let searchTerm :string;
    let resolution: string;
    let distance: string;
    let ignoreTopNDates: number = 20;
    let num_clusters: number = 5;

    async function updateGraph() {
        console.log(currentModel);
        console.log(JSON.stringify({
            model_name: currentModel,
            query: searchTerm,
            resolution: resolution,
            distance: distance,
            num_clusters: num_clusters,
            ignoreTopNDates: ignoreTopNDates
        }))
        const res = await fetch ('http://localhost:8000/predict/', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: currentModel,
                query: searchTerm,
                resolution: resolution,
                distance: distance,
                num_clusters: num_clusters,
                ignore_top_n_dates: ignoreTopNDates
            })
        });
        const d = await res.json();
        graph_data.set(d)
    }
</script>

<style>
    .wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }

    #plotDiv {
        width: 100%;
        height: 100%;
    }

    .slider {
        -webkit-appearance: none;
        width: 100%;
        height: 10px;
        background: #f9f9f9;
        outline: none;
        opacity: 0.7;
        -webkit-transition: .2s;
        transition: opacity .2s;
    }

    .slider:hover {
        opacity: 1;
    }

    .slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 10px;
        height: 10px;
        background: var(--primary-color);
        cursor: pointer;
    }

    .slider::-moz-range-thumb {
        width: 10px;
        height: 10px;
        background: var(--primary-color);
        cursor: pointer;
    }

    .sliderticks {
         display: flex;
         justify-content: space-between;
         padding: 0 10px;
     }

    .sliderticks p {
        position: relative;
        display: flex;
        justify-content: center;
        text-align: center;
        width: 1px;
        background: #D3D3D3;
        height: 10px;
        line-height: 40px;
        margin: 0 0 20px 0;
    }

    :root {
        --primary-color: #011526;
        --secondary-color: #c9dff2;
        --accent-color: #d98f4e;
    }

    h1 {
        text-align: center;
        font-family: Helvetica, sans-serif;
        margin-top: 50px;
    }

    h2 {
        text-align: left;
        font-family: Helvetica, sans-serif;
    }
</style>


<Nav/>
<div style="padding: 50px">
<h1>Pubmed Research Trends</h1>



<Searchbar bind:searchText={searchTerm} delay={delay} defaultText="Search for topics..." on:reloadsearch={updateGraph}/>

    <div>
        <h1>{plotHeader}</h1>
    </div>


<h2>Options</h2>
<div style="margin: 55px auto 0 auto; display: block">
    <div class="row mb-3">
        <label for="modelSelect" class="col-sm-2 col-form-label" >Model</label>
        <select class="col-sm-2" id="modelSelect" bind:value={currentModel} on:change={updateGraph}>
            {#each data.models as model}
                <option>{model.name}</option>
            {/each}
        </select>
    </div>

    <div class="row mb-3">
        <label for="resolutionSelect" class="col-sm-2 col-form-label">Resolution</label>
        <select id="resolutionSelect" class="col-sm-2" bind:value={resolution} on:change={updateGraph}>
            <option value="day">Day</option>
            <option value="week">Week</option>
            <option value="month">Month</option>
            <option value="quarter">Quarter</option>
            <option value="year">Year</option>
        </select>
    </div>

    <div class="row mb-3">
        <label for="distanceSelect" class="col-sm-2 col-form-label">Distance (for TF-IDF-based models)</label>
        <select id="distanceSelect" class="col-sm-2" bind:value={distance} on:change={updateGraph}>
            <option value="cosine">Cosine</option>
            <option value="euclidean">Euclidian</option>
        </select>
    </div>

    <div class="row mb-3">
        <label for="ignoreTopNDatesSlider" class="col-sm-2 col-form-label">Remove spikes</label>
        <div class="col-sm-3">

            <input id="ignoreTopNDatesSlider" bind:value={ignoreTopNDates} type="range" min="0" max="30" class="slider" on:change={updateGraph}/>
            <div class="sliderticks">
                <p>0</p>
                <p>5</p>
                <p>10</p>
                <p>15</p>
                <p>20</p>
                <p>25</p>
                <p>30</p>
            </div>
        </div>
    </div>

    <div class="row mb-3">
        <label for="noClusterInput" class="col-sm-2 col-form-label">Number of clusters</label>
        <div class="col-sm-3">

            <input id="noClusterInput" bind:value={num_clusters} type="range" min="1" max="10" class="slider" on:change={updateGraph}/>
            <div class="sliderticks">
                <p>0</p>
                <p>2</p>
                <p>3</p>
                <p>4</p>
                <p>5</p>
                <p>6</p>
                <p>7</p>
                <p>8</p>
                <p>9</p>
                <p>10</p>
            </div>
        </div>
    </div>
</div>

<diV>
    <div class="wrapper">
        <div style="width:95%;" id="plotDiv" bind:this={plotDiv}></div>
    </div>
</diV>
</div>
