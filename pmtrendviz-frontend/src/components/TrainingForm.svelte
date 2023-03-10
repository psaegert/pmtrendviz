<script>

  import { Dropdown, DropdownItem, DropdownMenu, DropdownToggle, Form, FormGroup } from "sveltestrap";

  export let model_args;
  export let general_args;
  var selected = Object.keys(model_args)[0];

  let args_entries = Object.entries(model_args);

  export let form;
</script>

<style>
  p {
      margin-bottom: 10px;
  }
</style>

<p>
  <Dropdown>
    <DropdownToggle caret style="background-color: var(--primary-color)">{selected}</DropdownToggle>
    <DropdownMenu>
      {#each args_entries as [name, arg_list]}
        <DropdownItem on:click={(e) => selected = e.target.innerText}>{name}</DropdownItem>
      {/each}
    </DropdownMenu>
  </Dropdown>
</p>

<Form bind:this={form}>
  {#each model_args[selected] as arg_block}
    <div class="form-group">
      <label >{arg_block.title}</label>
      <p/>
      {#each Object.entries(arg_block.properties) as [arg_name, arg]}
        <FormGroup row  title={arg.description}>
          <label for={arg.title} class="col-sm-4 col-form-label col-form-label-sm">{arg.title}</label>
          <div class="col-sm-5">
            {#if arg.type == 'integer'}
              <!--suppress XmlDuplicatedId -->
              <input id={arg.title} type="number" class="form-control form-control-sm" value={arg.default} />
            {:else if arg.type == 'number'}
              <!--suppress XmlDuplicatedId -->
              <input id={arg.title} type="number" step="0.001" class="form-control form-control-sm" value={arg.default} />
            {:else if arg.type == 'string'}
              <!--suppress XmlDuplicatedId -->
              <input id={arg.title} type="text" class="form-control form-control-sm" value={arg.default} />
            {:else if arg.type == 'boolean'}
              {#if arg.default}
                <!--suppress XmlDuplicatedId -->
                <input id={arg.title} type="checkbox" class="form-check-input" checked/>
              {:else}
                <!--suppress XmlDuplicatedId -->
                <input id={arg.title} type="checkbox" class="form-check-input" />
              {/if}
            {:else if arg.type == 'array' && arg.items.type == 'string'}
              <!--suppress XmlDuplicatedId -->
              <select id={arg.title} multiple class="form-control form-control-sm">
                {#each arg.items.enum as item}
                  <option>{item}</option>
                {/each}
              </select>
            {:else}
              <!--suppress XmlDuplicatedId -->
              <input id={arg.title} type="text" class="form-control form-control-sm" value={arg.default} />
            {/if}
          </div>
        </FormGroup>
      {/each}
      <hr/>
    </div>
  {/each}


<div class="form-group">
  <label >{general_args.title}</label>
  <p/>
  {#each Object.entries(general_args.properties) as [arg_name, arg]}
    <FormGroup row  title={arg.description}>
      <label for={arg.title} class="col-sm-4 col-form-label col-form-label-sm">{arg.title}</label>
      <div class="col-sm-5">
        {#if arg.type == 'integer'}
          <!--suppress XmlDuplicatedId -->
          <input id={arg.title} type="number" class="form-control form-control-sm" value={arg.default} />
        {:else if arg.type == 'number'}
          <!--suppress XmlDuplicatedId -->
          <input id={arg.title} type="number" step="0.001" class="form-control form-control-sm" value={arg.default} />
        {:else if arg.type == 'string'}
          <!--suppress XmlDuplicatedId -->
          <input id={arg.title} type="text" class="form-control form-control-sm" value={arg.default} />
        {:else if arg.type == 'boolean'}
          {#if arg.default}
            <!--suppress XmlDuplicatedId -->
            <input id={arg.title} type="checkbox" class="form-check-input" checked/>
          {:else}
            <!--suppress XmlDuplicatedId -->
            <input id={arg.title} type="checkbox" class="form-check-input" />
          {/if}
        {:else if arg.type == 'array' && arg.items.type == 'string'}
          <!--suppress XmlDuplicatedId -->
          <select id={arg.title} class="form-control form-control-sm">
            {#each arg.items.enum as item}
              <option>{item}</option>
            {/each}
          </select>
        {:else}
          <!--suppress XmlDuplicatedId -->
          <input id={arg.title} type="text" class="form-control form-control-sm" value={arg.default} />
        {/if}
      </div>
    </FormGroup>
  {/each}
    <hr/>
  </div>

</Form>