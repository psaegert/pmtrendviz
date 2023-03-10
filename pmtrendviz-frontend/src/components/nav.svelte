<script lang="ts">
  import {
    Collapse,
    Navbar,
    NavbarToggler,
    NavbarBrand,
    Nav,
    NavItem,
    NavLink,
  } from 'sveltestrap/src';

  let isOpen = false;
</script>

<style>
  :root {
      --primary-color: #011526;
      --secondary-color: #c9dff2;
      --accent-color: #d98f4e;
  }
</style>

<svelte:head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
</svelte:head>

<Navbar style="background-color: var(--primary-color);" expand="sm">
  <span>
  <NavbarToggler on:click={() => (isOpen = !isOpen)}/>
  <Collapse isOpen={isOpen} navbar expand="sm">
    <Nav class="mr-auto navbar-nav" navbar>
      <NavItem>
        <!-- <NavLink style="color: #ffffff; font-weight: bold;" href="/import">Import</NavLink> -->
        <NavLink style="color: #aaaaaa; font-weight: bold;" href="/predict">(Import)</NavLink>
      </NavItem>
      <NavItem>
        <!-- <NavLink style="color: #ffffff; font-weight: bold;" href="/train">Train</NavLink> -->
        <NavLink style="color: #aaaaaa; font-weight: bold;" href="/predict">(Train)</NavLink>
      </NavItem>
      <NavItem>
        <NavLink style="color: #ffffff; font-weight: bold;" href="/predict">Predict</NavLink>
      </NavItem>
    </Nav>
  </Collapse>
  </span>
</Navbar>
