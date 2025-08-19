<script>
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';

  onMount(() => {
    goto('/', { replaceState: true });
  });
</script>

<main>
  <p>Redirecting...</p>
</main>

<style>
  main {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: calc(100vh - 90px);
    padding-top: 80px;
  }
</style>
