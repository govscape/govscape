<script>
  import { onMount, onDestroy } from 'svelte';
  import { searchStore, searchActions } from '$lib/stores/search';
  import SearchBox from '$lib/components/SearchBox.svelte';
  import TypingEffect from '$lib/components/TypingEffect.svelte';

  const govDomains = [
    'epa.gov',
    'nsa.gov',
    'usda.gov',
    'sec.gov',
    'gpo.gov',
    'archives.gov',
  ];

  let isSmallScreen = false;

  function checkScreenSize() {
    isSmallScreen = window.innerWidth < 768;
  }

  onMount(() => {
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
  });

  onDestroy(() => {
    window.removeEventListener('resize', checkScreenSize);
    searchActions.reset();
  });
</script>

<main>
  <div class="title-container">
    <h1>
      {#if isSmallScreen}
        Search 4.7 Million PDFs across<br /><TypingEffect words={govDomains} />
      {:else}
        Search 4.7 Million PDFs across <TypingEffect words={govDomains} />
      {/if}
    </h1>
  </div>
  <SearchBox />
</main>

<style>
  main {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: calc(100vh - 50px);
    padding-top: 20vh;
  }

  .title-container {
    width: 98vw;
    max-width: 100vw;
    text-align: center;
  }

  .title-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.35;
    padding: 2rem;
    margin-bottom: 1.5rem;
  }

  @media (max-width: 767px) {
    main {
      padding-top: 50px;
    }
  }
</style>
