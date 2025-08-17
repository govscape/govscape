<script>
  import { onMount, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import { searchStore, searchActions } from '$lib/stores/search';
  import SearchBox from '$lib/components/SearchBox.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import TypingEffect from '$lib/components/TypingEffect.svelte';

  const govDomains = [
    'epa.gov',
    'nsa.gov',
    'usda.gov',
    'sec.gov',
    'gpo.gov',
    'archives.gov',
  ];

  let selectedPDF = null;
  let isSmallScreen = false;
  let shouldShowPreview = false;
  let hasSearched = false;
  
  $: if ($searchStore.results.length > 0 && !hasSearched) {
    hasSearched = true;
  }

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    shouldShowPreview = true;
  }

  function handleClosePreview() {
    shouldShowPreview = false;
    selectedPDF = null;
  }

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
    <h1 class:hidden={hasSearched}>
      {#if isSmallScreen}
        Search 4.7 Million PDFs across<br /><TypingEffect words={govDomains} />
      {:else}
        Search 4.7 Million PDFs across <TypingEffect words={govDomains} />
      {/if}
    </h1>
  </div>
  <SearchBox />
  <ResultsGrid on:pdfSelect={handlePDFSelect} />
  <PDFPreview 
    show={shouldShowPreview}
    pdfData={selectedPDF}
    on:close={handleClosePreview}
  />
</main>

<style>
  main {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: calc(100vh - 50px);
    padding-top: 80px;
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
    margin-bottom: 0.5rem;
    opacity: 1;
    transform: translateY(0);
    transition: all 0.3s ease;

    &.hidden {
      opacity: 0;
      transform: translateY(-20px);
      margin: 0;
      padding: 0;
      height: 0;
      pointer-events: none;
    }
  }
</style>
