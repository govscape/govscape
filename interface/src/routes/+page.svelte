<script>
  import { onMount, onDestroy } from 'svelte';
  import SearchBox from '$lib/components/SearchBox.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import TypingEffect from '$lib/components/TypingEffect.svelte';

  const govDomains = [
    'riversideca.gov',
    'tennille-ga.gov',
    'alabama.gov',
    'govinfo.gov',
    'sec.gov',
    'gpo.gov'
  ];

  let showPreview = false;
  let selectedPDF = null;
  let isSmallScreen = false;

  function handlePDFSelect(event) {
    selectedPDF = event.detail;
    showPreview = true;
  }

  function handleClosePreview() {
    showPreview = false;
    selectedPDF = null;
  }

  onMount(() => {
    function checkScreenSize() {
      isSmallScreen = window.innerWidth < 768;
    }
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
  });

  onDestroy(() => {
    window.removeEventListener('resize', checkScreenSize);
  });
</script>

<main>
  <div class="title-container {isSmallScreen ? 'small-screen' : ''}">
    <h1>
      {#if isSmallScreen}
        Search 1+ Million PDFs across <TypingEffect words={govDomains} />
      {:else}
        Search 1+ Million PDFs<br />across <TypingEffect words={govDomains} />
      {/if}
    </h1>
  </div>
  <SearchBox />
  <ResultsGrid on:pdfSelect={handlePDFSelect} />
  <PDFPreview 
    show={showPreview}
    pdfData={selectedPDF}
    on:close={handleClosePreview}
  />
  <footer class="coming-soon-banner">
    <div class="banner-content">
      <p class="uw-text">University of Washington Project</p>
      <p class="main-text">GovScape is coming soon</p>
      <p class="contact-text">
        For questions, please visit our <a href="/about">About page</a>
      </p>
    </div>
  </footer>
</main>

<style>
  main {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 150px;
    min-height: 100vh;
  }

  .title-container {
    width: 550px;
    max-width: 100vw;
    padding: 2rem;
    margin-bottom: 1rem;
    white-space: nowrap;
  }

  .title-container.small-screen {
    white-space: normal;
  }

  .title-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.35;
  }

  .coming-soon-banner {
    width: 100%;
    background: var(--color-secondary);
    color: white;
    padding: 2.5rem 0;
    margin-top: auto;
  }

  .banner-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    text-align: center;
  }

  .uw-text {
    font-family: var(--sans-serif-font);
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: rgba(255, 255, 255, 0.9);
    margin-bottom: 0.75rem;
  }

  .main-text {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    color: white;
  }

  .contact-text {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.9);
  }

  .contact-text a {
    color: white;
    text-decoration: none;
    font-weight: 500;
    transition: opacity 0.2s ease;
    display: inline-flex;
    align-items: center;
  }

  .contact-text a:hover {
    opacity: 0.8;
    text-decoration: underline;
  }
</style>
