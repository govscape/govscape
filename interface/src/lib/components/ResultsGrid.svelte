<script>
  import { createEventDispatcher, onMount, afterUpdate } from 'svelte';
  import Masonry from 'masonry-layout';
  import { searchStore } from '$lib/stores/search';

  const dispatch = createEventDispatcher();

  let gridElement;
  let masonry;

  $: results = $searchStore.results;

  function handlePDFSelect(pdf, page, crawl_date, crawl_url, sub_domain) {
    const pdfId = pdf.split('/').pop();

    dispatch('pdfSelect', { pdf, page, id: pdfId, crawl_date, crawl_url, sub_domain});  
  }

  onMount(() => {
    if (typeof window !== 'undefined' && gridElement) {
      masonry = new Masonry(gridElement, {
        itemSelector: '.grid-item',
        columnWidth: 280,
        gutter: 20,
        fitWidth: true
      });
    }
  });

  afterUpdate(() => {
    if (masonry && results.length > 0) {
      setTimeout(() => {
        masonry.reloadItems();
        masonry.layout();
      }, 0);
    }
  });
</script>

<div class="grid-container">
  <div bind:this={gridElement}>
    {#if results.length > 0}
      {#each results as result}
      <div class="grid-item">
        <div class="result-card" on:click={() => handlePDFSelect(result.pdf, result.page, result.crawl_date, result.crawl_url, result.sub_domain)}>
          <div class="image-container">
            <img 
              src={result.jpeg} 
              alt={`PDF Page ${result.page}`}
              loading="lazy"
            />
          </div>
          <div class="result-info">
            <div class="info-name">{result.crawl_url.split('/').pop().replaceAll("\%20", " ")}</div>
            <div class="info-subdomain">{result.sub_domain || 'Not Available'}</div>
          </div>
        </div>
      </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .grid-container {
    width: 90%;
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    justify-content: center;
    padding: 50px 0 100px 0;
  }

  .grid-item {
    width: 280px;
    margin-bottom: 20px;
  }

  .result-card {
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    overflow: hidden;
  }

  .result-card:hover {
    transform: translateY(-2px);
  }

  .image-container {
    width: 100%;
    height: 200px;
  }

  .image-container img {
    width: 100%;
    height: 100%;
    object-fit: none;
  }

  .result-info {
    max-width: 270px;
    padding: 12px;
    color: var(--text-color-primary);
  }

  .info-name {
    font-size: 1rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .info-subdomain {
    font-size: 0.8rem;
    color: var(--text-color-secondary);
  }
</style>
