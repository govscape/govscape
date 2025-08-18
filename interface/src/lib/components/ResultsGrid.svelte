<script>
  import { createEventDispatcher, onMount, afterUpdate, onDestroy } from 'svelte';
  import Masonry from 'masonry-layout';
  import { searchStore, searchActions } from '$lib/stores/search';

  const dispatch = createEventDispatcher();

  let gridElement;
  let masonry;
  let previousPage;

  $: results = $searchStore.results;
  $: loading = $searchStore.loading;
  $: hasMore = $searchStore.hasMore;
  $: page = $searchStore.page;
  $: pageSize = $searchStore.pageSize;
  $: totalCount = $searchStore.totalCount;
  $: totalPages = $searchStore.totalPages;

  onMount(() => {
    if (!gridElement) return;

    masonry = new Masonry(gridElement, {
      itemSelector: '.grid-item',
      columnWidth: 280,
      gutter: 20,
      fitWidth: true
    });
  });

  afterUpdate(() => {
    if (masonry) {
      masonry.reloadItems();
      masonry.layout();
    }
  });

  onDestroy(() => {
    if (masonry) {
      masonry.destroy();
    }
  });
  
  $: onPageChange = (() => {
    if (typeof window === 'undefined') return;
    if (previousPage !== undefined && page !== previousPage && gridElement) {
      window.scrollTo({ top: 0 });
    }
    previousPage = page;
  })();

  function handlePDFSelect(pdf, page, crawl_date, crawl_url, sub_domain) {
    const pdfId = pdf.split('/').pop();

    dispatch('pdfSelect', { pdf, page, id: pdfId, crawl_date, crawl_url, sub_domain});
  }

  function nextPage() {
    if (!loading && hasMore) {
      searchActions.goToPage(page + 1);
    }
  }

  function prevPage() {
    if (!loading && page > 1) {
      searchActions.goToPage(page - 1);
    }
  }
</script>

{#if loading}
  <div class="spinner-container" aria-label="Loading results">
    <div class="loader"></div>
  </div>
{/if}

<div class="grid-container">
  {#if results.length > 0}
  <div class="results-summary">
    <div class="summary-card">
      <div class="page-info">
        Page <span class="page-number">{page.toLocaleString()}</span>
        {#if totalPages}
        of <span class="page-number">{totalPages.toLocaleString()}</span>
        {/if}
      </div>
      {#if totalCount}
      <div class="results-info">
        <span class="results-range">
          {((page - 1) * pageSize) + 1} – {Math.min(((page - 1) * pageSize) + pageSize, totalCount)}
        </span>
        of
        <span class="total-results">{totalCount.toLocaleString()}</span>
        Results
      </div>
      {/if}
    </div>
  </div>
  {/if}
  
  <div class="masonry-wrapper" bind:this={gridElement}>
    {#each results as result (result.pdf + result.page)}
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
  </div>

  {#if results.length > 0}
  <div class="pagination-container">
    <button on:click={prevPage} disabled={loading || page <= 1} aria-label="Previous Page">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="15 18 9 12 15 6"></polyline>
      </svg>
    </button>
    <span>
      Page
      <span class="page-number">{page.toLocaleString()}</span>
      {#if totalPages}
      of <span class="page-number">{totalPages.toLocaleString()}</span>
      {/if}
    </span>
    <button on:click={nextPage} disabled={loading || !hasMore} aria-label="Next Page">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="9 18 15 12 9 6"></polyline>
      </svg>
    </button>
  </div>
  {/if}
</div>

<style>
  .grid-container {
    width: 90%;
    max-width: 1400px;
    padding: 10px 0 20px 0;
  }

  .masonry-wrapper {
    margin: 0 auto;
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
    transition: transform 0.2s;
  }

  .result-card:hover {
    transform: translateY(-3px);
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

  .pagination-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1rem 0 2rem 0;
    gap: 2rem;
    color: var(--text-color-secondary);
    font-family: var(--sans-serif-font);
    font-size: 0.9rem;
  }

  .pagination-container button {
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: var(--color-primary);
    border: none;
    box-shadow: 0 1px 6px rgba(32, 33, 36, 0.08);
    color: #fff;
    cursor: pointer;
    transition: background 0.15s, box-shadow 0.15s;
  }

  .pagination-container button:hover:not(:disabled) {
    box-shadow: 0 2px 8px rgba(32, 33, 36, 0.25);
  }
  
  .pagination-container button:active:not(:disabled) {
    background: var(--color-secondary);
    box-shadow: 0 0 2px rgba(32, 33, 36, 0.3);
  }

  .pagination-container button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
    box-shadow: none;
  }

  .pagination-container .page-number {
    font-weight: 500;
    color: var(--text-color-primary);
  }

  .spinner-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 40px 0;
  }

  .loader {
    width: 44px;
    height: 44px;
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-top: 4px solid var(--color-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .results-summary {
    display: flex;
    justify-content: center;
    margin: 0 0 1.25rem 0;
  }

  .summary-card {
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: var(--sans-serif-font);
  }

  .page-info {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .page-info, .results-info {
    font-size: 0.85rem;
    color: var(--text-color-secondary);
  }

  .results-summary .page-number,
  .results-summary .results-range,
  .results-summary .total-results {
    color: var(--text-color-primary);
    font-weight: 500;
  }

  .results-info {
    padding-left: 16px;
    border-left: 1px solid var(--border-color-primary);
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
</style>
