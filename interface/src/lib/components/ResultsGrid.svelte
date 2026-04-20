<script>
  // AI modified: 2026-03-14 4a6b1b72
  import { createEventDispatcher, onMount, afterUpdate, onDestroy } from 'svelte';
  import Masonry from 'masonry-layout';
  import { searchStore, searchActions } from '$lib/stores/search';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { userTracker } from '$lib/utils/userTracking.js';
  import ChevronLeftIcon from './icons/ChevronLeftIcon.svelte';
  import ChevronRightIcon from './icons/ChevronRightIcon.svelte';

  const dispatch = createEventDispatcher();

  let gridElement;
  let masonry;
  let previousPage;

  $: results = $searchStore.results;
  $: loading = $searchStore.loading;
  $: hasMore = $searchStore.hasMore;
  $: currentPage = $searchStore.page;
  $: pageSize = $searchStore.pageSize;
  $: totalCount = $searchStore.totalCount;
  $: totalPages = $searchStore.totalPages;
  $: error = $searchStore.error;
  $: startIndex = ((currentPage - 1) * pageSize) + 1;
  $: endIndex = ((currentPage - 1) * pageSize) + results.length;

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
    if (previousPage !== undefined && currentPage !== previousPage && gridElement) {
      window.scrollTo({ top: 0 });
      try {
        if (userTracker.hasConsent()) {
          userTracker.logPagination({
            query: $searchStore.query,
            searchType: $searchStore.currentSearchMode,
            filters: $searchStore.filters,
            page: currentPage,
          });
        }
      } catch (e) {}
    }
    previousPage = currentPage;
  })();

  function handlePDFSelect(pdf, page, crawlDate, crawlUrl, subDomain, crawlInstances, hasMoreCrawls, prettyName) {
    const pdfId = pdf.split('/').pop();

    try {
      userTracker.logPdfClick({
        id: pdfId,
        page,
        subDomain,
        crawlUrl,
        crawlDate
      });
    } catch (e) {}

    dispatch('pdfSelect', { pdf, page, id: pdfId, crawlDate, crawlUrl, subDomain, crawlInstances, hasMoreCrawls, prettyName });
  }

  function updatePageInURL(newPage) {
    const params = new URLSearchParams($page.url.searchParams);

    if (newPage > 1) {
      params.set('page', newPage.toString());
    } else {
      params.delete('page');
    }

    const newUrl = params.toString() ? `/search?${params.toString()}` : '/search';

    goto(newUrl, { replaceState: true, noScroll: true });
  }

  function nextPage() {
    if (!loading && hasMore) {
      const newPage = currentPage + 1;
      updatePageInURL(newPage);
    }
  }

  function prevPage() {
    if (!loading && currentPage > 1) {
      const newPage = currentPage - 1;
      updatePageInURL(newPage);
    }
  }
</script>

{#if loading}
  <div class="spinner-container" aria-label="Loading results">
    <div class="loader"></div>
  </div>
{/if}

<div class="grid-container">
  {#if error}
  <div class="error-banner" role="alert" aria-live="polite">
    {error}
  </div>
  {/if}
  {#if results.length > 0}
  <div class="results-summary">
    <button class="pagination-button" on:click={prevPage} disabled={loading || currentPage <= 1} aria-label="Previous Page">
      <ChevronLeftIcon />
    </button>
    <div class="summary-card">
      <div class="page-info">
        Page <span class="page-number">{currentPage.toLocaleString()}</span>
      </div>
      <div class="results-info">
        <span class="results-range">
          {startIndex} – {endIndex}
        </span>PDFs
      </div>
    </div>
    <button class="pagination-button" on:click={nextPage} disabled={loading || !hasMore} aria-label="Next Page">
      <ChevronRightIcon />
    </button>
  </div>
  {/if}

  <div class="masonry-wrapper" bind:this={gridElement}>
    {#each results as result (result.pdf + result.page)}
    <div class="grid-item">
      <div class="result-card" on:click={() => handlePDFSelect(result.pdf, result.page, result.crawlDate, result.crawlUrl, result.subDomain, result.crawlInstances, result.hasMoreCrawls, result.prettyName)}>
        <div class="image-container">
          <img
            src={result.jpeg}
            alt={`PDF Page ${result.page}`}
            loading="lazy"
          />
        </div>
        <div class="result-info">
          <div class="info-name">{result.prettyName || result.crawlUrl.split('/').pop().replaceAll("\%20", " ")}</div>
          <div class="info-subdomain">{result.subDomain || 'Not Available'}</div>
        </div>
      </div>
    </div>
    {/each}
  </div>

  {#if results.length > 0}
  <div class="pagination-container">
    <button class="pagination-button" on:click={prevPage} disabled={loading || currentPage <= 1} aria-label="Previous Page">
      <ChevronLeftIcon />
    </button>
    <span>
      Page
      <span class="page-number">{currentPage.toLocaleString()}</span>
    </span>
    <button class="pagination-button" on:click={nextPage} disabled={loading || !hasMore} aria-label="Next Page">
      <ChevronRightIcon />
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

  .pagination-button {
    width: 32px;
    height: 32px;
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

  .pagination-button:hover:not(:disabled) {
    box-shadow: 0 2px 8px rgba(32, 33, 36, 0.25);
  }

  .pagination-button:active:not(:disabled) {
    background: var(--color-secondary);
    box-shadow: 0 0 2px rgba(32, 33, 36, 0.3);
  }

  .pagination-button:disabled {
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
    align-items: center;
    gap: 1.5rem;
    margin: 0 0 1.25rem 0;
  }

  .error-banner {
    margin: 0 0 1rem 0;
    padding: 12px 16px;
    border-radius: 8px;
    background: #ffecec;
    color: #b00020;
    border: 1px solid #ffcccc;
    font-family: var(--sans-serif-font);
    text-align: center;
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

  .results-summary .results-range {
    margin-right: 6px;
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
