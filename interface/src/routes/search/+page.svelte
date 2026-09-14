<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { searchStore, searchActions } from '$lib/stores/search';
  import SearchBox from '$lib/components/SearchBox.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import { getApiBaseUrl, camelToSnake } from '$lib/utils/fetch';
  import { goto } from '$app/navigation';

  let shouldShowPreview = false;
  let selectedPDF = null;
  let downloadingCsv = false;
  let exportError = null;

  function handlePDFSelect(event) {
    const { id, page, crawlDate, crawlUrl, subDomain, crawlInstances, hasMoreCrawls, prettyName } = event.detail || {};
    selectedPDF = { id, page, crawlDate, crawlUrl, subDomain, crawlInstances, hasMoreCrawls, prettyName };
    shouldShowPreview = true;
  }

  function handleClosePreview() {
    shouldShowPreview = false;
    selectedPDF = null;
  }

  async function toggleExportEnabled() {
    searchActions.toggleExportEnabled();
    exportError = null;
  }

  function handlePageSizeChange(event) {
    const pageSize = Number(event.target.value);
    if (!Number.isFinite(pageSize) || pageSize <= 0) return;
    searchActions.setPageSize(pageSize);
    searchActions.goToPage(1, { isNewSearch: true });
    exportError = null;
  }

  async function downloadCsv() {
    const currentState = $searchStore;
    if (!currentState.query?.trim()) {
      exportError = 'Enter a search query before exporting.';
      return;
    }
    downloadingCsv = true;
    exportError = null;

    try {
      const body = JSON.stringify(camelToSnake({
        query: currentState.query,
        filters: currentState.filters,
        searchType: currentState.currentSearchMode,
        page: currentState.page,
        pageSize: currentState.pageSize,
      }));

      const response = await fetch(`${getApiBaseUrl()}/search/export/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to export CSV');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `govscape-search-page-${currentState.page}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export CSV failed:', err);
      exportError = err?.message || 'Export failed. Please try again.';
    } finally {
      downloadingCsv = false;
    }
  }

  function getParamsObject(searchParams) {
    const params = searchParams instanceof URLSearchParams
      ? searchParams
      : new URLSearchParams(searchParams || '');

    const q = params.get('q') || '';
    const mode = params.get('mode') || 'textual';
    const pageParam = parseInt(params.get('page') || '1', 10);

    const crawledAfter = params.get('after') || params.get('crawledAfter') || null;
    const crawledBefore = params.get('before') || params.get('crawledBefore') || null;
    const subDomain = params.get('subdomain') || params.get('subDomain') || null;

    return {
      q,
      mode: ['textual', 'visual', 'keyword'].includes(mode) ? mode : 'textual',
      page: Number.isFinite(pageParam) && pageParam > 0 ? pageParam : 1,
      filters: {
        crawledAfter: crawledAfter || null,
        crawledBefore: crawledBefore || null,
        subDomain: subDomain || null,
      }
    };
  }

  let lastApplied = null;
  let lastSearchParams = null;

  async function applyFromURL(urlSearchParams) {
    const { q, mode, page: pageNum, filters } = getParamsObject(urlSearchParams);

    if (!q.trim()) {
      goto('/', { replaceState: true });
      return;
    }

    const currentSignature = JSON.stringify({ q, mode, page: pageNum, filters });
    if (lastApplied === currentSignature) return;
    lastApplied = currentSignature;

    const currentSearchParams = JSON.stringify({ q, mode, filters });
    const isNewSearch = lastSearchParams !== currentSearchParams;
    lastSearchParams = currentSearchParams;

    searchActions.setSearchMode(mode);
    searchActions.setQuery(q);
    searchActions.updateFilters(filters);

    await searchActions.goToPage(pageNum, { isNewSearch });
  }

  function handleSetModeEvent(e) {
    const { mode } = e.detail || {};
    if (!mode?.id) return;
    const params = new URLSearchParams($page.url.searchParams);
    params.set('mode', mode.id);
    params.delete('page');
    const url = params.toString() ? `/search?${params.toString()}` : '/search';
    goto(url);
  }

  onMount(() => {
    applyFromURL($page.url.searchParams);
  });

  $: if ($page?.url) {
    applyFromURL($page.url.searchParams);
  }
</script>

<svelte:head>
  <title>{$searchStore.query ? `Search results for "${$searchStore.query}"` : 'Search'} - GovScape</title>
</svelte:head>

<main>
  <SearchBox on:setMode={handleSetModeEvent} />
  <section class="results-header">
    <div class="results-header-left">
      <h2>Search results for "{$searchStore.query}"</h2>
    </div>
    <div class="results-header-right">
      <label class="page-size-label" for="page-size-input">Export top:</label>
      <input
        id="page-size-input"
        class="page-size-input"
        type="number"
        min="1"
        step="1"
        value={$searchStore.pageSize}
        on:change={handlePageSizeChange}
      />
      <button
        class="export-toggle-button"
        type="button"
        on:click={toggleExportEnabled}
        aria-pressed={$searchStore.exportEnabled}
      >
        CSV export: {$searchStore.exportEnabled ? 'ON' : 'OFF'}
      </button>
      {#if $searchStore.exportEnabled}
        <button
          class="download-csv-button"
          type="button"
          on:click={downloadCsv}
          disabled={downloadingCsv || !$searchStore.query.trim()}
        >
          {downloadingCsv ? 'Preparing CSV…' : 'Download CSV'}
        </button>
      {/if}
    </div>
  </section>
  {#if exportError}
    <div class="export-error" role="alert">{exportError}</div>
  {/if}
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
    width: 100%;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 90%;
    max-width: 1400px;
    margin-bottom: 1rem;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .results-header-left h2 {
    font-size: 1.1rem;
    margin: 0;
    color: var(--text-color-primary);
  }

  .results-header-right {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .page-size-label {
    font-size: 0.9rem;
    color: var(--text-color-secondary);
    margin-right: 0.25rem;
  }

  .page-size-select {
    border-radius: 999px;
    border: 1px solid var(--color-primary);
    background: var(--background-color-primary);
    color: var(--text-color-primary);
    padding: 0.7rem 0.9rem;
    font-size: 0.9rem;
    min-width: 5.5rem;
    cursor: pointer;
  }

  .export-toggle-button,
  .download-csv-button {
    border-radius: 999px;
    border: 1px solid var(--color-primary);
    background: var(--background-color-primary);
    color: var(--color-primary);
    padding: 0.55rem 0.8rem;
    font-size: 0.85rem;
    cursor: pointer;
    transition: background-color 0.2s ease, color 0.2s ease;
  }

  .export-toggle-button[aria-pressed='true'],
  .download-csv-button:hover {
    background: var(--color-primary);
    color: white;
  }

  .download-csv-button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .export-error {
    width: 90%;
    max-width: 1400px;
    color: var(--danger-color, #d22);
    background: rgba(255, 224, 224, 0.85);
    border: 1px solid #f4c2c2;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
    text-align: left;
  }
</style>
