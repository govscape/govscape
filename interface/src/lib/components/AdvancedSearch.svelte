<script>
  import { fade, slide } from 'svelte/transition';
  import { searchStore, searchActions } from '$lib/stores/search';
  import { get } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { getApiBaseUrl, camelToSnake } from '$lib/utils/fetch';
  export let show = false;

  let crawledAfter = '';
  let crawledBefore = ''
  let subDomain = '';
  let pageCount = '';
  let downloadingCsv = false;
  let exportError = null;

  const subdomainOptions = [
    { value: '', label: 'Any subdomain' },
    { value: 'tennille-ga.gov', label: 'tennille-ga.gov' },
    { value: 'govinfo.gov', label: 'govinfo.gov' },
    { value: 'westhaverstraw.org', label: 'westhaverstraw.org' },
    { value: 'sec.gov', label: 'sec.gov' },
    { value: 'ca.gov', label: 'ca.gov' },
    { value: 'usda.gov', label: 'usda.gov' },
    { value: 'census.gov', label: 'census.gov' },
    { value: 'wa.gov', label: 'wa.gov' },
    { value: 'utah.gov', label: 'utah.gov' },
    { value: 'nasa.gov', label: 'nasa.gov' },
  ];

  $: if ($searchStore?.filters) {
    crawledAfter = $searchStore.filters.crawledAfter || '';
    crawledBefore = $searchStore.filters.crawledBefore || '';
    subDomain = $searchStore.filters.subDomain || '';
  }

  function updateURLWithFilters() {
    const store = get(searchStore);
    const params = new URLSearchParams();

    if (store.query && store.query.trim()) params.set('q', store.query.trim());
    if (store.currentSearchMode) params.set('mode', store.currentSearchMode);

    if (crawledAfter) params.set('after', crawledAfter);
    if (crawledBefore) params.set('before', crawledBefore);
    if (subDomain) params.set('subdomain', subDomain);

    // Reset to first page on filter changes by omitting page
    const url = params.toString() ? `/search?${params.toString()}` : '/search';
    goto(url);
  }

  function applyFilters() {
    let minPages = null;
    let maxPages = null;

    if (pageCount) {
      const parts = pageCount.split('-');
      if (parts.length === 1 && pageCount.endsWith('+')) {
        minPages = parseInt(parts[0].slice(0, -1));
      } else if (parts.length === 1) {
        minPages = parseInt(parts[0]);
        maxPages = minPages; // Exact match if only one number
      } else if (parts.length === 2) {
        minPages = parts[0] ? parseInt(parts[0]) : null;
        maxPages = parts[1] ? parseInt(parts[1]) : null;
      }
    }

    searchActions.updateFilters({
      crawledAfter: crawledAfter || null,
      crawledBefore: crawledBefore || null,
      subDomain: subDomain || null,
      minPages: minPages,
      maxPages: maxPages,
    });

  }

  function handleFacetKeydown(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      applyFilters();
      updateURLWithFilters();
    }
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
    const currentState = get(searchStore);
    if (!currentState.query?.trim()) {
      exportError = 'Enter a search query before exporting.';
      return;
    }

    downloadingCsv = true;
    exportError = null;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

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
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to export CSV');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `govscape-search-export-${currentState.pageSize}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export CSV failed:', err);
      exportError = err?.name === 'AbortError' ? 'Export timed out. Please try again.' : (err?.message || 'Export failed. Please try again.');
    } finally {
      clearTimeout(timeoutId);
      downloadingCsv = false;
    }
  }
</script>

{#if show}
  <div transition:slide={{ duration: 300 }}>
    <div class="advanced-search-container">
      <div class="advanced-search-title">Search Options</div>
      <div class="filters-grid">
        <div class="filter-item">
          <label for="crawlDate">Crawl Date</label>
          <div class="date-range-row">
            <input type="text" id="crawlDateAfter" placeholder="YYYY-MM-DD" bind:value={crawledAfter} on:change={applyFilters} on:keydown={handleFacetKeydown}/>
            <span class="em-dash">&mdash;</span>
            <input type="text" id="crawlDateBefore" placeholder="YYYY-MM-DD" bind:value={crawledBefore} on:change={applyFilters} on:keydown={handleFacetKeydown}/>
          </div>
        </div>
        <div class="filter-item">
          <label for="subdomain">Subdomain</label>
          <input type="text" id="subdomain" placeholder="Enter subdomain" bind:value={subDomain} list="subdomain-options" on:change={applyFilters} on:keydown={handleFacetKeydown} />
          <datalist id="subdomain-options">
            {#each subdomainOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </datalist>
        </div>

        <div class="filter-item export-item">
          <label for="advanced-search-page-size-input">Export Top Results</label>
          <div class="export-controls-row">
            <input
              id="advanced-search-page-size-input"
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
              CSV Export: {$searchStore.exportEnabled ? 'On' : 'Off'}
            </button>
          </div>
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
      </div>

      {#if exportError}
        <div class="export-error" role="alert">{exportError}</div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .advanced-search-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 24px;
    padding: 16px;
    background: #fff;
    border-radius: 24px;
    box-shadow: 0 1px 6px rgba(32, 33, 36, 0.08);
  }

  .advanced-search-title {
    color: var(--text-color-primary);
    margin-bottom: 1rem;
    align-self: flex-start;
  }

  .filters-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    width: 100%;
  }

  .filter-item {
    display: flex;
    flex-direction: column;
  }

  .filter-item label {
    font-size: 0.8rem;
    color: var(--text-color-secondary);
    margin-bottom: 0.3rem;
  }

  .filter-item select,
  .filter-item input {
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #fff;
    font-size: 0.8rem;
    color: var(--text-color-primary);
    padding: 8px;
  }

  .filter-item input::placeholder {
    color: var(--text-color-secondary);
  }

  .export-item {
    gap: 0.3rem;
  }

  .export-controls-row {
    display: flex;
    gap: 8px;
  }

  .export-controls-row .page-size-input {
    width: 4.5rem;
    flex: none;
  }

  .export-toggle-button,
  .download-csv-button {
    font-family: var(--sans-serif-font);
    font-weight: 500;
    font-size: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #fff;
    color: var(--text-color-secondary);
    padding: 8px 12px;
    cursor: pointer;
    white-space: nowrap;
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
  }

  .export-toggle-button {
    flex: 1;
  }

  .export-toggle-button[aria-pressed='true'] {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: #fff;
  }

  .download-csv-button {
    margin-top: 8px;
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: #fff;
  }

  .download-csv-button:hover:not(:disabled) {
    background: var(--color-secondary);
    border-color: var(--color-secondary);
  }

  .download-csv-button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .export-error {
    width: 100%;
    color: var(--danger-color, #d22);
    background: rgba(255, 224, 224, 0.85);
    border: 1px solid #f4c2c2;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-top: 12px;
    text-align: left;
  }

  .date-range-row {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .em-dash {
    margin: 0 4px;
    font-size: 1.2em;
    color: var(--text-color-secondary, #888);
  }

  .date-range-row input {
    width: 140px;      /* desired fixed width */
    min-width: 80px;   /* don't shrink below this */
    box-sizing: border-box;
  }

  @media (max-width: 480px) {
    .date-range-row input {
      width: 100px;    /* smaller on narrow screens */
    }
  }

  @media (max-width: 320px) {
    .date-range-row input {
      width: 80px;
    }
  }
</style>
