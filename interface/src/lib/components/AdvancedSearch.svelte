<script>
  import { fade, slide } from 'svelte/transition';
  import { searchStore, searchActions } from '$lib/stores/search';
  import { get } from 'svelte/store';
  import { goto } from '$app/navigation';
  export let show = false;

  let crawledAfter = '';
  let crawledBefore = ''
  let subDomain = '';
  let pageCount = '';

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

    updateURLWithFilters();
  }

  function handleFacetKeydown(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      applyFilters();
    }
  }
</script>

{#if show}
  <div transition:slide={{ duration: 300 }}>
    <div class="advanced-search-container">
      <div class="advanced-search-title">Search Filters</div>
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
      </div>
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

  @media (max-width: 767px) {
    .date-range-row input {
      flex: 1 1 120px;
      min-width: 120px;
    }
  }
</style>
