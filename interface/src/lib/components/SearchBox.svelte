<script>
  import { searchStore, searchActions } from '$lib/stores/search';
  import { tick } from 'svelte';
  import { get } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { createEventDispatcher } from 'svelte';
  import AdvancedSearch from './AdvancedSearch.svelte';
  import GlobeIcon from './icons/GlobeIcon.svelte';
  import FilterIcon from './icons/FilterIcon.svelte';

  const dispatch = createEventDispatcher();

  const suggestionsByMode = {
    visual: ['redacted documents', 'aerial photography', 'pie charts', 'maps of seattle'],
    textual: ['pediatric healthcare in rural areas', 'budgetary data for environmental surveys', 'scientific innovation and policy'],
    keyword: ['covid-19', 'social security', 'FAFSA']
  };
  const searchModes = [
    { id: 'textual', label: 'Semantic Text Search', placeholder: 'Search PDFs with context-rich text search...' },
    { id: 'visual', label: 'Visual Search', placeholder: 'Search PDFs using image semantics...' },
    { id: 'keyword', label: 'Keyword Search', placeholder: 'Enter keywords for search...' },
  ];

  let currentSearchMode = searchModes[0];
  let searchInputFocused = false;
  let showSuggestionsDropdown = false;
  let searchInputElement;
  let query = '';

  $: suggestions = suggestionsByMode[currentSearchMode.id] || [];

  // Keep local query in sync with store only when not typing (e.g., URL-driven changes)
  $: if (!searchInputFocused && $searchStore.query !== query) {
    query = $searchStore.query || '';
  }

  // Update store when local query changes (user typing)
  $: if (query !== undefined && get(searchStore).query !== query) {
    searchActions.setQuery(query);
  }

  // Keep local mode in sync with store
  $: if (currentSearchMode.id !== $searchStore.currentSearchMode) {
    const match = searchModes.find(m => m.id === $searchStore.currentSearchMode);
    currentSearchMode = match || searchModes[0];
  }

  function setMode(mode) {
    currentSearchMode = mode;
    searchActions.setSearchMode(mode.id);
    dispatch('setMode', { mode });
  }

  function buildSearchParams() {
    const { filters, currentSearchMode: modeFromStore, query: storeQuery } = get(searchStore);
    const params = new URLSearchParams();

    if (storeQuery && storeQuery.trim()) params.set('q', storeQuery.trim());
    if (modeFromStore) params.set('mode', modeFromStore);

    if (filters) {
      if (filters.crawledAfter) params.set('after', filters.crawledAfter);
      if (filters.crawledBefore) params.set('before', filters.crawledBefore);
      if (filters.subDomain) params.set('subdomain', filters.subDomain);
    }

    // Always reset to page 1 on new searches by omitting page
    return params;
  }

  function navigateToSearch() {
    const params = buildSearchParams();
    const url = params.toString() ? `/search?${params.toString()}` : '/search';
    goto(url);
  }

  function handleSearch() {
    searchActions.setQuery(query || '');
    navigateToSearch();
    if (searchInputElement) searchInputElement.blur();
  }

  function handleFilterToggle() {
    searchActions.toggleFilters();
  }

  async function applySuggestion(suggestion, event) {
    event?.preventDefault();
    query = suggestion;
    searchActions.setQuery(suggestion);
    await tick();
    showSuggestionsDropdown = false;
    setTimeout(() => navigateToSearch(), 0);
  }

  function handleInputFocus() {
    showSuggestionsDropdown = true;
    searchInputFocused = true;
  }

  function handleInputBlur() {
    showSuggestionsDropdown = false;
    searchInputFocused = false;
  }
</script>

<div class="search-container">
  <div class="search-mode-tabs">
    <div
      class="search-mode-toggle-bg"
      class:toggle-left={currentSearchMode.id === 'textual'}
      class:toggle-middle={currentSearchMode.id === 'visual'}
      class:toggle-right={currentSearchMode.id === 'keyword'}
    ></div>
    {#each searchModes as mode}
      <button
        type="button"
        class:active-tab={currentSearchMode.id === mode.id}
        on:click={() => setMode(mode)}
        aria-label={mode.label}
      >
        {mode.label}
      </button>
    {/each}
  </div>

  <div class="flexbox">
    <div class="suggestions-dropdown-wrapper">
      <form on:submit|preventDefault={handleSearch}>
        <div class="search-input-wrapper" class:input-focused={searchInputFocused}>
          <GlobeIcon />
          <input
            class="search-input"
            type="text"
            placeholder={currentSearchMode.placeholder}
            bind:this={searchInputElement}
            bind:value={query}
            on:focus={handleInputFocus}
            on:blur={handleInputBlur}
            aria-label="Search input"
          />
          <button
            type="button"
            on:click={handleFilterToggle}
            class="filter-toggle-button"
            class:active={$searchStore.showFilters}
            aria-label="Toggle advanced search filters"
            aria-pressed={$searchStore.showFilters}
          >
            <FilterIcon />
          </button>
        </div>
      </form>

      {#if showSuggestionsDropdown && suggestions.length > 0}
        <div class="suggestions-dropdown">
          <ul>
            {#each suggestions as suggestionText}
              <li on:mousedown={(e) => applySuggestion(suggestionText, e)} role="option" aria-selected="false" tabindex="0">
                <svg class="suggestion-item-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                </svg>
                <span>{suggestionText}</span>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    </div>
    <button type="button" class="search-icon-button" on:click={handleSearch} aria-label="Search">
      <svg class="search-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="9" r="8" />
        <line x1="21" y1="20" x2="16.65" y2="15.65" />
      </svg>
    </button>
  </div>

  <AdvancedSearch show={$searchStore.showFilters} />
</div>

<style>
  .search-container {
    display: flex;
    flex-direction: column;
    margin-bottom: 1rem;
    width: 55vw;
    min-width: 400px;
  }

  .suggestions-dropdown-wrapper {
    position: relative;
    width: 100%;
  }

  .search-input-wrapper {
    flex: 1;
    display: flex;
    align-items: center;
    border-radius: 24px;
    background-color: #fff;
    box-shadow: 0 1px 6px rgba(32, 33, 36, 0.08);
    padding: 0 10px 0 14px;
    transition: box-shadow 0.2s, border-radius 0.1s ease-out;
  }

  .search-input-wrapper.input-focused {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
  }

  .search-input {
    width: 100%;
    flex-grow: 1;
    padding: 12px 8px 12px 12px;
    border: none;
    background-color: transparent;
    color: var(--text-color-primary);
    font-family: var(--sans-serif-font);
    outline: none;
  }

  .search-input::placeholder {
    color: var(--text-color-secondary);
    font-family: var(--serif-font);
    font-size: 0.9rem;
  }

  .filter-toggle-button {
    background: none;
    border: none;
    padding: 8px;
    margin-left: 8px;
    cursor: pointer;
    color: var(--text-color-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: background-color 0.2s;
  }

  .filter-toggle-button:hover {
    background-color: rgba(60, 60, 60, 0.08);
  }

  .filter-toggle-button:active {
    background: rgba(60, 60, 60, 0.3);
  }

  .filter-toggle-button.active {
    background-color: var(--background-color-primary);
    color: var(--color-primary);
  }

  .filter-toggle-button.active:hover {
    box-shadow: 0 2px 8px rgba(32, 33, 36, 0.15);
  }

  .filter-toggle-button.active:active {
    background: var(--background-color-secondary);
    color: var(--text-color-primary);
  }

  .search-mode-tabs {
    position: relative;
    display: flex;
    background: #fff;
    border-radius: 999px;
    padding: 4px;
    margin: 0 auto 16px auto;
    width: fit-content;
    box-shadow: 0 2px 8px rgba(32, 33, 36, 0.08);
    gap: 8px;
  }

  .search-mode-toggle-bg {
    position: absolute;
    top: 4px;
    left: 4px;
    width: calc(33.33% - 4px);
    height: calc(100% - 8px);
    background: var(--color-primary);
    border-radius: 999px;
    transition: transform 0.25s cubic-bezier(0.4, 1.2, 0.4, 1), background 0.18s;
  }

  .search-mode-toggle-bg.toggle-left {
    transform: translateX(0%);
  }

  .search-mode-toggle-bg.toggle-middle {
    transform: translateX(100%);
  }

  .search-mode-toggle-bg.toggle-right {
    transform: translateX(203.5%);
  }

  .search-mode-tabs button {
    position: relative;
    min-width: 150px;
    flex: 1 1 0;
    border: none;
    background: transparent;
    color: var(--text-color-secondary);
    font-family: var(--sans-serif-font);
    font-size: 0.75rem;
    font-weight: 500;
    padding: 8px;
    border-radius: 999px;
    cursor: pointer;
    transition: color 0.18s;
    display: flex;
    align-items: center;
    justify-content: center;
    outline: none;
  }

  .search-mode-tabs button.active-tab {
    color: #fff;
  }

  .search-mode-tabs button:not(.active-tab):hover {
    background: #f3f4f6;
    color: var(--text-color-primary);
  }

  .suggestions-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background-color: #fff;
    border: 1px solid transparent;
    border-top: none;
    box-shadow: 0 6px 12px -4px rgba(32, 33, 36, 0.18);
    border-bottom-left-radius: 24px;
    border-bottom-right-radius: 24px;
    z-index: 100;
    padding-top: 10px;
    padding-bottom: 10px;
  }

  .suggestions-dropdown ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .suggestions-dropdown li {
    display: flex;
    align-items: center;
    padding: 8px 20px;
    cursor: pointer;
    font-family: var(--sans-serif-font);
    color: var(--text-color-primary);
    font-size: 0.9rem;
  }

  .suggestions-dropdown li:hover {
    background-color: var(--background-color-secondary);
  }

  .suggestion-item-icon {
    margin-right: 12px;
    color: var(--text-color-primary);
  }

  .search-icon-button {
    width: 44px;
    height: 44px;
    flex-shrink: 0;
    border-radius: 50%;
    background: var(--color-primary);
    border: none;
    box-shadow: 0 1px 6px rgba(32, 33, 36, 0.08);
    color: #fff;
    cursor: pointer;
    transition: background 0.15s, box-shadow 0.15s;
    margin-left: 16px;
  }

  .search-icon-button:hover {
    box-shadow: 0 2px 8px rgba(32, 33, 36, 0.25);
  }

  .search-icon-button:active {
    background: var(--color-secondary);
    box-shadow: 0 0 2px rgba(32, 33, 36, 0.3);
  }

  @media (max-width: 767px) {
    .search-container {
      width: 90vw;
      min-width: unset;
    }

    .search-mode-tabs button {
      min-width: unset;
      font-size: 0.7rem;
      padding: 6px 12px;
    }
  }
</style>
