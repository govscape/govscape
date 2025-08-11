import { writable, get } from 'svelte/store';
import { apiFetch, getImageBaseUrl } from '../utils/fetch';
import { userTracker } from '../utils/userTracking.js';

export const searchStore = writable({
  query: '',
  results: [],
  filters: {},
  currentSearchMode: 'textual',
  showFilters: false,
  loading: false,
  error: null,
  page: 1,
  hasMore: false,
});

export const searchActions = {
  setQuery: (query) => {
    searchStore.update(store => ({
      ...store,
      query
    }));
  },
  
  setSearchMode: (mode) => {
    searchStore.update(store => ({
      ...store,
      currentSearchMode: mode,
      results: [],
      error: null,
      page: 1,
      hasMore: false,
    }));
  },
  
  toggleFilters: () => {
    searchStore.update(store => ({
      ...store,
      showFilters: !store.showFilters
    }));
  },
  
  updateFilters: (newFilters) => {
    searchStore.update(store => ({
      ...store,
      filters: {
        ...store.filters,
        ...newFilters
      },
    }));
  },
  
  reset: () => {
    searchStore.set({
      query: '',
      filters: {},
      results: [],
      currentSearchMode: 'textual',
      loading: false,
      error: null,
      showFilters: false,
      page: 1,
      hasMore: false,
    });
  },
  
  performSearch: async () => {
    await searchActions.goToPage(1, { isNewSearch: true });
  },

  goToPage: async (pageNumber, options = {}) => {
    const { isNewSearch = false } = options;
    const currentStore = get(searchStore);

    if (currentStore.loading) return;
    if (!currentStore.query.trim()) {
      searchStore.update(store => ({ ...store, results: [], hasMore: false }));
      return;
    }

    searchStore.update(store => ({
      ...store,
      results: [],
      loading: true,
      error: null,
      page: pageNumber
    }));

    try {
      const { query, filters, currentSearchMode } = get(searchStore)
      const responseData = await apiFetch('/search/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, filters, search_type: currentSearchMode, page: pageNumber })
      });

      const imageBase = getImageBaseUrl(currentSearchMode);
      const results = (responseData.results || []).map(result => ({
        ...result,
        jpeg: result.jpeg && typeof result.jpeg === 'string'
              ? `${imageBase}/${result.jpeg.split('/').slice(-2).join('/')}`
              : null
      }));

      searchStore.update(store => ({
        ...store,
        results,
        hasMore: responseData.pagination.has_next_page,
        loading: false
      }));

      if (isNewSearch && userTracker.hasConsent()) {
        userTracker.logSearch(query, currentSearchMode, filters);
      }
    } catch (err) {
      console.error('Search error in store:', err);
      searchStore.update(store => ({
        ...store,
        error: err.message || 'Search failed',
        loading: false,
        results: [],
        hasMore: false
      }));
    }
  }
};
