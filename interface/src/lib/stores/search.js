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
  pageSize: 20,
  hasMore: false,
  totalCount: null,
  totalPages: null,
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
      pageSize: 20,
      hasMore: false,
      totalCount: null,
      totalPages: null,
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
      pageSize: 20,
      hasMore: false,
      totalCount: null,
      totalPages: null,
    });
  },

  performSearch: async () => {
    await searchActions.goToPage(1, { isNewSearch: true });
  },

  goToPage: async (pageNumber, options = {}) => {
    const { isNewSearch = false } = options;
    const currentStore = get(searchStore);

    if (currentStore.loading || !currentStore.query.trim()) return;

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
        body: JSON.stringify({ query, filters, searchType: currentSearchMode, page: pageNumber })
      });

      const imageBase = getImageBaseUrl();
      const results = (responseData.results || []).map(result => ({
        ...result,
        jpeg: result.jpeg && typeof result.jpeg === 'string'
              ? `${imageBase}/${result.jpeg.split('/').slice(-2).join('/')}`
              : null
      }));

      searchStore.update(store => ({
        ...store,
        results,
        loading: false,
        hasMore: responseData.pagination.hasNextPage,
        totalCount: responseData.pagination.totalCount,
        totalPages: responseData.pagination.totalPages
      }));

      if (isNewSearch && userTracker.hasConsent()) {
        userTracker.logSearch(query, currentSearchMode, filters);
      }
    } catch (err) {
      console.error('Search error in store:', err);
      searchStore.update(store => ({
        ...store,
        error: err?.message === 'Request timed out' ? 'Search timed out. Please try again or try a different search.' : (err.message || 'Search failed'),
        loading: false,
        results: [],
        hasMore: false,
        totalCount: null,
        totalPages: null
      }));
    }
  }
};
