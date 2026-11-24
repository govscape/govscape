import { writable, get } from 'svelte/store';
import { apiFetch } from '../utils/fetch';
import { pageImageSource } from '../utils/pageImages';
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
  hybridWeights: { textual: 0.33, visual: 0.33, keyword: 0.34 },
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

  setHybridWeights: (weights) => {
    searchStore.update(store => ({
      ...store,
      hybridWeights: {
        ...store.hybridWeights,
        ...weights
      }
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
      const body = { query, filters, searchType: currentSearchMode, page: pageNumber };
      if (currentSearchMode === 'hybrid_weights') {
        body.weights = get(searchStore).hybridWeights;
      }

      const responseData = await apiFetch('/search/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const results = (responseData.results || []).map(result => ({
        ...result,
        imageSource: pageImageSource(result.image, result.jpeg)
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
