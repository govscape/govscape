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
  error: null
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
      }
    }));
  },
  
  clearResults: () => {
    searchStore.update(store => ({
      ...store,
      results: [],
      error: null
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
      showFilters: false
    });
  },
  
  performSearch: async (searchMode) => {
    const currentStore = get(searchStore);
    const search_type = currentStore.currentSearchMode;

    const { query, filters } = currentStore;

    searchStore.update(store => ({ 
      ...store, 
      loading: true, 
      error: null
    }));

    try {
      const responseData = await apiFetch('/search/', { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query, filters, search_type }) 
      }, search_type); 
      
      const imageBase = getImageBaseUrl(search_type);
      const results = (responseData.results || []).map(result => ({
        ...result,
        jpeg: result.jpeg && typeof result.jpeg === 'string' 
              ? `${imageBase}/${result.jpeg.split('/').slice(-2).join('/')}` 
              : null
      }));

      searchStore.update(store => ({
        ...store,
        results: results,
        loading: false
      }));

      if (userTracker.hasConsent()) {
        userTracker.logSearch(query, search_type, filters);
      }
    } catch (err) {
      console.error('Search error in store:', err);
      searchStore.update(store => ({
        ...store,
        error: err.message || 'Search failed',
        loading: false,
        results: []
      }));
    }
  }
};
