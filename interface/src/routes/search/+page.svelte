<script>
  // AI modified: 2026-03-14 4a6b1b72
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { searchStore, searchActions } from '$lib/stores/search';
  import SearchBox from '$lib/components/SearchBox.svelte';
  import ResultsGrid from '$lib/components/ResultsGrid.svelte';
  import PDFPreview from '$lib/components/PDFPreview.svelte';
  import { goto } from '$app/navigation';

  let shouldShowPreview = false;
  let selectedPDF = null;

  function handlePDFSelect(event) {
    const { id, page, crawlDate, crawlUrl, subDomain, crawlInstances, hasMoreCrawls, prettyName } = event.detail || {};
    selectedPDF = { id, page, crawlDate, crawlUrl, subDomain, crawlInstances, hasMoreCrawls, prettyName };
    shouldShowPreview = true;
  }

  function handleClosePreview() {
    shouldShowPreview = false;
    selectedPDF = null;
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
  }
</style>
