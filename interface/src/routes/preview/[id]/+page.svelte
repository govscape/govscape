<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { apiFetch, getImageBaseUrl } from '$lib/utils/fetch';

  export let data;
  let id = data.id;

  let images = [];
  let loading = false;
  let error = null;
  let currentPageIndex = 0;
  let totalPages = 0;
  let isLinkCopied = false;
  let pdfSubDomain = '';
  let pdfCrawlDate = '';
  let pdfCrawlUrl = '';

  async function fetchImages() {
    if (!id) return;
    loading = true;
    error = null;
    images = [];

    try {
      const data = await apiFetch(`/pages/${id}`, { method: 'GET' });
      const imageBase = getImageBaseUrl();

      images = (data.images || []).map(img => {
        const parts = img.split('/');
        return `${imageBase}/${parts.slice(-2).join('/')}`;
      });
      totalPages = images.length;
      if (!pdfSubDomain) pdfSubDomain = data.sub_domain || '';
      if (!pdfCrawlDate) pdfCrawlDate = data.crawl_date || '';
      if (!pdfCrawlUrl) pdfCrawlUrl = data.crawl_url || '';

      const p = parseInt($page.url.searchParams.get('page') || '1', 10);
      const idx = Number.isFinite(p) ? Math.max(0, Math.min(p - 1, Math.max(totalPages - 1, 0))) : 0;
      currentPageIndex = idx;
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  function updatePageInURL(pageNum) {
    const params = new URLSearchParams($page.url.searchParams);
    if (pageNum > 1) params.set('page', String(pageNum));
    else params.delete('page');
    const newUrl = params.toString() ? `/preview/${id}?${params.toString()}` : `/preview/${id}`;
    goto(newUrl, { replaceState: true, noScroll: true });
  }

  function nextPage() {
    if (currentPageIndex < totalPages - 1) {
      currentPageIndex += 1;
      updatePageInURL(currentPageIndex + 1);
    }
  }

  function prevPage() {
    if (currentPageIndex > 0) {
      currentPageIndex -= 1;
      updatePageInURL(currentPageIndex + 1);
    }
  }

  function selectThumbnail(i) {
    currentPageIndex = i;
    updatePageInURL(i + 1);
  }

  $: if ($page?.url) {
    const p = parseInt($page.url.searchParams.get('page') || '1', 10);
    const idx = Number.isFinite(p) ? Math.max(0, Math.min(p - 1, Math.max(totalPages - 1, 0))) : 0;
    if (idx !== currentPageIndex) currentPageIndex = idx;
    const sd = $page.url.searchParams.get('sub_domain') || $page.url.searchParams.get('subDomain') || '';
    const cd = $page.url.searchParams.get('crawl_date') || $page.url.searchParams.get('crawlDate') || '';
    const cu = $page.url.searchParams.get('crawl_url') || $page.url.searchParams.get('crawlUrl') || '';
    if (sd) pdfSubDomain = sd;
    if (cd) pdfCrawlDate = cd;
    if (cu) pdfCrawlUrl = cu;
  }

  async function shareLink() {
    const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
    const params = new URLSearchParams();
    if (currentPageIndex > 0) params.set('page', String(currentPageIndex + 1));
    if (pdfSubDomain) params.set('sub_domain', pdfSubDomain);
    if (pdfCrawlDate) params.set('crawl_date', pdfCrawlDate);
    if (pdfCrawlUrl) params.set('crawl_url', pdfCrawlUrl);
    const previewUrl = `${baseUrl}/preview/${id}${params.toString() ? `?${params.toString()}` : ''}`;
    try {
      await navigator.clipboard.writeText(previewUrl);
    } catch (e) {
      try {
        const ta = document.createElement('textarea');
        ta.value = previewUrl;
        ta.style.position = 'fixed';
        ta.style.left = '-999999px';
        ta.style.top = '-999999px';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      } catch {}
    }
    isLinkCopied = true;
    setTimeout(() => { isLinkCopied = false; }, 2000);
  }

  async function downloadPDF() {
    if (!id) return;
    const s3Url = `https://bcgl-public-bucket.s3.amazonaws.com/archive/2020/PDFs/${id}.pdf`;
    try {
      const response = await fetch(s3Url);
      if (!response.ok) throw new Error('Failed to download PDF');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = id.split('/').pop() + '.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      alert('Error downloading PDF: ' + e.message);
    }
  }

  onMount(fetchImages);
</script>

<main>
  {#if loading}
    <div class="page-spinner-container" aria-label="Loading preview">
      <div class="loader"></div>
    </div>
  {:else if error}
  {:else}
    <div class="preview-page">

      <div class="preview-layout-grid">
        <div class="preview-main-content">
          <div class="preview-main-image-container">
            <img src={images[currentPageIndex]} alt={`Page ${currentPageIndex + 1}`} class="preview-main-image" />
          </div>
          <div class="page-navigation">
            <button class="carousel-nav prev" on:click={prevPage} disabled={currentPageIndex === 0} aria-label="Previous page">
              <i class="bi bi-chevron-left"></i>
            </button>
            <span class="page-number">Page {currentPageIndex + 1} of {totalPages}</span>
            <button class="carousel-nav next" on:click={nextPage} disabled={currentPageIndex === totalPages - 1} aria-label="Next page">
              <i class="bi bi-chevron-right"></i>
            </button>
          </div>
        </div>

        <aside class="preview-sidebar">
          <div class="preview-details">
            <h5 class="modal-title">{(pdfCrawlUrl && pdfCrawlUrl.split('/').pop().replaceAll("\%20", " ")) || (id && id.split('/').pop().replaceAll("%20", " "))}</h5>
            <div><b>Sub-Domain:</b> {pdfSubDomain || 'Not Available'}</div>
            <div><b>Crawl Date:</b> {pdfCrawlDate || 'Not Available'}</div>
            <div>
              <b>Crawl URL:</b>
              {#if pdfCrawlUrl}
                <a href={pdfCrawlUrl}>{pdfCrawlUrl}</a>
              {:else}
                Not Available
              {/if}
            </div>
            <div class="action-buttons">
              <button class="btn btn-primary" on:click={downloadPDF}>
                <i class="bi bi-download"></i>
                Download PDF
              </button>
              <button class="btn btn-secondary share-btn" class:copied={isLinkCopied} on:click={shareLink} disabled={isLinkCopied}>
                {#if isLinkCopied}
                  <i class="bi bi-clipboard-check"></i>
                  Link Copied
                {:else}
                  <i class="bi bi-share"></i>
                  Share Link
                {/if}
              </button>
            </div>
          </div>
          <div class="preview-thumbnail-panel">
            <h6 class="preview-thumbnail-panel-title">All Pages</h6>
            <div class="preview-thumbnail-grid">
              {#each images as img, i}
                <img src={img} alt={`Page ${i + 1}`} class="preview-thumbnail-item" class:active={i === currentPageIndex} on:click={() => selectThumbnail(i)} />
              {/each}
            </div>
          </div>
        </aside>
      </div>
    </div>
  {/if}
</main>

<style>
  :root {
    --preview-border-color: #e0e4e8;
    --preview-spacing-unit: 1rem;
    --preview-border-radius: 8px;
  }

  main {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: calc(100vh - 50px);
    padding-top: 50px;
  }

  .page-spinner-container {
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

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .preview-page {
    display: flex;
    flex-direction: column;
    gap: calc(var(--preview-spacing-unit) * 1.25);
    padding: calc(var(--preview-spacing-unit) * 1.5);
  }

  .preview-header {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .back-button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #fff;
    border: 1px solid var(--preview-border-color);
    cursor: pointer;
  }

  .preview-title {
    flex: 1;
    margin: 0;
    font-size: 1.2rem;
    color: var(--text-color-primary);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s ease;
  }

  .btn-primary {
    background-color: var(--color-primary);
    color: #fff;
    border-color: var(--color-primary);
  }

  .btn-secondary {
    background-color: #fff;
    color: var(--color-primary);
    border-color: var(--color-primary);
  }

  .btn:disabled {
    cursor: default;
  }

  .share-btn.copied {
    background-color: var(--color-primary);
    color: #fff;
    border-color: var(--color-primary);
  }

  .preview-layout-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
    gap: calc(var(--preview-spacing-unit) * 1.5);
    align-items: flex-start;
    width: 100%;
  }

  .preview-main-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--preview-spacing-unit);
    width: 100%;
  }

  .preview-main-image-container {
    width: 100%;
    border: 1px solid var(--preview-border-color);
    border-radius: var(--preview-border-radius);
    background-color: #f8f9fa;
  }

  .preview-main-image {
    display: block;
    width: 100%;
    height: auto;
    max-height: 80vh;
    object-fit: contain;
  }

  .page-navigation {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: var(--preview-spacing-unit);
    padding: calc(var(--preview-spacing-unit) * 0.5) 0;
    width: 100%;
  }

  .page-number {
    font-size: 0.95rem;
    color: var(--text-color-secondary);
  }

  .carousel-nav {
    background: none;
    border: 1px solid var(--preview-border-color);
    color: var(--text-color-primary);
    background-color: #fff;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: var(--preview-border-radius);
  }

  .carousel-nav:disabled {
    color: var(--preview-border-color);
    cursor: not-allowed;
  }

  .preview-sidebar {
    display: flex;
    flex-direction: column;
    gap: calc(var(--preview-spacing-unit) * 1.5);
    height: 100%;
  }

  .preview-details {
    font-size: 0.9rem;
    color: var(--text-color-primary);
    padding: var(--preview-spacing-unit) calc(var(--preview-spacing-unit) * 1.25);
    border-radius: var(--preview-border-radius);
    border: 1px solid var(--preview-border-color);
    overflow-wrap: anywhere;
    word-wrap: break-word;
    word-break: break-word;
  }

  .preview-details .modal-title {
    font-size: 1.1rem;
    margin: 0 0 0.5rem 0;
  }

  .preview-details a {
    overflow-wrap: anywhere;
    word-wrap: break-word;
    word-break: break-word;
  }

  .preview-details div {
    margin-bottom: calc(var(--preview-spacing-unit) * 0.5);
  }

  .preview-details div:last-child {
    margin-bottom: 0;
  }

  .preview-details b {
    font-weight: 600;
    color: var(--text-color-primary);
  }

  .action-buttons {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 1rem;
  }

  .action-buttons .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border: none;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
  }

  .action-buttons .btn:active {
    opacity: 0.8;
  }

  .action-buttons .btn-primary {
    background-color: var(--color-primary);
    color: #fff;
  }

  .action-buttons .btn-secondary {
    background-color: #fff;
    border: 1px solid var(--color-primary);
    color: var(--color-primary);
  }

  .action-buttons .btn-secondary.copied {
    background-color: var(--color-primary);
    border-color: var(--color-primary);
    color: #fff;
  }

  .action-buttons .btn:disabled {
    cursor: default;
  }

  /* Thumbnails */
  .preview-thumbnail-panel {
    background: #fff;
    border-radius: var(--preview-border-radius);
    border: 1px solid var(--preview-border-color);
    padding: 16px;
    display: flex;
    flex-direction: column;
  }

  .preview-thumbnail-panel .preview-thumbnail-panel-title {
    font-size: 0.9rem;
    color: var(--text-color-primary);
    margin: 0 4px 8px 6px;
  }

  .preview-thumbnail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: var(--preview-spacing-unit);
    overflow-y: auto;
    flex-grow: 1;
    padding: calc(var(--preview-spacing-unit) * 0.25);
  }

  .preview-thumbnail-item {
    width: 100%;
    aspect-ratio: 76 / 104;
    height: auto;
    object-fit: cover;
    border-radius: calc(var(--preview-border-radius) * 0.75);
    border: 2px solid var(--preview-border-color);
    background: #f8f9fa;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    cursor: pointer;
    transition: all 0.2s ease-in-out;
  }

  .preview-thumbnail-item:hover {
    border-color: var(--color-primary);
    box-shadow: 0 2px 8px rgba(0, 123, 255, 0.1);
  }

  .preview-thumbnail-item.active {
    border-color: var(--color-primary);
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.2);
    transform: scale(1.05);
    background: #eaf6ff;
    z-index: 2;
  }

  @media (max-width: 767px) {
    .preview-layout-grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto;
    }

    .preview-main-content {
      order: 1;
    }

    .preview-sidebar {
      order: 2;
    }
  }
</style>
