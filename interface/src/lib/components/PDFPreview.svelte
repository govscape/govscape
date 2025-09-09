<script>
  import { createEventDispatcher, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import { searchStore } from '$lib/stores/search';
  import { apiFetch, getImageBaseUrl } from '../utils/fetch';

  export let show = false;
  export let pdfData = null;

  const dispatch = createEventDispatcher();

  let images = [];
  let loading = false;
  let error = null;
  let currentPageIndex = 0;
  let totalPages = 0;

  async function fetchImages() {
    if (!pdfData?.id) return;

    loading = true;
    error = null;
    images = [];

    try {
      const data = await apiFetch(`/pages/${pdfData.id}`, { method: 'GET' });
      const imageBase = getImageBaseUrl();
      
      images = (data.images || []).map(img => {
        const parts = img.split('/');
        return `${imageBase}/${parts.slice(-2).join('/')}`;
      });
      totalPages = images.length;
      currentPageIndex = parseInt(pdfData.page);
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  $: if (typeof window !== 'undefined') {
    if (show) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
  }

  $: if (show && pdfData) {
    fetchImages();
  }

  function closeModal() {
    show = false;
    dispatch('close');
  }

  function nextPage() {
    currentPageIndex++;
  }

  function prevPage() {
    currentPageIndex--;
  }

  async function sharePreview() {
    if (!pdfData?.id || isLinkCopied) return;
    
    const baseUrl = window.location.origin;
    const previewUrl = `${baseUrl}/preview/${pdfData.id}${currentPageIndex > 0 ? `?page=${currentPageIndex + 1}` : ''}`;
    
    try {
      // Try to use the modern Clipboard API
      await navigator.clipboard.writeText(previewUrl);
      showShareFeedback();
    } catch (err) {
      // Fallback for older browsers
      try {
        const textArea = document.createElement('textarea');
        textArea.value = previewUrl;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        showShareFeedback();
      } catch (fallbackErr) {
        // If all else fails, still show the button feedback
        showShareFeedback();
      }
    }
  }

  let isLinkCopied = false;
  
  function showShareFeedback(message, isError = false) {
    isLinkCopied = true;
    
    // Reset button text after 2 seconds
    setTimeout(() => {
      isLinkCopied = false;
    }, 2000);
  }

  async function downloadPDF() {
    if (!pdfData?.id) return;
    // Construct the S3 URL
    const s3Url = `https://bcgl-public-bucket.s3.amazonaws.com/archive/2020/PDFs/${pdfData.id}.pdf`;
    try {
      const response = await fetch(s3Url);
      if (!response.ok) throw new Error('Failed to download PDF');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      // Create a temporary link to trigger download
      const a = document.createElement('a');
      a.href = url;
      a.download = pdfData.id.split('/').pop() + '.pdf' || 'document.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      alert('Error downloading PDF: ' + e.message);
    }
  }

  onDestroy(() => {
    if (typeof window !== 'undefined') {
      document.body.style.overflow = '';
    }
  });
</script>

{#if show}
  <div class="modal-backdrop" on:click={closeModal}>
    <div class="modal-content" on:click|stopPropagation>
      <div class="modal-header">
        <h5 class="modal-title">{pdfData?.crawl_url?.split('/').pop().replaceAll("\%20", " ") || ''}</h5>
        <button class="btn-close" on:click={closeModal}>
          <i class="bi bi-x"></i>
        </button>
      </div>
      <div class="modal-body">
        <div class="preview-layout-grid">
          <div class="preview-main-content">
            <div class="preview-main-image-container">
              <img src={images[currentPageIndex]} alt={`Page ${currentPageIndex + 1}`} class="preview-main-image" />
            </div>
            <div class="page-navigation">
              <button class="carousel-nav prev" on:click={prevPage} disabled={currentPageIndex === 0}>
                <i class="bi bi-chevron-left"></i>
              </button>
              <span class="page-number">
                Page {currentPageIndex + 1} of {totalPages}
              </span>
              <button class="carousel-nav next" on:click={nextPage} disabled={currentPageIndex === totalPages - 1}>
                <i class="bi bi-chevron-right"></i>
              </button>
            </div>
          </div>
          <aside class="preview-sidebar">
            <div class="preview-details">
              <div><b>Sub-Domain:</b> {pdfData?.sub_domain || 'Not Available'}</div>
              <div><b>Crawl Date:</b> {pdfData?.crawl_date || 'Not Available'}</div>
              <div><b>Crawl URL:</b> <a href={pdfData?.crawl_url || 'Not Available'}>{pdfData?.crawl_url || 'Not Available'}</a></div>
              <div class="action-buttons">
                <button class="btn btn-primary" on:click={downloadPDF}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7,10 12,15 17,10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download PDF
                </button>
                <button class="btn btn-secondary share-btn" class:copied={isLinkCopied} on:click={sharePreview} disabled={isLinkCopied}>
                  {#if isLinkCopied}
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polyline points="20,6 9,17 4,12"/>
                    </svg>
                    Link Copied
                  {:else}
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>
                      <polyline points="16,6 12,2 8,6"/>
                      <line x1="12" y1="2" x2="12" y2="15"/>
                    </svg>
                    Share Link
                  {/if}
                </button>
              </div>
            </div>
            <div class="preview-thumbnail-panel">
              <h6 class="preview-thumbnail-panel-title">All Pages</h6>
              <div class="preview-thumbnail-grid">
                {#each images as img, i}
                  <img src={img} alt={`Page ${i + 1}`} class="preview-thumbnail-item" class:active={i === currentPageIndex} on:click={() => currentPageIndex = i} />
                {/each}
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  :root {
    --preview-border-color: #e0e4e8;
    --preview-spacing-unit: 1rem;
    --preview-border-radius: 8px;
  }

  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1050;
    padding: var(--preview-spacing-unit);
  }

  .modal-content {
    background: #fff;
    width: 90%;
    max-width: 1300px;
    max-height: calc(100vh - (var(--preview-spacing-unit) * 2));
    border-radius: var(--preview-border-radius);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .modal-header {
    padding: 10px 12px 10px 16px;
    border-bottom: 1px solid var(--preview-border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }

  .modal-title {
    font-size: 1.2rem;
    color: var(--text-color-primary);
  }

  .btn-close {
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    font-size: 2rem;
  }

  .btn-close:hover {
    opacity: 1;
  }

  .modal-body {
    padding: calc(var(--preview-spacing-unit) * 1.5);
    overflow-y: auto;
    flex-grow: 1;
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
    max-height: 70%;
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
    border: none;
    color: var(--text-color-primary);
    cursor: pointer;
    padding: 4px 8px;
    border-radius: var(--preview-border-radius);
  }

  .carousel-nav:disabled {
    color: var(--preview-border-color);
    cursor: not-allowed;
  }

  .carousel-nav:not(:disabled):hover {
    background-color: rgba(0, 123, 255, 0.1);
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

    .modal-content {
      width: 95%;
      max-height: 90%;
    }
  }
</style>
