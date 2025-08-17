<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import CookieConsent from '$lib/components/CookieConsent.svelte';
  import { userTracker } from '$lib/utils/userTracking.js';

  let showCookieConsent = false;

  function handleCookieConsent(event) {
    const { accepted } = event.detail;

    userTracker.updateConsent(accepted);
    showCookieConsent = false;
  }

  onMount(() => {
    showCookieConsent = userTracker.needsConsent();
  });
</script>

<div class="app">
  <header>
    <a href="/" class="logo">
      <img draggable="false" src="/logo.png" alt="GovScape Logo" class="logo-image" />
    </a>
    <nav>
      <a href="/about">About</a>
      <a href="/faq">FAQ</a>
    </nav>
  </header>
  {#key $page.url.pathname}
    <slot />
  {/key}
  <footer>
    <div class="footer-content">
      <p class="main-text">GovScape is coming soon</p>
      <p class="contact-text">
        For questions, please visit our <a href="/about">About</a> page
      </p>
    </div>
  </footer>
  <CookieConsent 
    show={showCookieConsent}
    on:consent={handleCookieConsent}
  />
</div>

<style>
  header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
    background: #fff;
    border-bottom: 1px solid #e9ecef;
    height: var(--header-height);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 2rem;
  }

  .logo img {
    height: 40px;
    width: auto;
    vertical-align: middle;
  }

  nav {
    display: flex;
    gap: 2rem;
  }

  nav a {
    color: var(--text-color-primary);
    font-size: 0.85rem;
    text-decoration: none;
  }

  footer {
    width: 100%;
    background-color: var(--color-secondary);
    color: white;
    padding: 1rem 0;
  }

  .footer-content {
    max-width: 1200px;
    margin: 0 auto;
    text-align: center;
    font-family: var(--serif-font);
  }

  .footer-content .main-text {
    font-size: 1rem;
    margin-bottom: 0.65rem;
  }

  .footer-content .contact-text {
    font-size: 0.9rem;
    margin: 0;
  }

  .footer-content .contact-text a {
    color: #fff;
  }

  .footer-content .contact-text a:hover {
    opacity: 0.8;
    text-decoration: underline;
  }
</style>
