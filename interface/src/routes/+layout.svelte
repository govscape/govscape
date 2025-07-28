<script>
  import { onMount } from 'svelte';
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
  <slot />
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
</style>
