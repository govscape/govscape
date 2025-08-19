<script>
  import { page } from '$app/stores';
</script>

<div class="app">
  <header>
    <a href="/" class="logo">
      <img draggable="false" src="/logo.png" alt="GovScape Logo" class="logo-image" />
    </a>
  </header>
  {#key $page.url.pathname}
    <slot />
  {/key}
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
