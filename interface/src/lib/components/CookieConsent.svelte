<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, scale } from 'svelte/transition';
  
  const dispatch = createEventDispatcher();
  
  export let show = false;
  
  let dialog;
  let isOver18 = false;
  
  function handleConsent() {
    dispatch('consent', { accepted: true });
    show = false;
  }
  
  function handleDecline() {
    dispatch('consent', { accepted: false });
    show = false;
  }
</script>

{#if show}
  <div class="modal-backdrop" transition:fade={{ duration: 200 }}>
    <div 
      class="modal-content"
      transition:scale={{ duration: 200, start: 0.95 }}
      on:click|stopPropagation
      role="dialog"
      aria-labelledby="consent-title"
      aria-describedby="consent-description"
      bind:this={dialog}
    >
      <h2 id="consent-title">Research Study Consent</h2>
      <div id="consent-description" class="consent-text">
        <p>GovScape is a project created by researchers at the University of Washington in order to improve access to federal documents held within the End of Term Web Archive.</p>
        <p>In order to improve the website and its affordances, as well as study what people are searching for, we are asking users to consent to have their anonymized search data collected for these studies. There will be no personally-identifiable information collected including no IP addresses and no locations.</p>
        <p>We require users to be 18 years of age or older, and therefore, you must attest to being of the proper age by checking a box prior to starting the study.</p>
        <p>You are free to browse as long as you would like. If you agree but decide at any time that you would no longer like to participate, you may simply end your browsing session.</p>
        <p>If you have any questions at all regarding this study procedure, you may email <a href="mailto:bcgl@uw.edu">bcgl@uw.edu</a> at any time.</p>
      </div>
      <div class="age-check">
        <label>
          <input type="checkbox" bind:checked={isOver18}>
          I confirm that I am 18 years of age or older
        </label>
      </div>
      <div class="button-group">
        <button class="decline" on:click={handleDecline}>Decline</button>
        <button class="consent" on:click={handleConsent} disabled={!isOver18}>I Consent</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    justify-content: center;
    z-index: 1050;
  }

  .modal-content {
    background: #fff;
    width: 90%;
    margin-top: 5vh;
    max-width: 600px;
    max-height: 90vh;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    overflow-y: auto;
  }

  .modal-content h2 {
    font-size: 1.8rem;
    color: var(--text-color-primary, #333);
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .consent-text {
    margin-bottom: 2rem;
  }

  .consent-text p {
    color: var(--text-color-secondary, #666);
    margin-bottom: 1rem;
    line-height: 1.6;
  }

  .consent-text a {
    color: #0066cc;
    text-decoration: none;
  }

  .consent-text a:hover {
    text-decoration: underline;
  }

  .age-check {
    margin-bottom: 2rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 4px;
  }

  .age-check label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-color-primary, #333);
    cursor: pointer;
  }

  .age-check input[type="checkbox"] {
    width: 1.2rem;
    height: 1.2rem;
    cursor: pointer;
  }

  .button-group {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
  }

  .button-group button {
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .decline {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    color: var(--text-color-secondary, #666);
  }

  .decline:hover {
    background: #e9ecef;
  }

  .consent {
    background: #0066cc;
    border: none;
    color: white;
  }

  .consent:hover:not(:disabled) {
    background: #0052a3;
  }

  .consent:disabled {
    background: #ccc;
    cursor: not-allowed;
  }
</style>
