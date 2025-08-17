<main>
  <div class="about-container">
    <div class="about-header">
      <h1>About GovScape</h1>
    </div>
    <div class="about-content">
      <p class="description">
        GovScape is a project by the University of Washington and is coming soon. In the meantime, please contact Ben Lee with any questions.
      </p>
      <div class="contact-section">
        <div class="contact-section-header">Contact</div>
        <p>For questions or inquiries about GovScape, please contact:</p>
        <a href="mailto:bcgl@uw.edu" class="contact-link">
          <i class="bi bi-envelope"></i>
          Ben Lee (bcgl@uw.edu)
        </a>
      </div>
    </div>
  </div>
</main>

<style>
  main {
    min-height: calc(100vh - 90px);
    padding: 80px 2rem 0 2rem;
  }

  .about-container {
    max-width: 700px;
    width: 100%;
    background: #fff;
    padding: 0.5rem 1.5rem 1.5rem 1.5rem;
    margin: 2rem auto;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }

  .about-header {
    text-align: center;
    margin: 1.5rem 0;

    & > h1 {
      color: var(--color-primary);
      font-size: 2rem;
      font-weight: 700;
    }
  }

  .about-content .description {
    font-size: 1rem;
    line-height: 1.8;
    color: var(--text-color-primary);
    margin-bottom: 1.5rem;
    padding: 0 2rem;
  }

  .contact-section {
    background: var(--background-color-primary);
    padding: 2rem;
    border-radius: 12px;
  }

  .contact-section .contact-section-header {
    font-family: var(--sans-serif-font);
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    color: var(--color-primary);
  }

  .contact-link {
    display: inline-flex;
    align-items: center;
    color: var(--color-primary);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
  }

  .contact-link i {
    margin-right: 8px;
  }

  .contact-link:hover {
    color: var(--color-secondary);
  }

  @keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.2);
      opacity: 0.7;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  }
</style>
