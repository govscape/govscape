<main class="about-page">
  <div class="about-container">
    <div class="about-header">
      <h1>About GovScape</h1>
    </div>
    
    <div class="about-content">
      <div class="status-badge">
        <span class="dot"></span>
        Coming Soon
      </div>
      
      <div class="description">
        <p>
          GovScape is a project by the University of Washington and is coming soon. In the meantime, please contact Ben Lee with any questions.
        </p>
      </div>

      <div class="contact-section">
        <h2>Contact</h2>
        <p>
          For questions or inquiries about GovScape, please contact:
        </p>
        <a href="mailto:bcgl@uw.edu" class="contact-link">
          <i class="bi bi-envelope"></i>
          Ben Lee (bcgl@uw.edu)
        </a>
      </div>
    </div>
  </div>
</main>

<style>
  .about-page {
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--background-color-primary), white);
  }

  .about-container {
    max-width: 800px;
    width: 100%;
    background: white;
    padding: 3rem;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }

  .about-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  h1 {
    color: var(--color-primary);
    margin-bottom: 1rem;
    font-size: 2.5rem;
    font-weight: 700;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    background: var(--background-color-primary);
    color: var(--color-primary);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 2rem;
  }

  .dot {
    width: 8px;
    height: 8px;
    background: var(--color-primary);
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
  }

  .description {
    font-size: 1.2rem;
    line-height: 1.6;
    color: var(--text-color-primary);
    margin-bottom: 3rem;
  }

  .contact-section {
    background: var(--background-color-primary);
    padding: 2rem;
    border-radius: 12px;
  }

  .contact-section h2 {
    font-size: 1.4rem;
    color: var(--color-primary);
    margin-bottom: 1rem;
  }

  .contact-link {
    display: inline-flex;
    align-items: center;
    color: var(--color-primary);
    text-decoration: none;
    font-weight: 500;
    margin-top: 1rem;
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
