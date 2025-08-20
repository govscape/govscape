<script>
  function toggle(e) {
    e.currentTarget.parentElement.classList.toggle('opened');
  }
</script>

<main>
  <h1>Frequently Asked Questions</h1>

  <section class="faq">
    <details class="item">
      <summary on:click={toggle}>
        What is GovScape, and how does it work?
      </summary>
      <div class="content">
        <p>GovScape is a collaboration between the University of Washington's Information School and Paul G. Allen School for Computer Science & Engineering, along with a number of other institutions. GovScape currently supports multimodal search over 4.7 million PDFs (25 million PDF pages) within the End of Term Web Archive's 2020 crawl, completed at the end of the first Trump presidential administration. GovScape supports 3 forms of searching over these PDFs:</p>
        <ul>
          <li>
            <strong>"keyword search," or exact text search:</strong> this form of search is canonical keyword search over document text, i.e., basic keyword search.
          </li>
          <li>
            <strong>"text search," or vectorized natural language text search:</strong> with this form of search, you can define more flexible textual queries such as "budgetary data related to the Iraq war" or "rural healthcare for children," which will return PDF pages ranked based on the relevance of the page's text to your query – even if the exact string is not present. For this search functionality, we leverage embeddings from the <a href="https://huggingface.co/BAAI/bge-base-en-v1.5" target="_blank" rel="noopener noreferrer">BAAI/bge-base-en-v1.5</a> model. Not sure what this means? Give one of the example searches a try!
          </li>
          <li>
            <strong>"visual search" over individual PDF pages:</strong> here, you can try queries like "redacted documents," "pie charts," or "aerial photography" – PDF pages are returned based on the relevance of their visual features to the query. For this search functionality, we leverage CLIP embeddings generated using the <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank" rel="noopener noreferrer">openai/clip-vit-base-patch32</a> model.
          </li>
        </ul>
        <p>
          All three of these search methods can be combined with <strong>metadata filtering</strong> according to <strong>subdomain</strong> and <strong>crawl date</strong>, available by clicking the filter button just to the left of the magnifying glass (which executes the search).
        </p>
        <p class="note">
          In future versions of GovScape, we plan to support searching the PDFs across the End of Term Web Archive, including the 2024 crawl when it is released.
        </p>
        <p>You can learn more about GovScape in our paper available here: [forthcoming].</p>
      </div>
    </details>

    <details class="item">
      <summary on:click={toggle}>
        What is the End of Term Web Archive?
      </summary>
      <div class="content">
        <p>The End of Term Web Archive is a multi-institutional effort to archive and preserve federal .gov and .mil sites at the end of each presidential administration. We are big fans and are grateful to be able to utilize End of Term, which has been made public and free to access through the AWS Open Data Sponsorship Program.</p>
        <p>You can learn more about the End of Term Web Archive here: <a href="https://eotarchive.org/about/" target="_blank" rel="noopener noreferrer">https://eotarchive.org/about/</a></p>
      </div>
    </details>

    <details class="item">
      <summary on:click={toggle}>
        How was GovScape built? How do the search functionalities provided by GovScape work?
      </summary>
      <div class="content">
        <p>We detail our search methods, pre-processing pipeline, and system architecture in our paper available here: [forthcoming].</p>
      </div>
    </details>

    <details class="item">
      <summary on:click={toggle}>
        Where can I find the code for GovScape, and can I re-use any of it?
      </summary>
      <div class="content">
        <p>All code for GovScape is available at the GovScape repository: <a href="https://github.com/bcglee/govscape" target="_blank" rel="noopener noreferrer">https://github.com/bcglee/govscape</a>. We have made our code open source and available via the MIT License. If you end up using the code, we'd love to hear from you!</p>
      </div>
    </details>

    <details class="item">
      <summary on:click={toggle}>
        Who are you?
      </summary>
      <div class="content">
        <p>
          GovScape is led by <a href="https://www.bcglee.com/" target="_blank" rel="noopener noreferrer">Ben Lee</a> (Assistant Professor, University of Washington Information School) and <a href="https://kylebd99.github.io/kyle.deeds.github.io/" target="_blank" rel="noopener noreferrer">Kyle Deeds</a> (Incoming Assistant Professor, Boston University Department of Computer Science). Our full project team brings together an interdisciplinary team of computer scientists, digital humanists, and digital librarians:
        </p>
        <ul class="team">
          <li><a href="https://trevorowens.org/" target="_blank" rel="noopener noreferrer">Trevor Owens</a></li>
          <li><a href="https://facultyinfo.unt.edu/faculty-profile?profile=mep0037" target="_blank" rel="noopener noreferrer">Mark Phillips</a></li>
          <li><a href="https://hinx.me/" target="_blank" rel="noopener noreferrer">Ying-Hsiang Huang</a></li>
          <li><a href="https://www.szj.io/" target="_blank" rel="noopener noreferrer">Shannon Zejiang Shen</a></li>
          <li><a href="https://cyber.harvard.edu/people/sklein" target="_blank" rel="noopener noreferrer">SJ Klein</a></li>
          <li><a href="https://www.linkedin.com/in/claire-gong-597b8b196" target="_blank" rel="noopener noreferrer">Claire Gong</a></li>
          <li><a href="https://www.linkedin.com/in/shreya-shaji" target="_blank" rel="noopener noreferrer">Shreya Shaji</a></li>
          <li><a href="https://www.linkedin.com/in/alisonyan" target="_blank" rel="noopener noreferrer">Alison Yan</a></li>
          <li>Leslie Harka</li>
        </ul>
      </div>
    </details>

    <details class="item">
      <summary on:click={toggle}>
        How can I contact you?
      </summary>
      <div class="content">
        <p>For press inquiries, collaborations, questions, and general feedback, please feel free to email Ben Lee at: <a href="mailto:bcgl@uw.edu">bcgl@uw.edu</a>. We would also love to hear if you've found anything particularly useful with GovScape!</p>
      </div>
    </details>
  </section>
</main>

<style>
  main {
    min-height: calc(100vh - 50px);
    padding: 80px 2rem 2rem 2rem;
    max-width: 800px;
    margin: 0 auto;
  }

  h1 {
    color: var(--color-primary);
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin: 1rem 0 1.5rem 0;
  }

  .faq {
    background-color: #fff;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .faq .item {
    border-bottom: 1px solid #e9ecef;
  }

  .faq .item:last-child {
    border-bottom: none;
  }

  .faq summary {
    position: relative;
    padding: 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-color-primary);
    font-family: var(--sans-serif-font);
    cursor: pointer;
    list-style: none;
  }

  .faq summary::-webkit-details-marker {
    display: none;
  }

  .faq summary::after {
    content: '▼';
    position: absolute;
    top: 1.5rem;
    right: 1.5rem;
    font-size: 0.9rem;
    color: var(--color-primary);
  }

  :global(.faq .item.opened) summary::after {
    transform: rotate(180deg);
  }

  .faq .content {
    padding: 0 1.5rem 1.5rem;
  }

  .faq .content p,
  .faq .content li {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-color-primary);
    margin-bottom: 1rem;
  }

  .faq .content p:last-child {
    margin-bottom: 0;
  }

  .faq .content ul {
    padding-left: 1.5rem;
  }

  .faq .content .note {
    background: var(--background-color-primary);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid var(--color-primary);
  }

  .faq .content a {
    color: var(--color-primary);
    text-decoration: none;
    font-weight: 500;
  }
  .faq .content a:hover {
    color: var(--color-secondary);
    text-decoration: underline;
  }

  .faq .content .team {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.5rem;
    padding-left: 0;
    list-style: none;
  }

  .faq .content .team li {
    margin-bottom: 0;
  }
</style>
