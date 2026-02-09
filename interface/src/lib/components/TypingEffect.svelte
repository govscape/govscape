<script>
  import { onMount, onDestroy } from 'svelte';

  export let words = [];
  export let typingSpeed = 100;
  export let deletingSpeed = 50;
  export let delayBetweenWords = 2000;

  let text = '';
  let wordIndex = 0;
  let isDeleting = false;
  let timer;

  function tick() {
    if (!words.length) return;

    const currentWord = words[wordIndex];

    if (isDeleting) {
      text = text.slice(0, -1);
      if (text === '') {
        isDeleting = false;
        wordIndex = (wordIndex + 1) % words.length;
        timer = setTimeout(tick, typingSpeed);
      } else {
        timer = setTimeout(tick, deletingSpeed);
      }
    } else {
      if (text.length < currentWord.length) {
        text = currentWord.slice(0, text.length + 1);
        timer = setTimeout(tick, typingSpeed);
      } else if (text === currentWord) {
        timer = setTimeout(() => {
          isDeleting = true;
          tick();
        }, delayBetweenWords);
      }
    }
  }

  onMount(() => {
    if (words.length) {
      timer = setTimeout(tick, typingSpeed);
    }
  });

  onDestroy(() => {
    if (timer) clearTimeout(timer);
  });
</script>

<span class="highlighted-domain">
  {text}<span class="cursor">|</span>
</span>

<style>
  .cursor {
    animation: blink 1.5s step-end infinite;
  }

  .highlighted-domain {
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 24px;
    padding: 4px 8px 4px 12px;
  }

  @keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
  }
</style>
