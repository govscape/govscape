<script>
  export let width = 20;
  export let height = 20;
</script>

<svg
  xmlns="http://www.w3.org/2000/svg"
  {width}
  {height}
  viewBox="0 0 40 40"
  fill="none"
  stroke="currentColor"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
>
  <circle cx="20" cy="20" r="16" />
  <ellipse cx="20" cy="20" rx="6" ry="16" />
  <line x1="4" y1="20" x2="36" y2="20" />
  <line x1="20" y1="4" x2="20" y2="36" />
  <line x1="6" y1="13" x2="34" y2="13" />
  <line x1="6" y1="27" x2="34" y2="27" />
</svg>
