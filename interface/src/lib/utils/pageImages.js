import { getImageBaseUrl } from './fetch';

// Page images are stored many to an object ("packs", see DATA_MODEL.md) and
// fetched by byte range; the API gives each page's range as {pack, start, end}.
// Serving directories from before packs have one JPEG per page, given as a path.
// A page's image source is either {url, start, end} or a plain image URL.

export function pageImageSource(range, jpegPath) {
  const base = getImageBaseUrl();
  if (range) return { url: `${base}/${range.pack}`, start: range.start, end: range.end };
  if (jpegPath) return `${base}/${jpegPath.split('/').slice(-2).join('/')}`;
  return null;
}

// The sources of every page of a document, from the API's /pages response.
export function documentPageSources(pagesResponse) {
  if (pagesResponse.pageImageRanges?.length) {
    return pagesResponse.pageImageRanges.map(range => pageImageSource(range));
  }
  return (pagesResponse.images || []).map(path => pageImageSource(null, path));
}

// Object URLs of fetched pages, most recently used last. Evicted URLs are
// revoked; an <img> already showing one keeps its decoded image.
const MAX_CACHED_PAGES = 300;
const cache = new Map();

function cacheKey(source) {
  return `${source.url}#${source.start}`;
}

function remember(key, objectUrl) {
  cache.delete(key);
  cache.set(key, objectUrl);
  if (cache.size > MAX_CACHED_PAGES) {
    const [oldestKey, oldest] = cache.entries().next().value;
    cache.delete(oldestKey);
    oldest.then(url => URL.revokeObjectURL(url), () => {});
  }
  return objectUrl;
}

async function fetchRange(url, start, end) {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (response.status !== 206) {
    throw new Error(`Range request for ${url} returned HTTP ${response.status}`);
  }
  return response.arrayBuffer();
}

function toObjectUrl(bytes) {
  return URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
}

// Resolves to a URL an <img> can show.
export function loadPageImage(source) {
  if (typeof source === 'string') return Promise.resolve(source);
  const key = cacheKey(source);
  const cached = cache.get(key);
  if (cached) return remember(key, cached);
  const objectUrl = fetchRange(source.url, source.start, source.end).then(toObjectUrl);
  objectUrl.catch(() => cache.delete(key));
  return remember(key, objectUrl);
}

// A document's pages are contiguous in their pack, so the pages around `index`
// come in one request instead of one per page.
export function prefetchPages(sources, index, before = 2, after = 20) {
  const nearby = sources
    .slice(Math.max(0, index - before), index + after + 1)
    .filter(source => source && typeof source !== 'string' && !cache.has(cacheKey(source)));
  if (nearby.length < 2 || nearby.some(source => source.url !== nearby[0].url)) return;
  const start = nearby[0].start;
  const bytes = fetchRange(nearby[0].url, start, nearby[nearby.length - 1].end);
  bytes.catch(() => {});
  for (const source of nearby) {
    const key = cacheKey(source);
    const objectUrl = bytes.then(all =>
      toObjectUrl(all.slice(source.start - start, source.end - start + 1))
    );
    objectUrl.catch(() => cache.delete(key));
    remember(key, objectUrl);
  }
}

// Svelte action: <img use:pageImage={source}> shows the page once the image
// scrolls near the viewport, like loading="lazy".
export function pageImage(img, source) {
  let current;
  let observer;

  function show(next) {
    current = next;
    observer?.disconnect();
    if (!next) {
      img.removeAttribute('src');
      return;
    }
    observer = new IntersectionObserver(
      entries => {
        if (!entries.some(entry => entry.isIntersecting)) return;
        observer.disconnect();
        loadPageImage(next).then(
          url => {
            if (current === next) img.src = url;
          },
          error => console.warn(`Couldn't load page image: ${error.message}`)
        );
      },
      { rootMargin: '200px' }
    );
    observer.observe(img);
  }

  show(source);
  return {
    update: show,
    destroy() {
      observer?.disconnect();
      current = null;
    }
  };
}
