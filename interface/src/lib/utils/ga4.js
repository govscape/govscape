const GA4_CONFIG = {
  MEASUREMENT_ID: 'G-ERVV8VCXEX',
  COOKIE_EXPIRES_DAYS: 365,
};

function shouldEnableGA4() {
  return typeof window !== 'undefined' &&
         window.location.hostname !== 'localhost' &&
         window.location.hostname !== '127.0.0.1';
}

export function loadGA4Script() {
  return new Promise((resolve, reject) => {
    if (!shouldEnableGA4()) {
      console.warn('GA4 script is not enabled!');
      resolve();
      return;
    }

    window.dataLayer = window.dataLayer || [];
    window.gtag = function() { window.dataLayer.push(arguments); };
    window.gtag('js', new Date());
    
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${GA4_CONFIG.MEASUREMENT_ID}`;
    
    script.onload = () => {
      resolve();
    };
    
    script.onerror = () => {
      reject(new Error('Failed to load GA4 script'));
    };
    
    document.head.appendChild(script);
  });
}

export function setGA4Config() {
  window.gtag('config', GA4_CONFIG.MEASUREMENT_ID, {
    'cookie_expires': GA4_CONFIG.COOKIE_EXPIRES_DAYS * 24 * 60 * 60,
    'anonymize_ip': true
  });
}

export function trackGA4Search(searchData) {
  const {
    query,
    searchType,
    filters = {},
  } = searchData;

  const eventData = {
    'search_term': query.substring(0, 100),
    'search_type': searchType,
  };

  if (filters && Object.keys(filters).length > 0) {
    eventData.applied_filters = JSON.stringify(filters).substring(0, 100);
  }

  window.gtag('event', 'search', eventData);
}
