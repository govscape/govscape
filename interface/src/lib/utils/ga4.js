import { camelToSnake } from './fetch.js';

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

    window.gtag('consent', 'default', {
      'ad_storage': 'denied',
      'ad_user_data': 'denied',
      'ad_personalization': 'denied',
      'analytics_storage': 'denied',
      'functionality_storage': 'granted',
      'personalization_storage': 'granted',
      'security_storage': 'granted',
    });

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

export function updateGA4Consent(consentSettings) {
  if (!shouldEnableGA4() || typeof window.gtag !== 'function') {
    return;
  }

  window.gtag('consent', 'update', {
    'ad_storage': consentSettings.ad_storage || 'denied',
    'ad_user_data': consentSettings.ad_user_data || 'denied',
    'ad_personalization': consentSettings.ad_personalization || 'denied',
    'analytics_storage': consentSettings.analytics_storage || 'denied',
  });
}

export function grantAnalyticsConsent() {
  updateGA4Consent({
    analytics_storage: 'granted',
  });
}

export function trackGA4Search(searchData) {
  const {
    query,
    searchType,
    filters = {},
  } = searchData;

  const eventData = {
    'search_term': (query || '').toString().substring(0, 100),
    'search_type': searchType,
  };

  if (filters && Object.keys(filters).length > 0) {
    const snakeFilters = camelToSnake(filters);
    eventData.applied_filters = JSON.stringify(snakeFilters).substring(0, 100);
  }

  window.gtag('event', 'search', eventData);
}

export function trackGA4PdfClick(data) {
  const {
    id,
    page,
    subDomain,
    crawlUrl,
    crawlDate,
  } = data || {};

  const eventData = {
    'pdf_id': (id ?? '').toString().substring(0, 120),
    'page_number': Number(page) || 0,
  };

  if (subDomain) {
    eventData.sub_domain = subDomain.substring(0, 100);
  }

  if (crawlUrl) {
    eventData.crawl_url = crawlUrl.toString().substring(0, 200);
  }

  if (crawlDate) {
    eventData.crawl_date = crawlDate.toString().substring(0, 50);
  }

  window.gtag('event', 'pdf_click', eventData);
}

export function trackGA4Pagination(data) {
  const {
    query,
    searchType,
    filters = {},
    page,
  } = data || {};

  const eventData = {
    'search_term': (query || '').toString().substring(0, 100),
    'search_type': searchType,
    'page': Number(page) || 1,
  };

  if (filters && Object.keys(filters).length > 0) {
    const snakeFilters = camelToSnake(filters);
    eventData.applied_filters = JSON.stringify(snakeFilters).substring(0, 100);
  }

  window.gtag('event', 'pagination', eventData);
}
