const IS_DEV = import.meta.env.DEV;

// Default API request timeout (60 seconds)
const DEFAULT_API_TIMEOUT_MS = Number(60000);

const ENDPOINTS = {
  DEV: {
    API: 'http://localhost:8080/api',
    S3: 'https://bcgl-public-bucket.s3.amazonaws.com/dev-serving/img'
  },
  PROD: {
    API: 'https://govscape.net/api',
    S3: 'https://bcgl-public-bucket.s3.amazonaws.com/prod-serving/img'
  }
};

export const getApiBaseUrl = () => {
  if (IS_DEV) return ENDPOINTS.DEV.API;

  return ENDPOINTS.PROD.API;
};

export const getImageBaseUrl = () => {
  if (IS_DEV) return ENDPOINTS.DEV.S3;

  return ENDPOINTS.PROD.S3;
};

function snakeToCamel(obj) {
    if (obj === null || obj === undefined) return obj;
    if (typeof obj !== 'object') return obj;
    if (Array.isArray(obj)) return obj.map(snakeToCamel);

    return Object.keys(obj).reduce((acc, key) => {
        const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
        acc[camelKey] = snakeToCamel(obj[key]);
        return acc;
    }, {});
}

export function camelToSnake(obj) {
    if (obj === null || obj === undefined) return obj;
    if (typeof obj !== 'object') return obj;
    if (Array.isArray(obj)) return obj.map(camelToSnake);

    return Object.keys(obj).reduce((acc, key) => {
        const snakeKey = key.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
        acc[snakeKey] = camelToSnake(obj[key]);
        return acc;
    }, {});
}

export async function apiFetch(endpoint, options = {}) {
    const defaultOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'include',
    };

    const {
        timeoutMs = DEFAULT_API_TIMEOUT_MS,
        signal: externalSignal,
        ...restOptions
    } = options || {};

    const mergedOptions = {
        ...defaultOptions,
        ...restOptions,
        headers: {
            ...defaultOptions.headers,
            ...restOptions.headers,
        },
    };

    if (mergedOptions.body && typeof mergedOptions.body === 'string') {
        try {
            const parsed = JSON.parse(mergedOptions.body);
            mergedOptions.body = JSON.stringify(camelToSnake(parsed));
        } catch (e) {
        }
    }

    try {
        const apiUrl = getApiBaseUrl();
        const controller = !externalSignal ? new AbortController() : null;
        const timeoutId = !externalSignal
            ? setTimeout(() => controller.abort(), Math.max(0, timeoutMs))
            : null;

        const response = await fetch(`${apiUrl}${endpoint}`, {
            ...mergedOptions,
            signal: externalSignal || (controller && controller.signal) || undefined,
        });

        if (timeoutId) clearTimeout(timeoutId);

        if (!response.ok) {
            const contentType = response.headers.get('content-type');
            
            if (response.status >= 500) {
                const errorMessages = {
                    502: 'The service is temporarily unavailable. Please try again in a moment.',
                    503: 'The service is temporarily down for maintenance. Please try again later.',
                    504: 'The request took too long to process. Please try again.',
                };
                throw new Error(errorMessages[response.status] || 'Server error. Please try again later.');
            }
            
            // For client errors (4xx), try to get the error message
            // But check if response is HTML (which we don't want to display)
            if (contentType && contentType.includes('text/html')) {
                throw new Error(`Request failed with status ${response.status}. Please try again.`);
            }
            
            // For JSON or plain text errors, get the actual error message
            const errorText = await response.text();
            throw new Error(errorText || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        return snakeToCamel(data);
    } catch (error) {
        if (error?.name === 'AbortError') {
            throw new Error('Request timed out');
        }
        console.error('API request failed:', error);
        throw error;
    }
}
