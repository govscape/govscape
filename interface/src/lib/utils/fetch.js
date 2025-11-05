const IS_DEV = import.meta.env.DEV;

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

export async function apiFetch(endpoint, options = {}) {
    const defaultOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'include',
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    try {
        const apiUrl = getApiBaseUrl();
        const response = await fetch(`${apiUrl}${endpoint}`, mergedOptions);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}
