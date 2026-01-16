/**
 * API utility with timeout, retry logic, and better error handling
 * 
 * @module utils/api
 * @description Provides robust HTTP request utilities with automatic retry,
 * timeout handling, and exponential backoff for the Neuro Desk application.
 */

const DEFAULT_TIMEOUT = 10000; // 10 seconds
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_DELAY = 1000; // 1 second base delay

/**
 * Sleep for specified milliseconds
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Calculate exponential backoff delay
 */
const getRetryDelay = (attempt, baseDelay = DEFAULT_RETRY_DELAY) => {
  return baseDelay * Math.pow(2, attempt);
};

/**
 * Fetch with timeout and abort controller
 */
const fetchWithTimeout = async (url, options = {}, timeout = DEFAULT_TIMEOUT) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timeout - server took too long to respond');
    }
    throw error;
  }
};

/**
 * Retry fetch with exponential backoff
 * @param {AbortController} externalController - Optional external AbortController
 */
export const fetchWithRetry = async (
  url,
  options = {},
  {
    maxRetries = DEFAULT_MAX_RETRIES,
    timeout = DEFAULT_TIMEOUT,
    retryDelay = DEFAULT_RETRY_DELAY,
    retryableStatuses = [408, 429, 500, 502, 503, 504],
    onRetry = null,
    externalController = null
  } = {}
) => {
  let lastError;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    // Check if external controller aborted
    if (externalController && externalController.signal.aborted) {
      throw new Error('Request was cancelled')
    }
    
    try {
      // Merge external controller signal if provided
      const fetchOptions = externalController 
        ? { ...options, signal: externalController.signal }
        : options
      const response = await fetchWithTimeout(url, fetchOptions, timeout);
      
      // If successful or non-retryable error, return immediately
      if (response.ok || !retryableStatuses.includes(response.status)) {
        return response;
      }
      
      // If retryable error and not last attempt
      if (attempt < maxRetries) {
        const delay = getRetryDelay(attempt, retryDelay);
        if (onRetry) {
          onRetry(attempt + 1, maxRetries, delay);
        }
        await sleep(delay);
        continue;
      }
      
      return response;
    } catch (error) {
      lastError = error;
      
      // Retry on network errors and timeouts (AbortError) if not last attempt
      // Timeouts are retryable as they might be transient network issues
      if (attempt < maxRetries) {
        const delay = getRetryDelay(attempt, retryDelay);
        if (onRetry) {
          onRetry(attempt + 1, maxRetries, delay);
        }
        await sleep(delay);
        continue;
      }
      
      // Last attempt failed
      throw error;
    }
  }
  
  throw lastError;
};

/**
 * Analyze frame with improved error handling
 * @param {string} imageData - Base64 encoded image data
 * @param {Function} onRetry - Optional callback for retry events
 * @param {AbortController} externalController - Optional external AbortController for cleanup
 */
export const analyzeFrame = async (imageData, onRetry = null, externalController = null) => {
  // Validate imageData before making request
  if (!imageData || typeof imageData !== 'string') {
    throw new Error('Invalid image data: imageData is required and must be a string')
  }
  
  if (!imageData.startsWith('data:image/')) {
    throw new Error('Invalid image data: must be a valid image data URL')
  }
  
  if (imageData.length < 100) {
    throw new Error('Invalid image data: image data is too short or empty')
  }
  
  // Create internal controller if none provided
  const internalController = new AbortController()
  const controller = externalController || internalController
  
  try {
    const response = await fetchWithRetry(
      '/analyze',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
      },
      {
        maxRetries: 2, // Fewer retries for real-time analysis
        timeout: 8000, // 8 seconds timeout
        retryDelay: 500, // Faster retry for real-time
        onRetry,
        externalController: controller
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `Server error: ${response.status}`;
      
      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.error || errorMessage;
      } catch {
        errorMessage = errorText || errorMessage;
      }
      
      throw new Error(errorMessage);
    }

    const result = await response.json();
    
    // Validate response structure
    if (!result || typeof result !== 'object') {
      throw new Error('Invalid response format from server');
    }

    return result;
  } catch (error) {
    // Provide user-friendly error messages
    if (error.name === 'AbortError' || error.message.includes('timeout')) {
      throw new Error('Request timeout - the server is taking too long to respond. Please check your connection.');
    } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      throw new Error('Cannot connect to server. Make sure the backend is running on http://localhost:8000');
    } else {
      throw error;
    }
  }
};


