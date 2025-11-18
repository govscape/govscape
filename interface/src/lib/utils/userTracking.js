import Cookies from 'js-cookie';
import { 
  loadGA4Script,
  setGA4Config,
  trackGA4Search,
  trackGA4PdfClick,
  trackGA4Pagination,
} from './ga4.js';

export class UserTracker {
  constructor() {
    this.consentGiven = Cookies.get('govscape_consent') === 'true';
    if (this.consentGiven) this.init();
  }
  
  async init() {
    try {
      await loadGA4Script().then(() => {
        setGA4Config();
      });
    } catch (error) {
      console.error(error);
    }
  }

  updateConsent(accepted) {
    this.consentGiven = accepted;

    Cookies.set('govscape_consent', accepted.toString(), {
      expires: 365,
      sameSite: 'Lax',
      secure: location.protocol === 'https:',
      path: '/'
    });

    if (accepted) this.init();
  }

  needsConsent() {
    return !Cookies.get('govscape_consent');
  }

  hasConsent() {
    return this.consentGiven;
  }

  logSearch(query, searchType, filters = {}) {
    try {
      trackGA4Search({
        query: query,
        searchType: searchType,
        filters: filters,
      });
    } catch (error) {
      console.error(error);
    }
  }

  logPdfClick({ id, page, subDomain, crawlUrl, crawlDate }) {
    try {
      trackGA4PdfClick({
        id,
        page,
        subDomain,
        crawlUrl,
        crawlDate,
      });
    } catch (error) {
      console.error(error);
    }
  }

  logPagination({ query, searchType, filters = {}, page }) {
    try {
      trackGA4Pagination({
        query,
        searchType,
        filters,
        page,
      });
    } catch (error) {
      console.error(error);
    }
  }
}

export const userTracker = new UserTracker();
