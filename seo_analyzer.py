import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
import json
import logging
import time
import ssl
import re
import pathlib
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seo_analyzer")

@dataclass
class PageSEOData:
    """Data structure to hold SEO information about a single page"""
    url: str
    status_code: int = 0
    load_time_seconds: float = 0.0
    title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: Optional[str] = None
    h1_tags: List[str] = field(default_factory=list)
    h2_tags: List[str] = field(default_factory=list)
    h3_tags: List[str] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)  # List of dicts with 'src' and 'alt'
    word_count: int = 0
    internal_links: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)
    has_schema_markup: bool = False
    schema_types: List[str] = field(default_factory=list)
    canonical_url: Optional[str] = None
    has_viewport_meta: bool = False
    issues: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SiteSEOData:
    """Data structure to hold SEO information about an entire site"""
    domain: str
    pages: Dict[str, PageSEOData] = field(default_factory=dict)
    robots_txt_status: Dict[str, Any] = field(default_factory=dict)
    sitemap_status: Dict[str, Any] = field(default_factory=dict)
    has_ssl: bool = False
    avg_page_load_time: float = 0.0
    avg_word_count: float = 0.0
    mobile_friendly_score: float = 0.0
    issues: List[Dict[str, Any]] = field(default_factory=list)


class SEOAnalyzer:
    """Class for analyzing websites for SEO issues and opportunities"""
    
    def __init__(self, max_pages: int = 10, timeout: int = 10):
        self.max_pages = max_pages
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an active session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self.session
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL format"""
        if not url.startswith(('http://', 'https://')):
            return f'https://{url}'
        return url
    
    async def analyze_site(self, url: str) -> SiteSEOData:
        """Main method to analyze a website"""
        start_time = time.time()
        
        normalized_url = self._normalize_url(url)
        self.logger.info(f"Starting SEO analysis for {normalized_url}")
        
        domain = urlparse(normalized_url).netloc
        site_data = SiteSEOData(domain=domain)
        
        session = await self._ensure_session()
        
        # Check SSL
        site_data.has_ssl = normalized_url.startswith('https://')
        if not site_data.has_ssl:
            site_data.issues.append({
                "type": "security",
                "severity": "high",
                "description": "Site does not use HTTPS"
            })
        
        # Check robots.txt
        robots_txt_url = urljoin(normalized_url, "/robots.txt")
        site_data.robots_txt_status = await self._check_robots_txt(robots_txt_url)
        
        # Check sitemap
        sitemap_url = urljoin(normalized_url, "/sitemap.xml")
        site_data.sitemap_status = await self._check_sitemap(sitemap_url)
        
        # Crawl site pages
        visited_urls, page_data_dict = await self._crawl_site(normalized_url)
        site_data.pages = page_data_dict
        
        # Calculate site-wide metrics
        if page_data_dict:
            site_data.avg_page_load_time = sum(page.load_time_seconds for page in page_data_dict.values()) / len(page_data_dict)
            site_data.avg_word_count = sum(page.word_count for page in page_data_dict.values()) / len(page_data_dict)
            
            # Calculate mobile-friendliness score based on viewport tags and other factors
            mobile_friendly_pages = sum(1 for page in page_data_dict.values() if page.has_viewport_meta)
            site_data.mobile_friendly_score = (mobile_friendly_pages / len(page_data_dict)) * 10 if page_data_dict else 0
        
        self.logger.info(f"Completed SEO analysis for {normalized_url} in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Analyzed {len(site_data.pages)} pages out of {len(visited_urls)} visited URLs")
        
        return site_data
    
    async def _crawl_site(self, start_url: str) -> Tuple[Set[str], Dict[str, PageSEOData]]:
        """Crawl a website and collect SEO data from pages"""
        session = await self._ensure_session()
        base_domain = urlparse(start_url).netloc
        
        to_visit = [start_url]
        visited = set()
        analyzed_pages = {}
        
        while to_visit and len(analyzed_pages) < self.max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
            
            visited.add(current_url)
            self.logger.info(f"Analyzing page: {current_url}")
            
            try:
                page_data = await self._analyze_page(current_url)
                analyzed_pages[current_url] = page_data
                
                # Add internal links to the crawl queue
                for link in page_data.internal_links:
                    if link not in visited and urlparse(link).netloc == base_domain:
                        to_visit.append(link)
            except Exception as e:
                self.logger.error(f"Error analyzing {current_url}: {e}")
        
        return visited, analyzed_pages
    
    async def _analyze_page(self, url: str) -> PageSEOData:
        """Analyze a single page for SEO factors"""
        session = await self._ensure_session()
        parsed_url = urlparse(url)
        page_data = PageSEOData(url=url)
        
        start_time = time.time()
        try:
            async with session.get(url) as response:
                page_data.status_code = response.status
                page_data.load_time_seconds = time.time() - start_time
                
                if response.status != 200:
                    page_data.issues.append({
                        "type": "http_status",
                        "severity": "high" if response.status >= 400 else "medium",
                        "description": f"HTTP status code {response.status}"
                    })
                    return page_data
                
                html = await response.text()
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            page_data.issues.append({
                "type": "connection_error",
                "severity": "high",
                "description": f"Failed to connect: {str(e)}"
            })
            return page_data
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract basic SEO elements
        page_data.title = self._extract_title(soup)
        page_data.meta_description = self._extract_meta_description(soup)
        page_data.meta_keywords = self._extract_meta_keywords(soup)
        page_data.canonical_url = self._extract_canonical_url(soup)
        
        # Check for viewport meta tag (mobile-friendliness indicator)
        page_data.has_viewport_meta = self._has_viewport_meta(soup)
        
        # Extract headings
        page_data.h1_tags = [h1.get_text(strip=True) for h1 in soup.find_all('h1')]
        page_data.h2_tags = [h2.get_text(strip=True) for h2 in soup.find_all('h2')]
        page_data.h3_tags = [h3.get_text(strip=True) for h3 in soup.find_all('h3')]
        
        # Extract links
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        internal_links, external_links = self._extract_links(soup, base_url)
        page_data.internal_links = internal_links
        page_data.external_links = external_links
        
        # Extract images and check alt tags
        page_data.images = self._extract_images(soup)
        
        # Calculate word count
        page_data.word_count = self._calculate_word_count(soup)
        
        # Check for schema markup
        page_data.has_schema_markup, page_data.schema_types = self._check_schema_markup(soup)
        
        # Identify issues
        self._identify_page_issues(page_data, soup)
        
        return page_data
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else None
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content', '').strip() if meta_desc else None
    
    def _extract_meta_keywords(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta keywords"""
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        return meta_keywords.get('content', '').strip() if meta_keywords else None
    
    def _extract_canonical_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract canonical URL"""
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        return canonical.get('href', '').strip() if canonical else None
    
    def _has_viewport_meta(self, soup: BeautifulSoup) -> bool:
        """Check if page has viewport meta tag for mobile responsiveness"""
        viewport = soup.find('meta', attrs={'name': 'viewport'})
        return bool(viewport)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Tuple[List[str], List[str]]:
        """Extract internal and external links"""
        internal_links = []
        external_links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Handle relative URLs
            if not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            # Determine if internal or external
            if urlparse(href).netloc == urlparse(base_url).netloc:
                if href not in internal_links:
                    internal_links.append(href)
            else:
                if href not in external_links:
                    external_links.append(href)
        
        return internal_links, external_links
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract images and their alt text"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src:
                images.append({'src': src, 'alt': alt})
        return images
    
    def _calculate_word_count(self, soup: BeautifulSoup) -> int:
        """Calculate the number of words in the main content"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text and count words
        text = soup.get_text(separator=' ', strip=True)
        words = text.split()
        return len(words)
    
    def _check_schema_markup(self, soup: BeautifulSoup) -> Tuple[bool, List[str]]:
        """Check for schema markup / structured data"""
        schema_types = []
        
        # Look for JSON-LD schema
        ld_scripts = soup.find_all('script', type='application/ld+json')
        if ld_scripts:
            for script in ld_scripts:
                try:
                    json_data = json.loads(script.string)
                    if '@type' in json_data:
                        schema_types.append(json_data['@type'])
                except (json.JSONDecodeError, AttributeError):
                    pass
        
        # Look for microdata schema
        elements_with_itemtype = soup.find_all(itemtype=True)
        for el in elements_with_itemtype:
            itemtype = el.get('itemtype', '')
            if itemtype:
                schema_type = itemtype.split('/')[-1]
                if schema_type:
                    schema_types.append(schema_type)
        
        return bool(schema_types), schema_types
    
    def _identify_page_issues(self, page_data: PageSEOData, soup: BeautifulSoup) -> None:
        """Identify SEO issues on the page"""
        # Title issues
        if not page_data.title:
            page_data.issues.append({
                "type": "missing_title",
                "severity": "high",
                "description": "Page is missing a title tag"
            })
        elif len(page_data.title) < 10:
            page_data.issues.append({
                "type": "short_title",
                "severity": "medium",
                "description": f"Title is too short ({len(page_data.title)} chars)"
            })
        elif len(page_data.title) > 60:
            page_data.issues.append({
                "type": "long_title",
                "severity": "medium",
                "description": f"Title is too long ({len(page_data.title)} chars)"
            })
        
        # Meta description issues
        if not page_data.meta_description:
            page_data.issues.append({
                "type": "missing_meta_description",
                "severity": "medium",
                "description": "Page is missing a meta description"
            })
        elif len(page_data.meta_description) < 50:
            page_data.issues.append({
                "type": "short_meta_description",
                "severity": "low",
                "description": f"Meta description is too short ({len(page_data.meta_description)} chars)"
            })
        elif len(page_data.meta_description) > 160:
            page_data.issues.append({
                "type": "long_meta_description",
                "severity": "low",
                "description": f"Meta description is too long ({len(page_data.meta_description)} chars)"
            })
        
        # H1 issues
        if not page_data.h1_tags:
            page_data.issues.append({
                "type": "missing_h1",
                "severity": "medium",
                "description": "Page is missing an H1 tag"
            })
        elif len(page_data.h1_tags) > 1:
            page_data.issues.append({
                "type": "multiple_h1",
                "severity": "low",
                "description": f"Page has multiple H1 tags ({len(page_data.h1_tags)})"
            })
        
        # Image alt text issues
        images_without_alt = [img for img in page_data.images if not img.get('alt')]
        if images_without_alt:
            page_data.issues.append({
                "type": "images_missing_alt",
                "severity": "medium",
                "description": f"{len(images_without_alt)} images missing alt text"
            })
        
        # Mobile-friendliness issues
        if not page_data.has_viewport_meta:
            page_data.issues.append({
                "type": "no_viewport_meta",
                "severity": "high",
                "description": "Page is missing viewport meta tag for mobile devices"
            })
        
        # Content issues
        if page_data.word_count < 300:
            page_data.issues.append({
                "type": "thin_content",
                "severity": "medium",
                "description": f"Page has thin content ({page_data.word_count} words)"
            })
        
        # Page speed issues
        if page_data.load_time_seconds > 3.0:
            page_data.issues.append({
                "type": "slow_page",
                "severity": "high",
                "description": f"Page load time is slow ({page_data.load_time_seconds:.2f} seconds)"
            })
        
        # Schema issues
        if not page_data.has_schema_markup:
            page_data.issues.append({
                "type": "missing_schema",
                "severity": "medium",
                "description": "Page has no schema markup (structured data)"
            })
        
        # Links issues
        if len(page_data.internal_links) < 3:
            page_data.issues.append({
                "type": "few_internal_links",
                "severity": "medium",
                "description": f"Page has few internal links ({len(page_data.internal_links)})"
            })
    
    async def _check_robots_txt(self, robots_txt_url: str) -> Dict[str, Any]:
        """Check robots.txt file"""
        session = await self._ensure_session()
        result = {
            "exists": False,
            "content": None,
            "issues": []
        }
        
        try:
            async with session.get(robots_txt_url) as response:
                if response.status == 200:
                    result["exists"] = True
                    result["content"] = await response.text()
                    
                    # Check for sitemap in robots.txt
                    if "Sitemap:" not in result["content"]:
                        result["issues"].append({
                            "type": "no_sitemap_in_robots",
                            "severity": "medium",
                            "description": "No Sitemap directive found in robots.txt"
                        })
                else:
                    result["issues"].append({
                        "type": "robots_not_found",
                        "severity": "medium",
                        "description": f"robots.txt not found (status code: {response.status})"
                    })
        except Exception as e:
            self.logger.error(f"Error checking robots.txt: {e}")
            result["issues"].append({
                "type": "robots_error",
                "severity": "medium",
                "description": f"Error checking robots.txt: {str(e)}"
            })
        
        return result
    
    async def _check_sitemap(self, sitemap_url: str) -> Dict[str, Any]:
        """Check sitemap.xml file"""
        session = await self._ensure_session()
        result = {
            "exists": False,
            "urls_count": 0,
            "issues": []
        }
        
        try:
            async with session.get(sitemap_url) as response:
                if response.status == 200:
                    result["exists"] = True
                    sitemap_content = await response.text()
                    
                    soup = BeautifulSoup(sitemap_content, 'xml')
                    urls = soup.find_all('url')
                    result["urls_count"] = len(urls)
                    
                    if len(urls) == 0:
                        result["issues"].append({
                            "type": "empty_sitemap",
                            "severity": "medium",
                            "description": "Sitemap exists but contains no URLs"
                        })
                else:
                    result["issues"].append({
                        "type": "sitemap_not_found",
                        "severity": "medium",
                        "description": f"sitemap.xml not found (status code: {response.status})"
                    })
        except Exception as e:
            self.logger.error(f"Error checking sitemap: {e}")
            result["issues"].append({
                "type": "sitemap_error",
                "severity": "medium",
                "description": f"Error checking sitemap: {str(e)}"
            })
        
        return result

# Helper function to convert analysis data to format expected by the LLM processor
def convert_analysis_to_llm_format(site_data: SiteSEOData) -> Dict[str, Any]:
    """Convert SiteSEOData to a format expected by the LLM processor"""
    return {
        "technical_seo": {
            "score": _calculate_technical_score(site_data),
            "site_speed": {
                "avg_load_time": site_data.avg_page_load_time,
                "issues": [issue for issue in site_data.issues if issue.get("type", "").startswith("slow_")]
            },
            "mobile_optimisation": {
                "score": site_data.mobile_friendly_score,
                "has_viewport_meta": all(page.has_viewport_meta for page in site_data.pages.values())
            },
            "indexation": {
                "robots_txt": site_data.robots_txt_status,
                "sitemap": site_data.sitemap_status
            },
            "security": {
                "has_ssl": site_data.has_ssl
            },
            "structured_data": {
                "pages_with_schema": sum(1 for page in site_data.pages.values() if page.has_schema_markup),
                "schema_types": _get_unique_schema_types(site_data)
            }
        },
        "on_page_seo": {
            "score": _calculate_onpage_score(site_data),
            "content_quality": {
                "avg_word_count": site_data.avg_word_count,
                "thin_content_pages": sum(1 for page in site_data.pages.values() if any(
                    issue["type"] == "thin_content" for issue in page.issues
                ))
            },
            "meta_tags": {
                "pages_without_title": sum(1 for page in site_data.pages.values() if not page.title),
                "pages_without_description": sum(1 for page in site_data.pages.values() if not page.meta_description)
            },
            "heading_structure": {
                "pages_without_h1": sum(1 for page in site_data.pages.values() if not page.h1_tags)
            },
            "image_optimization": {
                "images_without_alt": sum(
                    len([img for img in page.images if not img.get('alt')]) 
                    for page in site_data.pages.values()
                )
            }
        },
        "off_page_seo": {
            "score": 5.0,  # Placeholder as we don't collect this data directly
            "backlinks": {
                "note": "Backlink data requires third-party API integration"
            },
            "social_signals": {
                "note": "Social signal data requires third-party API integration"
            }
        },
        "analyzed_pages": {
            url: {
                "title": page.title,
                "description": page.meta_description,
                "word_count": page.word_count,
                "load_time": page.load_time_seconds,
                "issues": page.issues
            } for url, page in site_data.pages.items()
        }
    }

def _calculate_technical_score(site_data: SiteSEOData) -> float:
    """
    Calculate a score for technical SEO (0-10) using a dynamic weighting system
    rather than predefined deductions.
    """
    # Define factor weights dynamically based on their relative importance
    factors = {
        "ssl": {
            "weight": 0.15,
            "score": 1.0 if site_data.has_ssl else 0.0
        },
        "robots_txt": {
            "weight": 0.10,
            "score": 1.0 if site_data.robots_txt_status.get("exists", False) else 0.0
        },
        "sitemap": {
            "weight": 0.10,
            "score": 1.0 if site_data.sitemap_status.get("exists", False) else 0.0
        },
        "page_load": {
            "weight": 0.20,
            "score": max(0.0, min(1.0, 1.0 - (site_data.avg_page_load_time - 1.0) / 4.0))
            # 1 second or less: 1.0, 5+ seconds: 0.0, linear scale between
        },
        "mobile_friendly": {
            "weight": 0.25,
            "score": site_data.mobile_friendly_score / 10.0
        },
        "high_severity_issues": {
            "weight": 0.20,
            "score": 0.0  # Will be calculated below
        }
    }
    
    # Calculate high severity issues score
    total_pages = len(site_data.pages)
    if total_pages > 0:
        # Count high severity technical issues
        high_severity_issues = sum(
            1 for page in site_data.pages.values() 
            for issue in page.issues 
            if issue.get("severity") == "high" and issue.get("type") in [
                "slow_page", "no_viewport_meta", "http_status", "connection_error"
            ]
        )
        # Calculate a score based on the ratio of pages with high severity issues
        issues_ratio = high_severity_issues / total_pages
        factors["high_severity_issues"]["score"] = max(0.0, 1.0 - min(1.0, issues_ratio * 2))
    
    # Calculate weighted score
    weighted_score = 0.0
    for factor_name, factor_data in factors.items():
        weighted_score += factor_data["weight"] * factor_data["score"]
    
    # Convert to 0-10 scale
    final_score = weighted_score * 10.0
    
    return round(max(0.0, min(10.0, final_score)), 1)

def _calculate_onpage_score(site_data: SiteSEOData) -> float:
    """
    Calculate a score for on-page SEO (0-10) using a dynamic weighting system
    rather than predefined deductions.
    """
    total_pages = len(site_data.pages)
    
    if total_pages == 0:
        return 5.0  # Default score if no pages analyzed
    
    # Define factor weights dynamically based on their relative importance
    factors = {
        "titles": {
            "weight": 0.20,
            "score": 0.0  # Will be calculated below
        },
        "meta_descriptions": {
            "weight": 0.15,
            "score": 0.0  # Will be calculated below
        },
        "h1_tags": {
            "weight": 0.15,
            "score": 0.0  # Will be calculated below
        },
        "content_length": {
            "weight": 0.20,
            "score": 0.0  # Will be calculated below
        },
        "schema_markup": {
            "weight": 0.10,
            "score": 0.0  # Will be calculated below
        },
        "image_alt_text": {
            "weight": 0.10,
            "score": 0.0  # Will be calculated below
        },
        "content_structure": {
            "weight": 0.10,
            "score": 0.0  # Will be calculated below
        }
    }
    
    # Calculate scores for each factor
    # Title tags
    pages_with_title = sum(1 for page in site_data.pages.values() if page.title)
    factors["titles"]["score"] = pages_with_title / total_pages
    
    # Meta descriptions
    pages_with_desc = sum(1 for page in site_data.pages.values() if page.meta_description)
    factors["meta_descriptions"]["score"] = pages_with_desc / total_pages
    
    # H1 tags
    pages_with_h1 = sum(1 for page in site_data.pages.values() if page.h1_tags)
    factors["h1_tags"]["score"] = pages_with_h1 / total_pages
    
    # Content length
    # Score based on average word count - 300 words as minimum, 1500+ as ideal
    avg_word_count = site_data.avg_word_count
    if avg_word_count >= 1500:
        content_score = 1.0
    elif avg_word_count < 300:
        content_score = 0.3  # minimum quality score
    else:
        content_score = 0.3 + (avg_word_count - 300) / (1500 - 300) * 0.7
    
    factors["content_length"]["score"] = content_score
    
    # Schema markup
    pages_with_schema = sum(1 for page in site_data.pages.values() if page.has_schema_markup)
    factors["schema_markup"]["score"] = pages_with_schema / total_pages
    
    # Image alt text
    total_images = sum(len(page.images) for page in site_data.pages.values())
    if total_images > 0:
        images_with_alt = sum(
            len([img for img in page.images if img.get('alt')]) 
            for page in site_data.pages.values()
        )
        factors["image_alt_text"]["score"] = images_with_alt / total_images
    else:
        # No images on the site, so this factor isn't applicable
        # Redistribute weight to other factors
        extra_weight = factors["image_alt_text"]["weight"]
        factors["image_alt_text"]["weight"] = 0
        
        for factor in ["titles", "meta_descriptions", "h1_tags", "content_length"]:
            factors[factor]["weight"] += extra_weight / 4
    
    # Content structure (headings, paragraphs, etc.)
    # Calculate based on presence of structured content elements like h2, h3, etc.
    structure_elements = 0
    for page in site_data.pages.values():
        if page.h2_tags:
            structure_elements += 1
        if page.h3_tags:
            structure_elements += 1
    
    factors["content_structure"]["score"] = min(1.0, structure_elements / (total_pages * 2))
    
    # Calculate weighted score
    weighted_score = 0.0
    for factor_name, factor_data in factors.items():
        weighted_score += factor_data["weight"] * factor_data["score"]
    
    # Convert to 0-10 scale
    final_score = weighted_score * 10.0
    
    return round(max(0.0, min(10.0, final_score)), 1)

def _get_unique_schema_types(site_data: SiteSEOData) -> List[str]:
    """Get all unique schema types used across the site"""
    unique_types = set()
    for page in site_data.pages.values():
        for schema_type in page.schema_types:
            unique_types.add(schema_type)
    return list(unique_types)

async def analyze_url(url: str, max_pages: int = 5) -> Dict[str, Any]:
    """Analyze a URL and return structured data for SEO analysis"""
    try:
        async with SEOAnalyzer(max_pages=max_pages) as analyzer:
            site_data = await analyzer.analyze_site(url)
            return convert_analysis_to_llm_format(site_data)
    except Exception as e:
        logger.error(f"Error analyzing {url}: {e}")
        # Return minimal structure to prevent errors
        return {
            "technical_seo": {"score": 0},
            "on_page_seo": {"score": 0},
            "off_page_seo": {"score": 0},
            "analyzed_pages": {},
            "error": str(e)
        }