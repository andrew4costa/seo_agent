import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import sys
import json
import logging
import time
from typing import Dict, List, Set, Any, Tuple, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fixed_enhanced_analyzer")

class EnhancedSEOAnalyzer:
    """A fixed version of the SEO analyzer that avoids recursion issues"""
    
    def __init__(self, max_pages: int = 3, timeout: int = 10):
        self.max_pages = max_pages
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def analyze_url(self, url: str, analyze_competitors: bool = False, competitors: List[str] = None) -> Dict[str, Any]:
        """Analyze a URL and return a comprehensive SEO analysis"""
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        domain = urlparse(url).netloc
        logger.info(f"Analyzing website: {domain}")
        
        results = {
            "url": url,
            "domain": domain,
            "analyzed_pages": {},
            "technical_seo": {
                "score": 0,
                "has_ssl": url.startswith('https://'),
                "robots_txt": await self._check_robots_txt(url),
                "sitemap": await self._check_sitemap(url)
            },
            "on_page_seo": {
                "score": 0,
                "meta_tags": {},
                "content_quality": {}
            },
            "summary": {
                "pages_analyzed": 0,
                "total_word_count": 0,
                "avg_word_count": 0
            }
        }
        
        # Analyze pages
        pages_analyzed, page_data = await self._analyze_pages(url)
        results["analyzed_pages"] = page_data
        results["summary"]["pages_analyzed"] = len(page_data)
        
        if page_data:
            # Calculate averages and summaries
            total_word_count = sum(data.get("word_count", 0) for data in page_data.values())
            results["summary"]["total_word_count"] = total_word_count
            results["summary"]["avg_word_count"] = total_word_count / len(page_data) if page_data else 0
            
            # Count meta tag issues
            missing_titles = sum(1 for data in page_data.values() if not data.get("title"))
            missing_descriptions = sum(1 for data in page_data.values() if not data.get("meta_description"))
            missing_h1s = sum(1 for data in page_data.values() if not data.get("h1_tags"))
            
            results["on_page_seo"]["meta_tags"] = {
                "missing_titles": missing_titles,
                "missing_descriptions": missing_descriptions,
                "missing_h1s": missing_h1s
            }
            
            # Calculate thin content
            thin_content = sum(1 for data in page_data.values() if data.get("word_count", 0) < 300)
            results["on_page_seo"]["content_quality"] = {
                "thin_content_pages": thin_content,
                "avg_word_count": results["summary"]["avg_word_count"]
            }
            
            # Calculate scores
            technical_score = self._calculate_technical_score(results)
            on_page_score = self._calculate_on_page_score(results)
            
            results["technical_seo"]["score"] = technical_score
            results["on_page_seo"]["score"] = on_page_score
        
        # Add competitor analysis if requested
        if analyze_competitors and competitors:
            competitor_analysis = await self._analyze_competitors(url, competitors)
            results["competitor_analysis"] = competitor_analysis
        
        return results
    
    async def _check_robots_txt(self, url: str) -> Dict[str, Any]:
        """Check if robots.txt exists and its contents"""
        robots_url = urljoin(url, "/robots.txt")
        result = {"exists": False, "content": None}
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            
            async with self.session.get(robots_url, allow_redirects=True, max_redirects=5) as response:
                if response.status == 200:
                    result["exists"] = True
                    result["content"] = await response.text()
                    
                    # Check for sitemap in robots.txt
                    result["has_sitemap_directive"] = "Sitemap:" in result["content"]
        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
        
        return result
    
    async def _check_sitemap(self, url: str) -> Dict[str, Any]:
        """Check if sitemap.xml exists and its contents"""
        sitemap_url = urljoin(url, "/sitemap.xml")
        result = {"exists": False, "urls_count": 0}
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            
            async with self.session.get(sitemap_url, allow_redirects=True, max_redirects=5) as response:
                if response.status == 200:
                    result["exists"] = True
                    content = await response.text()
                    
                    # Count URLs (basic approximation)
                    url_count = content.count("<url>")
                    if url_count == 0:
                        url_count = content.count("<loc>")
                    
                    result["urls_count"] = url_count
        except Exception as e:
            logger.error(f"Error checking sitemap: {e}")
        
        return result
    
    async def _analyze_pages(self, start_url: str) -> Tuple[int, Dict[str, Dict[str, Any]]]:
        """Analyze multiple pages from a website"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        base_domain = urlparse(start_url).netloc
        to_visit = [(start_url, 0)]  # (url, depth)
        visited = set()
        analyzed_pages = {}
        
        MAX_DEPTH = 2  # Limit crawl depth to avoid recursion issues
        
        while to_visit and len(analyzed_pages) < self.max_pages:
            current_url, depth = to_visit.pop(0)
            if current_url in visited or depth > MAX_DEPTH:
                continue
            
            visited.add(current_url)
            logger.info(f"Analyzing page: {current_url} (depth: {depth})")
            
            try:
                page_data = await self._analyze_single_page(current_url)
                analyzed_pages[current_url] = page_data
                
                # Add internal links to the crawl queue if we're not at max depth
                if depth < MAX_DEPTH:
                    for link in page_data.get("internal_links", []):
                        # Skip already visited URLs and non-HTML resources
                        if link not in visited and urlparse(link).netloc == base_domain:
                            # Skip URLs with common non-HTML extensions
                            if not any(link.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.pdf']):
                                to_visit.append((link, depth + 1))
            except Exception as e:
                logger.error(f"Error analyzing {current_url}: {e}")
        
        return len(visited), analyzed_pages
    
    async def _analyze_single_page(self, url: str) -> Dict[str, Any]:
        """Analyze a single page for SEO factors"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        result = {
            "url": url,
            "load_time_seconds": 0,
            "status_code": 0,
            "title": None,
            "meta_description": None,
            "h1_tags": [],
            "h2_tags": [],
            "internal_links": [],
            "external_links": [],
            "images": [],
            "images_without_alt": 0,
            "word_count": 0,
            "has_schema_markup": False,
            "issues": []
        }
        
        try:
            start_time = time.time()
            
            # Fetch page with timeout and redirect limits
            async with self.session.get(url, allow_redirects=True, max_redirects=5, 
                                     timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                result["status_code"] = response.status
                result["load_time_seconds"] = time.time() - start_time
                
                if response.status != 200:
                    result["issues"].append({
                        "type": "http_status",
                        "severity": "high" if response.status >= 400 else "medium",
                        "description": f"HTTP status code {response.status}"
                    })
                    return result
                
                try:
                    html = await response.text()
                except UnicodeDecodeError:
                    result["issues"].append({
                        "type": "encoding_error",
                        "severity": "high",
                        "description": "Failed to decode page content (encoding issue)"
                    })
                    return result
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            result["issues"].append({
                "type": "connection_error",
                "severity": "high",
                "description": f"Error: {str(e)}"
            })
            return result
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        result["title"] = title_tag.get_text(strip=True) if title_tag else None
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        result["meta_description"] = meta_desc.get('content', '').strip() if meta_desc else None
        
        # Extract headings
        result["h1_tags"] = [h1.get_text(strip=True) for h1 in soup.find_all('h1')]
        result["h2_tags"] = [h2.get_text(strip=True) for h2 in soup.find_all('h2')]
        
        # Calculate word count
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=' ', strip=True)
        result["word_count"] = len(text.split())
        
        # Check for schema markup
        ld_scripts = soup.find_all('script', type='application/ld+json')
        schema_elements = soup.find_all(itemtype=True)
        result["has_schema_markup"] = bool(ld_scripts or schema_elements)
        
        # Extract links
        internal_links = []
        external_links = []
        base_domain = urlparse(url).netloc
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Handle relative URLs
            if not href.startswith(('http://', 'https://')):
                href = urljoin(url, href)
            
            # Check if internal or external
            link_domain = urlparse(href).netloc
            if link_domain == base_domain:
                if href not in internal_links:
                    internal_links.append(href)
            else:
                if href not in external_links:
                    external_links.append(href)
        
        result["internal_links"] = internal_links
        result["external_links"] = external_links
        
        # Extract images and check alt text
        images = []
        images_without_alt = 0
        
        for img in soup.find_all('img', src=True):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if src:
                # Handle relative URLs
                if not src.startswith(('http://', 'https://')):
                    src = urljoin(url, src)
                
                images.append({'src': src, 'alt': alt})
                
                if not alt:
                    images_without_alt += 1
        
        result["images"] = images
        result["images_without_alt"] = images_without_alt
        
        # Identify issues
        self._identify_page_issues(result)
        
        return result
    
    def _identify_page_issues(self, page_data: Dict[str, Any]) -> None:
        """Identify SEO issues on the page"""
        # Title issues
        if not page_data["title"]:
            page_data["issues"].append({
                "type": "missing_title",
                "severity": "high",
                "description": "Page is missing a title tag"
            })
        elif len(page_data["title"]) < 10:
            page_data["issues"].append({
                "type": "short_title",
                "severity": "medium",
                "description": f"Title is too short ({len(page_data['title'])} chars)"
            })
        elif len(page_data["title"]) > 60:
            page_data["issues"].append({
                "type": "long_title",
                "severity": "medium",
                "description": f"Title is too long ({len(page_data['title'])} chars)"
            })
        
        # Meta description issues
        if not page_data["meta_description"]:
            page_data["issues"].append({
                "type": "missing_meta_description",
                "severity": "medium",
                "description": "Page is missing a meta description"
            })
        
        # H1 issues
        if not page_data["h1_tags"]:
            page_data["issues"].append({
                "type": "missing_h1",
                "severity": "medium",
                "description": "Page is missing an H1 tag"
            })
        elif len(page_data["h1_tags"]) > 1:
            page_data["issues"].append({
                "type": "multiple_h1",
                "severity": "low",
                "description": f"Page has multiple H1 tags ({len(page_data['h1_tags'])})"
            })
        
        # Image alt text issues
        if page_data["images_without_alt"] > 0:
            page_data["issues"].append({
                "type": "images_missing_alt",
                "severity": "medium",
                "description": f"{page_data['images_without_alt']} images missing alt text"
            })
        
        # Content issues
        if page_data["word_count"] < 300:
            page_data["issues"].append({
                "type": "thin_content",
                "severity": "medium",
                "description": f"Page has thin content ({page_data['word_count']} words)"
            })
        
        # Page speed issues
        if page_data["load_time_seconds"] > 3.0:
            page_data["issues"].append({
                "type": "slow_page",
                "severity": "high",
                "description": f"Page load time is slow ({page_data['load_time_seconds']:.2f} seconds)"
            })
        
        # Schema issues
        if not page_data["has_schema_markup"]:
            page_data["issues"].append({
                "type": "missing_schema",
                "severity": "medium",
                "description": "Page has no schema markup (structured data)"
            })
    
    def _calculate_technical_score(self, site_data: Dict[str, Any]) -> float:
        """Calculate a score for technical SEO (0-10)"""
        score = 10.0
        
        # Deduct for various technical issues
        if not site_data["technical_seo"]["has_ssl"]:
            score -= 2.0
        
        if not site_data["technical_seo"]["robots_txt"]["exists"]:
            score -= 1.0
        
        if not site_data["technical_seo"]["sitemap"]["exists"]:
            score -= 1.0
        
        # Deduct for page issues (avg across pages)
        total_issues = sum(len(page["issues"]) for page in site_data["analyzed_pages"].values())
        avg_issues_per_page = total_issues / len(site_data["analyzed_pages"]) if site_data["analyzed_pages"] else 0
        
        # More than 5 issues per page on average is bad
        if avg_issues_per_page > 5:
            score -= min(3.0, avg_issues_per_page / 5)
        
        return max(0.0, round(score, 1))
    
    def _calculate_on_page_score(self, site_data: Dict[str, Any]) -> float:
        """Calculate a score for on-page SEO (0-10)"""
        score = 10.0
        pages_count = len(site_data["analyzed_pages"])
        
        if not pages_count:
            return 5.0  # Default
        
        # Deduct for missing titles/descriptions/H1s
        missing_titles = site_data["on_page_seo"]["meta_tags"]["missing_titles"]
        missing_descriptions = site_data["on_page_seo"]["meta_tags"]["missing_descriptions"]
        missing_h1s = site_data["on_page_seo"]["meta_tags"]["missing_h1s"]
        
        score -= (missing_titles / pages_count) * 2.0
        score -= (missing_descriptions / pages_count) * 1.5
        score -= (missing_h1s / pages_count) * 1.5
        
        # Deduct for thin content
        thin_content = site_data["on_page_seo"]["content_quality"]["thin_content_pages"]
        score -= (thin_content / pages_count) * 2.0
        
        # Deduct for missing schema markup
        pages_without_schema = sum(1 for page in site_data["analyzed_pages"].values() 
                                  if not page.get("has_schema_markup", False))
        score -= (pages_without_schema / pages_count) * 1.0
        
        return max(0.0, round(score, 1))
    
    async def _analyze_competitors(self, main_url: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitors (simplified version)"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        
        result = {
            "main_url": main_url,
            "competitors_analyzed": 0,
            "competitor_data": {}
        }
        
        # Process only up to 3 competitors
        max_competitors = min(len(competitor_urls), 3)
        
        for i in range(max_competitors):
            competitor_url = competitor_urls[i]
            try:
                # Ensure competitor URL is properly formatted
                if not competitor_url.startswith(('http://', 'https://')):
                    competitor_url = f'https://{competitor_url}'
                
                logger.info(f"Analyzing competitor: {competitor_url}")
                
                # Analyze only the homepage for each competitor
                competitor_data = await self._analyze_single_page(competitor_url)
                result["competitor_data"][competitor_url] = {
                    "url": competitor_url,
                    "load_time_seconds": competitor_data.get("load_time_seconds", 0),
                    "word_count": competitor_data.get("word_count", 0),
                    "title": competitor_data.get("title"),
                    "has_schema": competitor_data.get("has_schema_markup", False),
                    "meta_tags_present": bool(competitor_data.get("meta_description")),
                    "internal_links_count": len(competitor_data.get("internal_links", [])),
                    "external_links_count": len(competitor_data.get("external_links", []))
                }
                result["competitors_analyzed"] += 1
            except Exception as e:
                logger.error(f"Error analyzing competitor {competitor_url}: {e}")
                result["competitor_data"][competitor_url] = {"error": str(e)}
        
        return result

async def analyze_website(url: str, max_pages: int = 3, competitors: List[str] = None) -> Dict[str, Any]:
    """Analyze a website with the fixed enhanced analyzer"""
    async with EnhancedSEOAnalyzer(max_pages=max_pages) as analyzer:
        results = await analyzer.analyze_url(
            url, 
            analyze_competitors=bool(competitors),
            competitors=competitors
        )
    return results

async def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Please provide a URL to analyze")
        print("Usage: python fixed_enhanced_analyzer.py <url> [max_pages=3] [competitor1,competitor2,...]")
        return
    
    url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Check for competitor URLs
    competitors = None
    if len(sys.argv) > 3:
        competitors = sys.argv[3].split(',')
    
    # Run analysis
    start_time = time.time()
    logger.info(f"Starting analysis for {url} (max_pages={max_pages})")
    
    results = await analyze_website(url, max_pages, competitors)
    
    # Print summary results
    duration = time.time() - start_time
    logger.info(f"Analysis completed in {duration:.2f} seconds")
    
    technical_score = results["technical_seo"]["score"]
    on_page_score = results["on_page_seo"]["score"]
    pages_analyzed = results["summary"]["pages_analyzed"]
    
    print("\nSEO Analysis Results:")
    print(f"URL: {results['url']}")
    print(f"Technical SEO Score: {technical_score}/10")
    print(f"On-Page SEO Score: {on_page_score}/10")
    print(f"Pages analyzed: {pages_analyzed}")
    print(f"Average word count: {results['summary']['avg_word_count']:.0f} words")
    
    print("\nTechnical Findings:")
    print(f"- SSL/HTTPS: {'Yes' if results['technical_seo']['has_ssl'] else 'No'}")
    print(f"- Robots.txt: {'Found' if results['technical_seo']['robots_txt']['exists'] else 'Not found'}")
    print(f"- Sitemap.xml: {'Found' if results['technical_seo']['sitemap']['exists'] else 'Not found'}")
    
    print("\nOn-Page SEO Findings:")
    print(f"- Pages missing title: {results['on_page_seo']['meta_tags']['missing_titles']}")
    print(f"- Pages missing meta description: {results['on_page_seo']['meta_tags']['missing_descriptions']}")
    print(f"- Pages missing H1: {results['on_page_seo']['meta_tags']['missing_h1s']}")
    print(f"- Pages with thin content: {results['on_page_seo']['content_quality']['thin_content_pages']}")
    
    # Output competitor data if available
    if competitors and "competitor_analysis" in results:
        print("\nCompetitor Analysis:")
        for comp_url, comp_data in results["competitor_analysis"]["competitor_data"].items():
            if "error" in comp_data:
                print(f"- {comp_url}: Error - {comp_data['error']}")
                continue
            
            print(f"\nCompetitor: {comp_url}")
            print(f"- Load time: {comp_data.get('load_time_seconds', 0):.2f} seconds")
            print(f"- Word count: {comp_data.get('word_count', 0)} words")
            print(f"- Has schema markup: {comp_data.get('has_schema', False)}")
            print(f"- Internal links: {comp_data.get('internal_links_count', 0)}")
    
    # Save full results to file
    output_file = f"{urlparse(url).netloc}_fixed_enhanced.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full analysis saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())