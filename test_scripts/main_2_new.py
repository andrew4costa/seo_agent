import asyncio
import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass
import re
import ssl
from collections import Counter
import argparse
import json
from datetime import datetime
from textblob import TextBlob
from collections import Counter
from dotenv import load_dotenv
import os
from functools import lru_cache
import time
from typing import Coroutine
import sys
import traceback

load_dotenv()  # Loads environment variables from a .env file

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API Key is not set in the environment variable.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Retry logic with exponential backoff
async def retry_on_rate_limit(func, *args, **kwargs):
    retries = 5
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except RateLimitError as e:
            # Wait before retrying with exponential backoff strategy
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            raise e  # Propagate non-rate-limit exceptions

async def safe_task(coro: Coroutine, timeout: int = 10):
    """Wraps a coroutine to add timeout and error handling."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Task timed out after {timeout} seconds.")
    except Exception as e:
        logger.exception(f"Task failed with error: {e}")
    return None

def log_execution_time(func):
    """Decorator to log execution time of a coroutine."""
    async def wrapper(*args, **kwargs):
        start_time = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        end_time = asyncio.get_event_loop().time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

# Data Classes (same as in the LLM-integrated script)
@dataclass
class TechnicalSEOMetrics:
    page_speed: Dict
    mobile_friendly: bool
    ssl_features: bool
    robots_txt: Dict
    sitemap_status: Dict
    crawlability: Dict
    http_status: int
    schema_markup: Dict

@dataclass
class OnPageSEOMetrics:
    title_tag: str
    meta_description: str
    heading_structure: Dict
    content_quality: Dict
    internal_links: List[str]
    image_optimisation: Dict
    content_structure: Dict

@dataclass
class OffPageSEOMetrics:
    backlinks: List[Dict]
    domain_superiority: float
    social_signals: Dict
    brand_mentions: List[Dict]
    local_citations: List[Dict]

@dataclass
class LLMResponse:
    content: str
    timestamp: datetime
    tokens_used: int
    model: str

class LLMProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key,
        self.api_url = "https://api.openai.com/v1/chat/completions",
        self.logger = logging.getLogger(__name__)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def analyse_content(self, prompt: str) -> Dict:
        """
        Analyses content using OpenAI's APi with structured output.

        Args: prompt(str)

        Returns: Dict
        """
        try:
            messages = [
                {"role": "system", "content": "You are an SEO expert analyzing content quality. Provide analysis in JSON format."},
                {"role": "user", "content": prompt}
            ]

            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    self.api_url,
                    headers= self.headers,
                    json={
                        "model": "gpt-4",
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 1000
                    }
                ) as response:
                    if response.status != 200:
                        error_content = await response.text()
                        self.logger.error(f"API Error: {error_content}")
                        return {"error": f"API returned status {response.status}"}
                    
                    result = await response.json()

                    try:
                        analysis = json.loads(result['choices'][0]['message']['content'])
                        return analysis
                    except json.JSONDecodeError:
                        return {
                            "analysis": result['choices'][0]['message']['content'],
                            "format_error": "Response was not in JSON format" 
                        }
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during content analysis: {str(e)}")
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Unexpected error during content analysis: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    async def generate_recommendations(self, prompt: str) -> List[Dict]:
        """
        Generates SEO recommendations using OpenAI's API.
        
        Args:
            prompt (str): The recommendation prompt
            
        Returns:
            List[Dict]: List of recommendations with priority scores and impact
        """
        try:
            messages = [
                {"role": "system", "content": """You are an SEO expert generating recommendations. 
                 Format each recommendation as a JSON object with fields:
                 - recommendation (str)
                 - priority_score (int 1-10)
                 - impact (str: 'high', 'medium', 'low')
                 - implementation_difficulty (str: 'easy', 'medium', 'hard')
                 - estimated_time (str)"""},
                {"role": "user", "content": prompt}
            ]
            
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "model": "gpt-4",  # or another appropriate model
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 1500
                    }
                ) as response:
                    if response.status != 200:
                        error_content = await response.text()
                        self.logger.error(f"API Error: {error_content}")
                        return [{"error": f"API returned status {response.status}"}]
                    
                    result = await response.json()
                    
                    try:
                        # Parse the response content as JSON
                        recommendations = json.loads(result['choices'][0]['message']['content'])
                        if isinstance(recommendations, list):
                            return recommendations
                        else:
                            return [recommendations]  # Wrap single recommendation in list
                    except json.JSONDecodeError:
                        # If the response isn't valid JSON, return error in list format
                        return [{
                            "error": "Response parsing error",
                            "raw_response": result['choices'][0]['message']['content']
                        }]

        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during recommendation generation: {str(e)}")
            return [{"error": f"Network error: {str(e)}"}]
        except Exception as e:
            self.logger.error(f"Unexpected error during recommendation generation: {str(e)}")
            return [{"error": f"Unexpected error: {str(e)}"}]

@dataclass
class PageData:
    url: str
    title: str = None
    description: str = None
    keywords: List[str] = None
    status_code: int = None
    content_type: str = None
    links: List[str] = None
    internal_links: List[str] = None
    external_links: List[str] = None
    word_count: int = 0
    headings: Dict = None
    meta_robots: str = None
    canonical: str = None
    h1: str = None
    h2: str = None

class ComprehensiveSEOAgent:
    def __init__(self, api_key: str, user_agent: str = "ComprehensiveSEOAgent/1.0", concurrency: int = 10):
        self.api_key = api_key
        self.llm = LLMProcessor(api_key)
        self.logger = logger
        self.visited_urls = set()
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.session = None

    async def __aenter__(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), headers={'User-Agent': self.user_agent}, trust_env=True)
        return self

    async def __aexit__(self, *err):
        if self.session:
            await self.session.close()

    async def fetch_page(self, url: str) -> PageData:
        """Fetches a single page and extracts basic information."""
        try:
            if url in self.visited_urls:
                #self.logger.info(f"Already visited: {url}")
                return None
            self.visited_urls.add(url)
            async with self.session.get(url, timeout=30) as response:
                page_data = PageData(url=url)
                page_data.status_code = response.status
                page_data.content_type = response.content_type

                if response.status == 200 and 'text/html' in response.content_type:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    page_data.title = soup.title.string if soup.title else None
                    meta_description = soup.find("meta", attrs={"name": "description"})
                    page_data.description = meta_description["content"] if meta_description else None
                    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
                    page_data.keywords = meta_keywords["content"].split(",") if meta_keywords else None
                    page_data.meta_robots = soup.find("meta", attrs={"name": "robots"})["content"] if soup.find("meta", attrs={"name": "robots"}) else None
                    page_data.canonical = soup.find("link", rel="canonical")["href"] if soup.find("link", rel="canonical") else None
                    page_data.h1 = soup.find("h1").text.strip() if soup.find("h1") else None
                    page_data.h2 = soup.find("h2").text.strip() if soup.find("h2") else None

                    links = [link.get('href') for link in soup.find_all('a', href=True)]
                    page_data.links = links
                    page_data.internal_links = [urljoin(url, link) for link in links if not urlparse(link).netloc or urlparse(link).netloc == urlparse(url).netloc]
                    page_data.external_links = [link for link in links if urlparse(link).netloc and urlparse(link).netloc != urlparse(url).netloc]
                    text_content = soup.get_text()
                    page_data.word_count = len(text_content.split())
                    page_data.headings = self._extract_headings(soup)
                return page_data

        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching {url}")
            return None
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred: {e}")
            return None

    def _extract_headings(self, soup):
        headings = {}
        for i in range(1, 7):
            headings[f'h{i}'] = [h.text.strip() for h in soup.find_all(f'h{i}')]
        return headings

    async def analyse_website(self, seed_url: str, limit: int = 100):
        """Crawls and analyzes a website."""
        print(f"[SEO AGENT] Starting crawl at: {seed_url}")
        queue = asyncio.Queue()
        await queue.put(seed_url)
        processed_count = 0
        
        # Initialize overall analysis structure
        analysis = {
            "technical_seo": {},
            "on_page_seo": {},
            "off_page_seo": {}
        }
        
        # Analyze the main seed URL for more comprehensive data
        if processed_count < limit:
            print(f"[SEO AGENT] Analyzing seed URL thoroughly: {seed_url}")
            try:
                # Collect technical SEO metrics
                technical_metrics = await self.analyse_technical_seo(seed_url)
                analysis["technical_seo"] = technical_metrics
                
                # Collect on-page SEO metrics
                on_page_metrics = await self.analyse_on_page_seo(seed_url)
                analysis["on_page_seo"] = on_page_metrics
                
                # Collect off-page SEO metrics
                off_page_metrics = await self.analyse_off_page_seo(seed_url)
                analysis["off_page_seo"] = off_page_metrics
                
                print(f"[SEO AGENT] Comprehensive analysis of seed URL complete.")
            except Exception as e:
                print(f"[SEO AGENT] Error analyzing seed URL: {str(e)}")
        
        while not queue.empty() and processed_count < limit:
            url = await queue.get()
            print(f"[SEO AGENT] Fetching page: {url}")
            page_data = await self.fetch_page(url)

            if page_data:
                processed_count += 1
                print(f"[SEO AGENT] Analyzing: {url} ({processed_count}/{limit})")
                # Simplified analysis for subsequent pages
                if url != seed_url:  # Skip if it's the seed URL we already analyzed
                    try:
                        # Just collect some basic data
                        internal_linking_data = await self._analyse_internal_linking(url)
                        content_structure_data = await self._analyse_content_structure(url)
                    except Exception as e:
                        print(f"[SEO AGENT] Error analyzing page {url}: {str(e)}")
                
                # Add to queue for crawling
                if page_data.internal_links:
                    for link in page_data.internal_links:
                        if link not in self.visited_urls: #prevent adding already visited url to the queue
                            await queue.put(link)
            else:
                print(f"[SEO AGENT] Skipped or failed to fetch: {url}")
            queue.task_done()
        await queue.join()
        print("[SEO AGENT] Website crawl finished.")
        
        return analysis

    async def analyse_technical_seo(self, url: str) -> Dict:
        technical_metrics = {
            "site_speed": await self._analyse_site_speed(url),
            "mobile_optimisation": await self._check_mobile_optimisation(url),
            "security": await self._analyse_security(url),
            "indexation": await self.analyse_indexation(url),
            "site_architecture": await self._analyse_site_architecture(url),
            "crawl_efficiency": await self._analyse_crawl_efficiency(url),
            "international_targeting": await self._check_international_targeting(url),
            "structured_data": await self._analyse_structured_data(url)
        }

        return {
            "metrics": technical_metrics,
            "issues": await self._identify_technical_issues(technical_metrics),
            "score": await self._calculate_techincal_score(technical_metrics)
        }
    
    async def analyse_on_page_seo(self, url: str) -> Dict:
        on_page_metrics = {
            "content_quality": await self._analyse_content_quality(url),
            "keyword_optimisation": await self._analyse_keyword_usage(url),
            "meta_tags": await self._analyse_meta_tags(url),
            "heading_structure": await self._analyse_heading_structure(url),
            "internal_linking": await self._analyse_internal_linking(url),
            "content_structure": await self._analyse_content_structure(url),
            "user_experience": await self._analyse_user_experience(url),
            "multimedia_optimisation": await self._analyse_multimedia(url)
        }

        return {
            "metrics": on_page_metrics,
            "issues": await self._identify_on_page_issues(on_page_metrics),
            "score": await self._calculate_on_page_score(on_page_metrics)
        }
    
    async def analyse_off_page_seo(self, url: str) -> Dict:
        off_page_metrics = {
            "backlink_profile": await self._analyse_backlinks(url),
            "brand_signals": await self._analyse_brand_signals(url),
            "social_presence": await self._analyse_social_presence(url),
            "local_seo": await self._analyse_local_seo(url),
            "competitor_comparison": await self._analyse_competitors(url),
            "authority_metrics": await self._analyse_authority(url),
            "mentions": await self._analyse_brand_mentions(url),
            "trust_signals": await self._analyse_trust_signals(url)
        }

        return {
            "metrics": off_page_metrics,
            "issues": await self._identify_off_page_issues(off_page_metrics),
            "score": await self._calculate_off_page_score(off_page_metrics)
        }
    
    async def _analyse_site_speed(self, url: str) -> Dict:
        """
        Approximates page load time. Returns a dictionary with metrics or error information.
        """
        metrics = {  # Initialize metrics dictionary outside the try block
            "page_load_time": 0,
            "approximate": True,
            "message": "",  # Initialize message
            "error": None  # Initialize error to None
        }
        try:
            start_time = datetime.now()
            async with self.session.get(url, timeout=30) as response:
                await response.read()
            end_time = datetime.now()
            load_time = (end_time - start_time).total_seconds()

            metrics["page_load_time"] = load_time
            metrics["message"] = "This is an approximate page load time measured server-side. For accurate performance metrics (LCP, FID, CLS), use browser-based tools."

        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching {url} for speed analysis: {e}")
            metrics["error"] = str(e)
            metrics["message"] = "Error fetching the page for speed analysis."

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching {url} for speed analysis.")
            metrics["error"] = "Timeout"
            metrics["message"] = "Timeout fetching the page for speed analysis."

        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during speed analysis: {e}")
            metrics["error"] = str(e)
            metrics["message"] = "An unexpected error occurred during speed analysis."

        return metrics
                

    async def _analyse_internal_linking(self, url: str) -> Dict:
        """Analyze internal linking structure and patterns"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                base_domain = urlparse(url).netloc
                internal_links = []
                external_links = []

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if not href.startswith(('http', 'https', '//')):
                        href = urljoin(url, href)

                    parsed_href = urlparse(href)
                    if parsed_href.netloc == base_domain:
                        internal_links.append({
                            'url': href,
                            'text': link.get_text(strip=True),
                            'location': self._determine_link_location(link),
                            'followed': 'nofollow' not in link.get('rel', [])
                        })
                    else:
                        external_links.append({
                            'url': href,
                            'text': link.get_text(strip=True),
                            'followed': 'nofollow' not in link.get('rel', [])
                        })

                return {
                    'internal_links_count': len(internal_links),
                    'external_links_count': len(external_links),
                    'internal_links': internal_links,
                    'internal_link_distribution': self._analyze_link_distribution(internal_links),
                    'anchor_text_analysis': self._analyze_anchor_texts(internal_links),
                    'link_depth': self._calculate_link_depth(internal_links)
                }

        except Exception as e:
            self.logger.error(f"Error analyzing internal linking: {str(e)}")
            return {"error": str(e)}
        
    def _determine_link_location(self, link) -> str:
        """Determine the location of a link in the page structure"""
        parents = link.find_parents()
        for parent in parents:
            if parent.name == 'nav':
                return 'navigation'
            elif parent.name in ['header', 'head']:
                return 'header'
            elif parent.name == 'footer':
                return 'footer'
            elif parent.name == 'main':
                return 'main_content'
            elif parent.name == 'aside':
                return 'sidebar'
        return 'other'

    def _analyze_link_distribution(self, internal_links: List[Dict]) -> Dict:
        locations = Counter(link['location'] for link in internal_links)
        return dict(locations)

    def _analyze_anchor_texts(self, internal_links: List[Dict]) -> Dict:
        anchor_texts = [link['text'] for link in internal_links]
        return {
            'length_distribution': {
                'short': len([t for t in anchor_texts if len(t) < 10]),
                'medium': len([t for t in anchor_texts if 10 <= len(t) <= 50]),
                'long': len([t for t in anchor_texts if len(t) > 50])
            },
            'most_common': Counter(anchor_texts).most_common(5),
            'empty_anchors': len([t for t in anchor_texts if not t.strip()])
        }

    def _calculate_link_depth(self, internal_links: List[Dict]) -> Dict:
        depths = [len(urlparse(link['url']).path.split('/')) - 1 for link in internal_links]
        return {
            'average_depth': sum(depths) / len(depths) if depths else 0,
            'max_depth': max(depths) if depths else 0,
            'depth_distribution': Counter(depths)
        }
    
    async def _analyse_content_structure(self, url: str) -> Dict:
        """Analyze the structure and organization of content"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Analyze heading structure
                headings = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []}
                for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    headings[tag.name].append({
                        'text': tag.get_text(strip=True),
                        'length': len(tag.get_text(strip=True))
                    })

                # Analyze paragraphs
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
                
                # Analyze content blocks
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                content_blocks = []
                if main_content:
                    for block in main_content.find_all(['div', 'section', 'article']):
                        if len(block.get_text(strip=True)) > 100:  # Minimum content length
                            content_blocks.append({
                                'type': block.name,
                                'length': len(block.get_text(strip=True)),
                                'has_heading': bool(block.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
                            })

                return {
                    'heading_structure': {
                        'counts': {level: len(heads) for level, heads in headings.items()},
                        'hierarchy_issues': self._check_heading_hierarchy(headings),
                        'headings': headings
                    },
                    'paragraph_analysis': {
                        'count': len(paragraphs),
                        'average_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
                        'length_distribution': self._analyze_paragraph_lengths(paragraphs)
                    },
                    'content_blocks': {
                        'count': len(content_blocks),
                        'average_length': sum(block['length'] for block in content_blocks) / len(content_blocks) if content_blocks else 0,
                        'blocks_with_headings': sum(1 for block in content_blocks if block['has_heading'])
                    },
                    'readability_metrics': self._calculate_readability_metrics(paragraphs)
                }

        except Exception as e:
            self.logger.error(f"Error analyzing content structure: {str(e)}")
            return {"error": str(e)}

    def _check_heading_hierarchy(self, headings: Dict) -> List[str]:
        """Check for issues in heading hierarchy"""
        issues = []
        if len(headings['h1']) == 0:
            issues.append("Missing H1 heading")
        elif len(headings['h1']) > 1:
            issues.append("Multiple H1 headings found")
        
        # Check for skipped levels
        previous_level = 1
        for level in range(2, 7):
            if headings[f'h{level}'] and not headings[f'h{previous_level}']:
                issues.append(f"Skipped heading level {previous_level}")
            previous_level = level

        return issues

    def _analyze_paragraph_lengths(self, paragraphs: List[str]) -> Dict:
        """Analyze the distribution of paragraph lengths"""
        lengths = [len(p) for p in paragraphs]
        return {
            'short': len([l for l in lengths if l < 50]),
            'medium': len([l for l in lengths if 50 <= l <= 200]),
            'long': len([l for l in lengths if l > 200])
        }

    def _calculate_readability_metrics(self, paragraphs: List[str]) -> Dict:
        """Calculate various readability metrics for the content"""
        combined_text = ' '.join(paragraphs)
        words = combined_text.split()
        sentences = combined_text.split('.')
        
        # Basic metrics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        
        return {
            'average_word_length': avg_word_length,
            'average_sentence_length': avg_sentence_length,
            'estimated_reading_time': len(words) / 200  # Assuming 200 words per minute
        }

    async def _analyse_social_signals(self, url: str) -> Dict:
        """Analyze social media presence and engagement"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

            # Analyze social meta tags
            og_tags = {
            tag.get('property'): tag.get('content')
            for tag in soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
            }

            twitter_tags = {
                tag.get('name'): tag.get('content')
                for tag in soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
            }

            # Find social media links
            social_links = []
            social_platforms = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']
            for link in soup.find_all('a', href=True):
                if any(platform in link['href'].lower() for platform in social_platforms):
                    social_links.append({
                        'platform': next(p for p in social_platforms if p in link['href'].lower()),
                        'url': link['href']
                    })

            return {
                'social_meta_tags': {
                    'open_graph': og_tags,
                    'twitter_cards': twitter_tags,
                    'implementation_score': self._calculate_social_meta_score(og_tags, twitter_tags)
                },
                'social_presence': {
                    'platforms': [link['platform'] for link in social_links],
                    'total_profiles': len(social_links)
                },
                'sharing_functionality': self._analyze_sharing_functionality(soup)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing social signals: {str(e)}")
            return {"error": str(e)}

    def _calculate_social_meta_score(self, og_tags: Dict, twitter_tags: Dict) -> float:
        """Calculate a score for social meta tag implementation"""
        score = 0
        max_score = 10

        # Check essential Open Graph tags
        essential_og = ['og:title', 'og:description', 'og:image', 'og:url']
        score += sum(2 for tag in essential_og if tag in og_tags)

        # Check essential Twitter Card tags
        essential_twitter = ['twitter:card', 'twitter:title', 'twitter:description', 'twitter:image']
        score += sum(2 for tag in essential_twitter if tag in twitter_tags)

        return min(score, max_score)

    def _analyze_sharing_functionality(self, soup: BeautifulSoup) -> Dict:
        """Analyze social sharing functionality on the page"""
        share_buttons = []
        
        # Common share button classes and text
        share_indicators = [
            'share', 'social', 'facebook', 'twitter', 'linkedin', 'pinterest',
            'whatsapp', 'telegram', 'email'
        ]

        for element in soup.find_all(['a', 'button', 'div']):
            classes = ' '.join(element.get('class', [])).lower()
            text = element.get_text(strip=True).lower()
            
            if any(indicator in classes or indicator in text for indicator in share_indicators):
                share_buttons.append({
                    'type': element.name,
                    'platform': next((p for p in share_indicators if p in classes or p in text), 'unknown'),
                    'visible': self._is_element_visible(element)
                })

        return {
            'share_buttons_present': len(share_buttons) > 0,
            'platforms_available': list(set(btn['platform'] for btn in share_buttons)),
            'implementation_quality': self._assess_sharing_implementation(share_buttons)
        }

    def _is_element_visible(self, element) -> bool:
        """Check if an element is likely to be visible"""
        style = element.get('style', '').lower()
        classes = ' '.join(element.get('class', [])).lower()
        
        hidden_indicators = ['display: none', 'visibility: hidden', 'hidden', 'invisible']
        return not any(indicator in style or indicator in classes for indicator in hidden_indicators)

    def _assess_sharing_implementation(self, share_buttons: List[Dict]) -> str:
        """Assess the quality of social sharing implementation"""
        if not share_buttons:
            return "none"
        elif len(share_buttons) >= 3 and all(btn['visible'] for btn in share_buttons):
            return "good"
        elif len(share_buttons) >= 1:
            return "basic"
        else:
            return "poor"

    async def _analyse_brand_mentions(self, url: str) -> Dict:
        """Analyze brand mentions and visibility"""
        domain = urlparse(url).netloc
        brand_name = domain.split('.')[0]  # Simple assumption for brand name
        
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                text_content = soup.get_text(strip=True)
                
                # Find brand mentions in content
                brand_mentions = self._find_brand_mentions(text_content, brand_name)
                
                # Analyze logo presence
                logo_analysis = self._analyze_logo_presence(soup, brand_name)
                
                # Analyze brand positioning
                positioning = self._analyze_brand_positioning(soup, brand_name)

                return {
                    'brand_mentions': {
                        'total_mentions': len(brand_mentions),
                        'mentions': brand_mentions
                    },
                    'logo_presence': logo_analysis,
                    'brand_positioning': positioning
                }

        except Exception as e:
            self.logger.error(f"Error analyzing brand mentions: {str(e)}")
            return {"error": str(e)}

    def _find_brand_mentions(self, text: str, brand_name: str) -> List[Dict]:
        mentions = []
        brand_variations = [
            brand_name,
            brand_name.lower(),
            brand_name.upper(),
            brand_name.title()
        ]
        for variation in brand_variations:
            for match in re.finditer(rf"\b{variation}\b", text, re.IGNORECASE):
                mentions.append({
                    'text': match.group(),
                    'location': match.start()
                })
        return mentions
    
    def _analyze_logo_presence(self, soup: BeautifulSoup, brand_name: str) -> Dict:
        """Analyze logo presence on the page."""
        logo_tags = []
        for tag in soup.find_all('img'):
            alt_text = tag.get('alt', '')
            src = tag.get('src', '')
            if re.search(brand_name, alt_text, re.IGNORECASE) or re.search(brand_name, src, re.IGNORECASE):
                logo_tags.append(tag)
        return {
            'logo_count': len(logo_tags),
            'logo_present': bool(logo_tags)
        }
    
    def _analyze_brand_positioning(self, soup: BeautifulSoup, brand_name: str) -> Dict:
        """Analyze brand positioning based on surrounding text."""
        try:
            raw_text = soup.get_text(separator=' ', strip=True)

            brand_mentions = [m.start() for m in re.findtier(re.escape(brand_name), raw_text, re.IGNORECASE)]

            if not brand_mentions:
                return {
                    "brand_mentions": 0,
                    "sentiment": None,
                    "topics": None,
                    "summary": "Brand not mentioned in the content"
                }
            
            sentences = raw_text.split('.')
            brand_contexts = [s for s in sentences if re.search(re.escape(brand_name), s, re.IGNORECASE)]

            if not brand_contexts:
                return {
                    "brand_mentions": 0,
                    "sentiment": None,
                    "topics": None,
                    "summary": "Brand name appears without enough surrounding context."
                }
            
            sentiments = [TextBlob(context).semtiment.polarity for context in brand_contexts]
            avg_sentiment = sum(sentiments) / len(sentiments)

            word_context = ' '.join(brand_contexts)
            words = [word.lower() for word in re.findall(r'\b\w+\b', word_context)]
            common_topics = [word for word, _ in Counter(words).most_common(5)]

            return {
                "brand_mentions": len(brand_mentions),
                "sentiment": avg_sentiment,
                "topics": common_topics,
                "summary": f"The brand {brand_name} appears {len(brand_mentions)} times. Average sentiment is {'positive' if avg_sentiment > 0 else 'negative' if avg_sentiment < 0 else 'neutral'}. Common topics around the brand: {', '.join(common_topics) if common_topics else 'None'}"
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "summary": "An error occured during the brand position analysis."
            }
    
    async def _analyse_content_quality(self, url: str) -> Dict:
        prompt = f"""
        Analyse the content quality for {url} considering:
        1. Comprehensiveness
        2. Expertise demonstration
        3. Originality
        4. User Value
        5. Content freshness

        Format response as JSON with scores and recommendations for each point.
        """
        try:
            return await self.llm.analyse_content(prompt)
        except Exception as e:
            self.logger.error(f"Error during content quality analysis: {e}")
            return {"error": str(e)}
    
    async def _analyse_backlinks(self, url: str) -> Dict:
        # implement backlink analysis logic here
        # integrate with tools like ahrefs, moz or majestic
        return {
            "total_backlinks": 0,
            "referring_domains": 0,
            "domain_authority_distribution": {},
            "anchor_text_analysis": {},
            "link_quality_metrics": {}
        }
    
    async def generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate prioritised recommendations based on analysis."""
        if not analysis:
            self.logger.error("Analysis data is None.")
            return []
        try:
            # Use asyncio.gather to execute recommendation functions concurrently
            tasks = [
                self._generate_technical_recommendations(analysis.get("technical_seo", {})),
                self._generate_on_page_recommendations(analysis.get("on_page_seo", {})),
                self._generate_off_page_recommendations(analysis.get("off_page_seo", {})),
            ]
            technical_recommendations, on_page_recommendations, off_page_recommendations = await asyncio.gather(*tasks)
            # Combine and sort recommendations
            all_recommendations = (
                technical_recommendations +
                on_page_recommendations +
                off_page_recommendations
            )
            return sorted(
                all_recommendations,
                key=lambda x: (x.get("priority_score", 0), x.get("impact", "")),
                reverse=True
            )
        except Exception as e:
            self.logger.exception(f"Error generating recommendations: {e}")
            return []




    @lru_cache(maxsize=100)    
    async def _generate_technical_recommendations(self, technical_analysis: Dict) -> List[Dict]:
        """Generate technical SEO recommendations efficiently."""
        if not technical_analysis:
            self.logger.warning("Technical analysis data is missing or invalid.")
            return []

        try:
            # Reduce the size of the analysis prompt to avoid overloading the LLM
            condensed_analysis = {}
            for key in ["site_speed", "mobile_optimisation", "indexation", "security", "structured_data"]:
                value = technical_analysis.get(key, "Summary not available")
                condensed_analysis[key] = str(value)[:250]

            prompt = f"""
            Based on this condensed technical SEO analysis:
            {json.dumps(condensed_analysis)}

            Generate prioritised recommendations for:
            1. Site speed optimisation
            2. Mobile optimisation
            3. Indexation improvements
            4. Security enhancements
            5. Structured data implementation

            Format as a JSON list with priority_score and impact for each recommendation.
            """

            return await self.llm.generate_recommendations(prompt)
        except Exception as e:
            self.logger.error(f"Error generating technical SEO recommendations: {e}")
            return []

    
    @lru_cache(maxsize=100)
    async def _generate_on_page_recommendations(self, on_page_analysis: Dict) -> List[Dict]:
        """generate on-page SEO recommendations"""
        
        prompt = f"""
        Based on this on-page SEO analysis:
        {json.dumps(on_page_analysis)}

        Generate prioritised recommendations for:
        1. Content Optimisation
        2. Keyword usage
        3. Meta tag improvements
        4. Internal linking
        5. Content structure

        Format as JSON list with priority_score and impact for each recommendation.
        """

        return await self.llm.generate_recommendations(prompt)
    @lru_cache(maxsize=100)
    async def _generate_off_page_recommendation(self, off_page_analysis: Dict) -> List[Dict]:
        """Generate off-page SEO recommendations"""
        prompt = f"""
        Based on this off-page SEO analysis:
        {json.dumps(off_page_analysis)}

        Generate prioritised recommendations for:
        1. Link building opportunities
        2. Brand signal improvements
        3. Social media presence
        4. Local SEO optimisation
        5. Autority building

        Format as JSON list with priority_score and impact for each recommendation.
        """

        return await self.llm.generate_recommendations(prompt)
    
    async def _analyse_keyword_usage(self, url: str) -> Dict:
        """Analyze keyword usage and optimization on the page"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract all text content
                text_content = soup.get_text(separator=' ', strip=True)
                
                # Extract title and description
                title = soup.title.string if soup.title else ''
                meta_description = soup.find('meta', attrs={'name': 'description'})
                description = meta_description['content'] if meta_description else ''
                
                # Extract h1 and other headings
                h1 = soup.find('h1')
                h1_text = h1.get_text(strip=True) if h1 else ''
                
                all_headings = []
                for i in range(1, 7):
                    for heading in soup.find_all(f'h{i}'):
                        all_headings.append(heading.get_text(strip=True))
                
                # Extract keywords from content
                words = re.findall(r'\b\w+\b', text_content.lower())
                word_count = len(words)
                
                # Create word frequency distribution
                word_freq = Counter(words)
                
                # Remove common stop words
                stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                for stop_word in stop_words:
                    if stop_word in word_freq:
                        del word_freq[stop_word]
                
                # Get most common words (potential keywords)
                potential_keywords = word_freq.most_common(10)
                
                # Extract keywords from title and headings
                title_words = set(re.findall(r'\b\w+\b', title.lower())) - stop_words
                heading_words = set(re.findall(r'\b\w+\b', ' '.join(all_headings).lower())) - stop_words
                
                # Check keyword presence in important elements
                top_keywords = [keyword for keyword, _ in potential_keywords[:5]]
                keyword_placement = {}
                
                for keyword in top_keywords:
                    keyword_placement[keyword] = {
                        'in_title': keyword in title.lower(),
                        'in_description': keyword in description.lower(),
                        'in_h1': keyword in h1_text.lower(),
                        'in_url': keyword in url.lower(),
                        'in_headings': any(keyword in heading.lower() for heading in all_headings),
                        'count': word_freq[keyword],
                        'density': (word_freq[keyword] / word_count) * 100 if word_count > 0 else 0
                    }
                
                return {
                    'word_count': word_count,
                    'potential_keywords': potential_keywords,
                    'title_keywords': list(title_words),
                    'heading_keywords': list(heading_words),
                    'keyword_placement': keyword_placement,
                    'keyword_score': self._calculate_keyword_score(keyword_placement, title_words, heading_words)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing keyword usage: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_keyword_score(self, keyword_placement, title_words, heading_words) -> float:
        """Calculate a score for keyword optimization"""
        score = 0
        
        # Check if there are keywords to analyze
        if not keyword_placement:
            return 0
        
        # Evaluate placement of top keywords
        placement_score = 0
        for keyword, data in keyword_placement.items():
            keyword_score = 0
            if data['in_title']:
                keyword_score += 2
            if data['in_description']:
                keyword_score += 1
            if data['in_h1']:
                keyword_score += 2
            if data['in_url']:
                keyword_score += 1
            if data['in_headings']:
                keyword_score += 1
            
            # Check density (ideal is 1-3%)
            density = data['density']
            if 1 <= density <= 3:
                keyword_score += 1
            elif density > 3:  # Potential keyword stuffing
                keyword_score -= 1
            
            placement_score += keyword_score
        
        # Average the scores
        if len(keyword_placement) > 0:
            placement_score = placement_score / len(keyword_placement)
            score += min(7, placement_score)  # Cap at 7 points
        
        # Consistency between title and headings
        common_keywords = title_words.intersection(heading_words)
        if len(common_keywords) >= 2:
            score += 2
        elif len(common_keywords) == 1:
            score += 1
        
        # Bonus point for having unique keywords
        if len(title_words) >= 3 and len(heading_words) >= 3:
            score += 1
        
        return max(0, min(10, score))
    
    async def _analyse_meta_tags(self, url: str) -> Dict:
        """Analyze meta tags implementation"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else None
                title_length = len(title) if title else 0
                
                # Extract meta description
                meta_description = soup.find('meta', attrs={'name': 'description'})
                description = meta_description['content'] if meta_description else None
                description_length = len(description) if description else 0
                
                # Extract meta keywords
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                keywords = meta_keywords['content'] if meta_keywords else None
                
                # Check canonical URL
                canonical = soup.find('link', rel='canonical')
                canonical_url = canonical['href'] if canonical else None
                
                # Check meta robots
                meta_robots = soup.find('meta', attrs={'name': 'robots'})
                robots_content = meta_robots['content'] if meta_robots else None
                
                # Check social meta tags
                og_title = soup.find('meta', attrs={'property': 'og:title'})
                og_description = soup.find('meta', attrs={'property': 'og:description'})
                og_image = soup.find('meta', attrs={'property': 'og:image'})
                
                twitter_card = soup.find('meta', attrs={'name': 'twitter:card'})
                twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
                twitter_description = soup.find('meta', attrs={'name': 'twitter:description'})
                twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
                
                return {
                    'title': {
                        'content': title,
                        'length': title_length,
                        'optimal_length': 50 <= title_length <= 60 if title else False
                    },
                    'description': {
                        'content': description,
                        'length': description_length,
                        'optimal_length': 140 <= description_length <= 160 if description else False
                    },
                    'keywords': keywords,
                    'canonical': canonical_url,
                    'robots': robots_content,
                    'social_tags': {
                        'open_graph': {
                            'title': og_title['content'] if og_title else None,
                            'description': og_description['content'] if og_description else None,
                            'image': og_image['content'] if og_image else None
                        },
                        'twitter': {
                            'card': twitter_card['content'] if twitter_card else None,
                            'title': twitter_title['content'] if twitter_title else None,
                            'description': twitter_description['content'] if twitter_description else None,
                            'image': twitter_image['content'] if twitter_image else None
                        }
                    },
                    'meta_tags_score': self._calculate_meta_tags_score(title, description, keywords, canonical_url, robots_content,
                                                                  og_title, og_description, og_image, 
                                                                  twitter_card, twitter_title, twitter_description, twitter_image)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing meta tags: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_meta_tags_score(self, title, description, keywords, canonical, robots, 
                             og_title, og_description, og_image, 
                             twitter_card, twitter_title, twitter_description, twitter_image) -> float:
        """Calculate a score for meta tags implementation"""
        score = 0
        
        # Title assessment
        if title:
            score += 1
            if 50 <= len(title) <= 60:
                score += 1
        
        # Description assessment
        if description:
            score += 1
            if 140 <= len(description) <= 160:
                score += 1
        
        # Canonical URL
        if canonical:
            score += 1
        
        # Meta robots (having it explicitly set is good)
        if robots:
            score += 0.5
        
        # Open Graph tags
        og_score = 0
        if og_title:
            og_score += 0.5
        if og_description:
            og_score += 0.5
        if og_image:
            og_score += 0.5
        score += og_score
        
        # Twitter Card tags
        twitter_score = 0
        if twitter_card:
            twitter_score += 0.5
        if twitter_title:
            twitter_score += 0.25
        if twitter_description:
            twitter_score += 0.25
        if twitter_image:
            twitter_score += 0.5
        score += twitter_score
        
        # Bonus for full social implementation
        if og_score >= 1.5 and twitter_score >= 1.5:
            score += 1
        
        return max(0, min(10, score))
    
    async def _analyse_heading_structure(self, url: str) -> Dict:
        """Analyze heading structure and hierarchy"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract headings
                headings = {}
                for level in range(1, 7):
                    heading_tags = soup.find_all(f'h{level}')
                    headings[f'h{level}'] = [
                        {
                            'text': h.get_text(strip=True),
                            'length': len(h.get_text(strip=True)),
                            'position': self._get_element_position(h, soup)
                        }
                        for h in heading_tags
                    ]
                
                # Calculate heading counts
                heading_counts = {level: len(heads) for level, heads in headings.items()}
                
                # Check heading hierarchy
                hierarchy_issues = []
                
                # Check for H1
                if heading_counts['h1'] == 0:
                    hierarchy_issues.append("Missing H1 heading")
                elif heading_counts['h1'] > 1:
                    hierarchy_issues.append(f"Multiple H1 headings ({heading_counts['h1']})")
                
                # Check for skipped levels
                for i in range(1, 5):  # Check h1 to h5
                    next_level = i + 1
                    if heading_counts[f'h{i}'] == 0 and heading_counts[f'h{next_level}'] > 0:
                        hierarchy_issues.append(f"Skipped H{i} level (using H{next_level} without H{i})")
                
                # Check sequence (simplified)
                all_headings = []
                for level in range(1, 7):
                    for h in headings[f'h{level}']:
                        all_headings.append({
                            'level': level,
                            'position': h['position'],
                            'text': h['text']
                        })
                
                # Sort headings by position
                all_headings.sort(key=lambda h: h['position'])
                
                # Check for hierarchy sequence issues
                sequence_issues = []
                for i in range(1, len(all_headings)):
                    current = all_headings[i]
                    previous = all_headings[i-1]
                    
                    # Check if level jumps by more than one
                    if current['level'] > previous['level'] + 1:
                        sequence_issues.append(f"Jump from H{previous['level']} to H{current['level']}")
                
                return {
                    'heading_counts': heading_counts,
                    'headings': headings,
                    'hierarchy_issues': hierarchy_issues,
                    'sequence_issues': sequence_issues,
                    'heading_structure_score': self._calculate_heading_structure_score(heading_counts, hierarchy_issues, sequence_issues)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing heading structure: {str(e)}")
            return {"error": str(e)}
    
    def _get_element_position(self, element, soup) -> int:
        """Get the position of an element in the document"""
        all_elements = soup.find_all()
        for i, el in enumerate(all_elements):
            if el is element:
                return i
        return 0
    
    def _calculate_heading_structure_score(self, heading_counts, hierarchy_issues, sequence_issues) -> float:
        """Calculate a score for heading structure"""
        score = 10  # Start with perfect score and deduct for issues
        
        # H1 issues
        if heading_counts['h1'] == 0:
            score -= 3  # Major issue
        elif heading_counts['h1'] > 1:
            score -= 2  # Significant issue
        
        # Deduct for hierarchy issues
        score -= len(hierarchy_issues) * 1.5
        
        # Deduct for sequence issues (less severe)
        score -= len(sequence_issues) * 0.5
        
        # Check if heading structure is deep enough
        depth = sum(1 for level, count in heading_counts.items() if count > 0)
        if depth <= 1:
            score -= 2  # Only one heading level
        elif depth == 2:
            score -= 1  # Only two heading levels
        
        # Bonus for good distribution
        if heading_counts['h1'] == 1 and heading_counts['h2'] >= 2 and heading_counts['h3'] >= 2:
            score += 1
        
        return max(0, min(10, score))
    
    async def _analyse_user_experience(self, url: str) -> Dict:
        """Analyze user experience factors"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check readability
                text_content = soup.get_text(separator=' ', strip=True)
                word_count = len(text_content.split())
                
                # Check for multimedia elements
                images = soup.find_all('img')
                videos = soup.find_all(['video', 'iframe'])
                
                # Check for interactive elements
                forms = soup.find_all('form')
                buttons = soup.find_all('button')
                interactive_elements = soup.find_all(['select', 'input', 'textarea'])
                
                # Check for popups
                popups = soup.find_all(class_=lambda c: c and any(term in str(c).lower() for term in ['popup', 'modal', 'overlay']))
                
                # Check for mobile-friendly features
                viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
                responsive_design = viewport_meta and 'width=device-width' in viewport_meta.get('content', '')
                
                # Check for navigation elements
                navigation = soup.find('nav') or soup.find(class_=lambda c: c and 'nav' in str(c).lower())
                menu = soup.find(class_=lambda c: c and 'menu' in str(c).lower())
                
                # Check for internal links
                internal_links = []
                page_domain = urlparse(url).netloc
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if not href.startswith(('http', 'https', '//')):
                        internal_links.append(href)
                    else:
                        link_domain = urlparse(href).netloc
                        if link_domain == page_domain:
                            internal_links.append(href)
                
                return {
                    'content_metrics': {
                        'word_count': word_count,
                        'readability_score': self._estimate_readability_score(text_content)
                    },
                    'multimedia': {
                        'images': len(images),
                        'videos': len(videos)
                    },
                    'interactivity': {
                        'forms': len(forms),
                        'buttons': len(buttons),
                        'interactive_elements': len(interactive_elements)
                    },
                    'navigation': {
                        'has_nav_menu': bool(navigation or menu),
                        'internal_links': len(internal_links)
                    },
                    'mobile_friendly': responsive_design,
                    'potential_issues': {
                        'popups': len(popups) > 0,
                        'excessive_ads': self._check_for_excessive_ads(soup)
                    },
                    'user_experience_score': self._calculate_ux_score(word_count, len(images), len(videos), len(forms), 
                                                             len(internal_links), responsive_design, len(popups))
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing user experience: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_readability_score(self, text: str) -> Dict:
        """Estimate readability metrics for text"""
        # Simplified implementation
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return {'score': 0, 'level': 'Unknown'}
        
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count == 0:
            return {'score': 0, 'level': 'Unknown'}
        
        avg_words_per_sentence = word_count / sentence_count
        
        # Simplified Flesch Reading Ease calculation
        syllable_count = sum(self._count_syllables(word) for word in words)
        if word_count > 0:
            syllables_per_word = syllable_count / word_count
        else:
            syllables_per_word = 0
        
        # Approximate Flesch Reading Ease
        flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * syllables_per_word)
        
        # Determine reading level
        reading_level = 'Unknown'
        if flesch_score >= 90:
            reading_level = 'Very Easy'
        elif flesch_score >= 80:
            reading_level = 'Easy'
        elif flesch_score >= 70:
            reading_level = 'Fairly Easy'
        elif flesch_score >= 60:
            reading_level = 'Standard'
        elif flesch_score >= 50:
            reading_level = 'Fairly Difficult'
        elif flesch_score >= 30:
            reading_level = 'Difficult'
        else:
            reading_level = 'Very Difficult'
        
        return {
            'score': round(flesch_score, 2),
            'level': reading_level,
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_syllables_per_word': round(syllables_per_word, 2)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified algorithm)"""
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Remove trailing silent e
        if word.endswith('e'):
            word = word[:-1]
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        return max(1, count)  # Every word has at least one syllable
    
    def _check_for_excessive_ads(self, soup: BeautifulSoup) -> bool:
        """Check for signs of excessive advertisements"""
        ad_indicators = ['ad', 'ads', 'advert', 'banner', 'sponsor', 'promotion']
        ad_elements = soup.find_all(class_=lambda c: c and any(ind in c.lower() for ind in ad_indicators))
        ad_iframes = soup.find_all('iframe')
        
        # Count potential ad elements
        ad_count = len(ad_elements) + len(ad_iframes)
        
        # Check for common ad network scripts
        ad_scripts = 0
        for script in soup.find_all('script', src=True):
            src = script['src'].lower()
            if any(network in src for network in ['doubleclick', 'adsense', 'adroll', 'adform']):
                ad_scripts += 1
        
        # Consider excessive if more than 5 potential ad elements or 3 ad scripts
        return ad_count > 5 or ad_scripts > 3
    
    def _calculate_ux_score(self, word_count, image_count, video_count, form_count, internal_link_count, responsive, popup_count) -> float:
        """Calculate a score for user experience"""
        score = 5  # Start with an average score
        
        # Content assessment
        if word_count > 300:
            score += 1
        
        # Multimedia elements enhance UX
        if image_count > 0:
            score += min(1, image_count * 0.2)  # Up to 1 point for images
        if video_count > 0:
            score += min(1, video_count * 0.5)  # Up to 1 point for videos
        
        # Interactive elements
        if form_count > 0:
            score += 0.5  # Forms can enhance engagement
        
        # Navigation quality
        if internal_link_count > 5:
            score += 1  # Good internal linking
        
        # Mobile optimization is important
        if responsive:
            score += 1.5
        else:
            score -= 2  # Major issue in modern web
        
        # Popups can hurt UX
        if popup_count > 0:
            score -= min(2, popup_count * 0.5)  # Deduct up to 2 points
        
        return max(0, min(10, score))
    
    async def _analyse_multimedia(self, url: str) -> Dict:
        """Analyze multimedia elements on the page"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Analyze images
                images = soup.find_all('img')
                image_data = []
                
                for img in images:
                    alt_text = img.get('alt', '')
                    src = img.get('src', '')
                    width = img.get('width', '')
                    height = img.get('height', '')
                    lazy_loading = img.get('loading') == 'lazy'
                    
                    # Approximate file size calculation
                    file_size = None
                    if src.startswith(('http://', 'https://')):
                        try:
                            async with self.session.head(src) as img_response:
                                content_length = img_response.headers.get('Content-Length')
                                if content_length:
                                    file_size = int(content_length)
                        except Exception:
                            pass
                    
                    image_data.append({
                        'src': src,
                        'alt_text': alt_text,
                        'has_alt': bool(alt_text),
                        'dimensions_specified': bool(width or height),
                        'lazy_loading': lazy_loading,
                        'file_size': file_size
                    })
                
                # Analyze videos
                videos = soup.find_all(['video', 'iframe'])
                video_data = []
                
                for video in videos:
                    if video.name == 'video':
                        src = video.get('src', '')
                        controls = video.has_attr('controls')
                        autoplay = video.has_attr('autoplay')
                        width = video.get('width', '')
                        height = video.get('height', '')
                        
                        video_data.append({
                            'type': 'html5',
                            'src': src,
                            'has_controls': controls,
                            'autoplay': autoplay,
                            'dimensions_specified': bool(width or height)
                        })
                    else:  # iframe
                        src = video.get('src', '')
                        width = video.get('width', '')
                        height = video.get('height', '')
                        provider = 'unknown'
                        
                        if 'youtube' in src:
                            provider = 'youtube'
                        elif 'vimeo' in src:
                            provider = 'vimeo'
                        elif 'dailymotion' in src:
                            provider = 'dailymotion'
                        
                        video_data.append({
                            'type': 'embed',
                            'provider': provider,
                            'src': src,
                            'dimensions_specified': bool(width or height)
                        })
                
                # Check for image optimization issues
                image_issues = []
                for img in image_data:
                    if not img['has_alt']:
                        image_issues.append(f"Missing alt text: {img['src']}")
                    if img['file_size'] and img['file_size'] > 200000:  # 200KB
                        image_issues.append(f"Large image: {img['src']} ({img['file_size'] / 1024:.1f}KB)")
                    if not img['dimensions_specified']:
                        image_issues.append(f"Missing dimensions: {img['src']}")
                
                return {
                    'images': {
                        'count': len(images),
                        'with_alt': sum(1 for img in image_data if img['has_alt']),
                        'lazy_loaded': sum(1 for img in image_data if img['lazy_loading']),
                        'issues': image_issues
                    },
                    'videos': {
                        'count': len(videos),
                        'by_type': {
                            'html5': sum(1 for v in video_data if v['type'] == 'html5'),
                            'embed': sum(1 for v in video_data if v['type'] == 'embed')
                        },
                        'by_provider': Counter(v['provider'] for v in video_data if v['type'] == 'embed')
                    },
                    'multimedia_score': self._calculate_multimedia_score(image_data, video_data, image_issues)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing multimedia: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_multimedia_score(self, image_data, video_data, image_issues) -> float:
        """Calculate a score for multimedia implementation"""
        score = 5  # Start with a middle score
        
        # Image optimization assessment
        if image_data:
            alt_text_ratio = sum(1 for img in image_data if img['has_alt']) / len(image_data)
            score += alt_text_ratio * 2  # Up to 2 points for alt text
            
            lazy_load_ratio = sum(1 for img in image_data if img['lazy_loading']) / len(image_data)
            score += lazy_load_ratio * 1  # Up to 1 point for lazy loading
            
            dimensions_ratio = sum(1 for img in image_data if img['dimensions_specified']) / len(image_data)
            score += dimensions_ratio * 1  # Up to 1 point for dimensions
            
            # Deduct for issues
            score -= min(2, len(image_issues) * 0.25)  # Up to 2 points deduction
        
        # Video implementation assessment
        if video_data:
            # HTML5 videos with controls are good
            html5_with_controls = sum(1 for v in video_data if v['type'] == 'html5' and v.get('has_controls', False))
            if html5_with_controls > 0:
                score += 0.5
            
            # Videos with dimensions specified
            dimensions_ratio = sum(1 for v in video_data if v['dimensions_specified']) / len(video_data)
            score += dimensions_ratio * 0.5  # Up to 0.5 points
        
        return max(0, min(10, score))

    async def _analyse_brand_signals(self, url: str) -> Dict:
        """Analyze brand signals and presence"""
        try:
            domain = urlparse(url).netloc
            brand_name = domain.split('.')[0]  # Simple assumption for brand name
            
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check for brand elements
                logo = soup.find('img', alt=lambda alt: alt and brand_name.lower() in alt.lower())
                logo_presence = logo is not None
                
                # Check for brand mentions in content
                text_content = soup.get_text(strip=True).lower()
                brand_mentions = text_content.count(brand_name.lower())
                
                # Check for branded links (like social profiles)
                social_profiles = []
                social_platforms = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com']
                for link in soup.find_all('a', href=True):
                    href = link['href'].lower()
                    for platform in social_platforms:
                        if platform in href:
                            try:
                                # Check if the link contains the brand name
                                platform_path = urlparse(href).path.strip('/')
                                if brand_name.lower() in platform_path:
                                    social_profiles.append({
                                        'platform': platform,
                                        'url': href
                                    })
                            except Exception:
                                pass
                
                # Check for contact information
                email = None
                phone = None
                address = None
                
                # Look for email
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                email_matches = re.findall(email_pattern, text_content)
                if email_matches:
                    email = email_matches[0]
                
                # Look for phone
                phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
                phone_matches = re.findall(phone_pattern, text_content)
                if phone_matches:
                    phone = phone_matches[0]
                
                # Look for structured address elements
                address_elements = soup.find_all(itemtype=lambda x: x and 'postaladdress' in str(x).lower())
                if address_elements:
                    address = address_elements[0].get_text(strip=True)
                
                # Check for other brand elements
                copyright_notice = soup.find(string=lambda text: text and '©' in text)
                has_copyright = copyright_notice is not None
                
                return {
                    'brand_name': brand_name,
                    'logo_present': logo_presence,
                    'brand_mentions': brand_mentions,
                    'social_profiles': social_profiles,
                    'contact_info': {
                        'email': email,
                        'phone': phone,
                        'address': address
                    },
                    'branding_elements': {
                        'copyright_notice': has_copyright
                    },
                    'brand_signals_score': self._calculate_brand_signals_score(logo_presence, brand_mentions, 
                                                                         len(social_profiles), email, phone, has_copyright)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing brand signals: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_brand_signals_score(self, has_logo, brand_mentions, social_profile_count, has_email, has_phone, has_copyright) -> float:
        """Calculate a score for brand signals"""
        score = 0
        
        # Logo is a basic branding element
        if has_logo:
            score += 2
        
        # Brand mentions in content
        if brand_mentions > 0:
            mentions_score = min(brand_mentions / 2, 2)  # Up to 2 points for mentions
            score += mentions_score
        
        # Social profiles
        score += min(social_profile_count, 2)  # Up to 2 points for social profiles
        
        # Contact information
        if has_email:
            score += 1
        if has_phone:
            score += 1
        
        # Copyright notice
        if has_copyright:
            score += 1
        
        # Bonus for comprehensive branding
        if has_logo and brand_mentions >= 3 and social_profile_count >= 2 and has_email and has_phone and has_copyright:
            score += 1  # Bonus point for having everything
        
        return max(0, min(10, score))
    
    async def _analyse_social_presence(self, url: str) -> Dict:
        """Analyze social media presence and engagement"""
        try:
            domain = urlparse(url).netloc
            brand_name = domain.split('.')[0]  # Simple assumption for brand name
            
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find social media links
                social_links = {}
                social_platforms = {
                    'facebook': ['facebook.com', 'fb.com'],
                    'twitter': ['twitter.com', 'x.com'],
                    'instagram': ['instagram.com', 'insta.com'],
                    'linkedin': ['linkedin.com'],
                    'youtube': ['youtube.com', 'youtu.be'],
                    'pinterest': ['pinterest.com'],
                    'tiktok': ['tiktok.com'],
                    'github': ['github.com'],
                    'medium': ['medium.com']
                }
                
                for link in soup.find_all('a', href=True):
                    href = link['href'].lower()
                    for platform, domains in social_platforms.items():
                        for domain in domains:
                            if domain in href:
                                social_links[platform] = href
                                break
                
                # Check for social sharing buttons
                share_buttons = []
                share_keywords = ['share', 'tweet', 'follow', 'like', 'subscribe']
                
                for element in soup.find_all(['a', 'button', 'div']):
                    element_text = element.get_text(strip=True).lower()
                    element_class = ' '.join(element.get('class', [])).lower()
                    
                    if any(keyword in element_text for keyword in share_keywords) or \
                       any(keyword in element_class for keyword in share_keywords) or \
                       any(platform in element_class for platform in social_platforms.keys()):
                        share_buttons.append({
                            'text': element_text,
                            'type': element.name
                        })
                
                # Check for social embeds
                twitter_embeds = soup.find_all('blockquote', class_='twitter-tweet')
                facebook_embeds = soup.find_all('div', class_=lambda c: c and 'fb-' in str(c))
                instagram_embeds = soup.find_all('blockquote', class_='instagram-media')
                youtube_embeds = soup.find_all('iframe', src=lambda s: s and 'youtube.com/embed' in s)
                
                embeds = {
                    'twitter': len(twitter_embeds),
                    'facebook': len(facebook_embeds),
                    'instagram': len(instagram_embeds),
                    'youtube': len(youtube_embeds)
                }
                
                # Check for social meta tags
                og_tags = {
                    tag.get('property'): tag.get('content')
                    for tag in soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
                }
                
                twitter_tags = {
                    tag.get('name'): tag.get('content')
                    for tag in soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
                }
                
                return {
                    'social_profiles': {
                        'platforms': list(social_links.keys()),
                        'count': len(social_links),
                        'urls': social_links
                    },
                    'social_sharing': {
                        'has_share_buttons': len(share_buttons) > 0,
                        'buttons_count': len(share_buttons)
                    },
                    'social_embeds': embeds,
                    'social_meta_tags': {
                        'open_graph': bool(og_tags),
                        'twitter_cards': bool(twitter_tags)
                    },
                    'social_presence_score': self._calculate_social_presence_score(len(social_links), len(share_buttons), 
                                                                          sum(embeds.values()), bool(og_tags), bool(twitter_tags))
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing social presence: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_social_presence_score(self, profile_count, share_buttons_count, embed_count, has_og, has_twitter) -> float:
        """Calculate a score for social media presence"""
        score = 0
        
        # Social profiles
        if profile_count > 0:
            score += min(profile_count, 3)  # Up to 3 points for social profiles
        
        # Social sharing buttons
        if share_buttons_count > 0:
            score += min(share_buttons_count / 2, 2)  # Up to 2 points for share buttons
        
        # Social embeds
        if embed_count > 0:
            score += min(embed_count, 2)  # Up to 2 points for embeds
        
        # Social meta tags
        if has_og:
            score += 1.5
        if has_twitter:
            score += 1.5
        
        return max(0, min(10, score))
    
    async def _analyse_local_seo(self, url: str) -> Dict:
        """Analyze local SEO implementation"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check for structured data relevant to local SEO
                local_business_schema = None
                org_schema = None
                
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        data = json.loads(script.string)
                        if isinstance(data, dict):
                            if data.get('@type') == 'LocalBusiness':
                                local_business_schema = data
                            elif data.get('@type') == 'Organization':
                                org_schema = data
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if item.get('@type') == 'LocalBusiness':
                                        local_business_schema = item
                                    elif item.get('@type') == 'Organization':
                                        org_schema = item
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # Check for business address
                address_elements = soup.find_all(itemtype=lambda x: x and 'postaladdress' in str(x).lower())
                address_text = None
                if address_elements:
                    address_text = address_elements[0].get_text(strip=True)
                
                # Check for microdata address
                microdata_address = soup.find(itemtype=lambda x: x and 'postaladdress' in str(x).lower())
                
                # Check for phone number
                phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
                text_content = soup.get_text(strip=True)
                phone_matches = re.findall(phone_pattern, text_content)
                phone_number = phone_matches[0] if phone_matches else None
                
                # Check for maps embed
                maps_embed = soup.find('iframe', src=lambda s: s and ('maps.google' in s or 'maps.googleapis' in s))
                
                # Check for business hours
                hours_element = soup.find(class_=lambda c: c and ('hours' in str(c).lower() or 'schedule' in str(c).lower()))
                has_hours = hours_element is not None
                
                # Check for local keywords in content
                location_indicators = soup.find_all(string=lambda s: s and any(city in str(s).lower() for city in [
                    'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio', 
                    'san diego', 'dallas', 'san jose', 'austin', 'boston', 'seattle', 'denver', 'detroit'
                ]))
                
                # Results from the analysis
                return {
                    'local_business_schema': local_business_schema is not None,
                    'organization_schema': org_schema is not None,
                    'address': {
                        'text': address_text,
                        'structured': microdata_address is not None
                    },
                    'phone_number': phone_number is not None,
                    'maps_integration': maps_embed is not None,
                    'business_hours': has_hours,
                    'local_indicators': len(location_indicators),
                    'local_seo_score': self._calculate_local_seo_score(local_business_schema, org_schema, 
                                                               microdata_address, phone_number, maps_embed, has_hours)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing local SEO: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_local_seo_score(self, local_business_schema, org_schema, has_address, has_phone, has_map, has_hours) -> float:
        """Calculate a score for local SEO implementation"""
        score = 0
        
        # LocalBusiness schema is very important
        if local_business_schema is not None:
            score += 4
        # Organization schema is good but not as specific
        elif org_schema is not None:
            score += 2
        
        # Address information is critical
        if has_address is not None:
            score += 2
        
        # Phone number is important
        if has_phone is not None:
            score += 1
        
        # Maps integration
        if has_map is not None:
            score += 2
        
        # Business hours
        if has_hours:
            score += 1
        
        return max(0, min(10, score))
    
    async def _analyse_competitors(self, url: str) -> Dict:
        """
        Perform a basic competitor analysis.
        Note: This is a simplified implementation as true competitor analysis 
        would require more sophisticated tools and data sources.
        """
        try:
            domain = urlparse(url).netloc
            
            # This would typically use a service like SEMrush, Ahrefs, Moz, etc.
            # For this implementation, we'll do a limited analysis
            
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract external links that could be competitors
                external_domains = set()
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith(('http://', 'https://')) and domain not in href:
                        link_domain = urlparse(href).netloc
                        if '.' in link_domain and not any(pattern in link_domain for pattern in [
                            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
                            'youtube.com', 'google.com', 'pinterest.com', 'amazon.com'
                        ]):
                            external_domains.add(link_domain)
                
                # Extract potential industry keywords
                text_content = soup.get_text(separator=' ', strip=True)
                words = re.findall(r'\b\w+\b', text_content.lower())
                word_freq = Counter(words)
                
                # Remove common stop words
                stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                for stop_word in stop_words:
                    if stop_word in word_freq:
                        del word_freq[stop_word]
                
                # Get potential industry keywords
                industry_keywords = [word for word, count in word_freq.most_common(15)]
                
                return {
                    'potential_competitors': list(external_domains)[:10],  # Limit to top 10
                    'industry_keywords': industry_keywords,
                    'competitor_analysis': {
                        'note': 'Limited competitor analysis. For comprehensive analysis, use specialized SEO tools.',
                        'comparison_metrics': {
                            'domain_authority': 'Not available in this implementation',
                            'backlink_comparison': 'Not available in this implementation',
                            'keyword_overlap': 'Not available in this implementation'
                        }
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing competitors: {str(e)}")
            return {"error": str(e)}
    
    async def _analyse_authority(self, url: str) -> Dict:
        """
        Analyze website authority metrics.
        Note: True authority metrics require external data sources like Moz, Ahrefs, etc.
        """
        try:
            domain = urlparse(url).netloc
            
            # This is a simplified implementation
            # Real implementation would use APIs from Moz, Ahrefs, Majestic, etc.
            
            # Check for authority indicators on the page
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for trust indicators
                about_page_link = soup.find('a', href=lambda h: h and 'about' in h.lower())
                has_about_page = about_page_link is not None
                
                privacy_policy_link = soup.find('a', href=lambda h: h and 'privacy' in h.lower())
                has_privacy_policy = privacy_policy_link is not None
                
                terms_link = soup.find('a', href=lambda h: h and ('terms' in h.lower() or 'conditions' in h.lower()))
                has_terms = terms_link is not None
                
                # Check for author information
                author_elements = soup.find_all(class_=lambda c: c and 'author' in str(c).lower())
                has_author_info = len(author_elements) > 0
                
                # Check for publication dates
                date_elements = soup.find_all(class_=lambda c: c and ('date' in str(c).lower() or 'published' in str(c).lower()))
                has_dates = len(date_elements) > 0
                
                # Check for citations
                citation_elements = soup.find_all(['cite', 'blockquote'])
                has_citations = len(citation_elements) > 0
                
                # Check for quality outbound links
                quality_domains = ['edu', 'gov', 'org']
                quality_outbound_links = []
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith(('http://', 'https://')) and domain not in href:
                        link_domain = urlparse(href).netloc
                        if any(link_domain.endswith('.' + domain_type) for domain_type in quality_domains):
                            quality_outbound_links.append(href)
                
                return {
                    'authority_indicators': {
                        'has_about_page': has_about_page,
                        'has_privacy_policy': has_privacy_policy,
                        'has_terms': has_terms,
                        'has_author_info': has_author_info,
                        'has_publication_dates': has_dates,
                        'has_citations': has_citations,
                        'quality_outbound_links': len(quality_outbound_links)
                    },
                    'estimated_metrics': {
                        'note': 'These metrics would typically come from third-party tools.',
                        'domain_authority': 'Not available in this implementation',
                        'page_authority': 'Not available in this implementation',
                        'trust_flow': 'Not available in this implementation',
                        'citation_flow': 'Not available in this implementation'
                    },
                    'authority_score': self._calculate_authority_score(has_about_page, has_privacy_policy, 
                                                                has_terms, has_author_info, has_dates, 
                                                                has_citations, len(quality_outbound_links))
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing authority: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_authority_score(self, has_about, has_privacy, has_terms, has_author, has_dates, has_citations, quality_links_count) -> float:
        """Calculate a score for website authority indicators"""
        score = 0
        
        # Basic trust pages
        if has_about:
            score += 1
        if has_privacy:
            score += 1
        if has_terms:
            score += 1
        
        # Content quality indicators
        if has_author:
            score += 1.5
        if has_dates:
            score += 1
        if has_citations:
            score += 1.5
        
        # Quality outbound links
        if quality_links_count > 0:
            score += min(quality_links_count, 3)  # Up to 3 points
        
        return max(0, min(10, score))
    
    async def _analyse_trust_signals(self, url: str) -> Dict:
        """Analyze trust signals on the website"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check for SSL (HTTPS)
                is_https = urlparse(url).scheme == 'https'
                
                # Check for trust badges
                badge_keywords = ['secure', 'trust', 'verified', 'guarantee', 'certification', 'accredited', 'seal']
                trust_badges = []
                
                for img in soup.find_all('img'):
                    alt_text = img.get('alt', '').lower()
                    src = img.get('src', '').lower()
                    if any(keyword in alt_text or keyword in src for keyword in badge_keywords):
                        trust_badges.append({
                            'alt': img.get('alt', ''),
                            'src': img.get('src', '')
                        })
                
                # Check for testimonials
                testimonial_keywords = ['testimonial', 'review', 'feedback', 'rating', 'client', 'customer']
                testimonial_sections = []
                
                for element in soup.find_all(['div', 'section', 'article']):
                    element_text = element.get_text(strip=True).lower()
                    element_class = ' '.join(element.get('class', [])).lower()
                    
                    if any(keyword in element_text for keyword in share_keywords) or \
                       any(keyword in element_class for keyword in share_keywords) or \
                       any(platform in element_class for platform in social_platforms.keys()):
                        share_buttons.append({
                            'text': element_text,
                            'type': element.name
                        })
                
                # Check for social embeds
                twitter_embeds = soup.find_all('blockquote', class_='twitter-tweet')
                facebook_embeds = soup.find_all('div', class_=lambda c: c and 'fb-' in str(c))
                instagram_embeds = soup.find_all('blockquote', class_='instagram-media')
                youtube_embeds = soup.find_all('iframe', src=lambda s: s and 'youtube.com/embed' in s)
                
                embeds = {
                    'twitter': len(twitter_embeds),
                    'facebook': len(facebook_embeds),
                    'instagram': len(instagram_embeds),
                    'youtube': len(youtube_embeds)
                }
                
                # Check for social meta tags
                og_tags = {
                    tag.get('property'): tag.get('content')
                    for tag in soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
                }
                
                twitter_tags = {
                    tag.get('name'): tag.get('content')
                    for tag in soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
                }
                
                return {
                    'social_profiles': {
                        'platforms': list(social_links.keys()),
                        'count': len(social_links),
                        'urls': social_links
                    },
                    'social_sharing': {
                        'has_share_buttons': len(share_buttons) > 0,
                        'buttons_count': len(share_buttons)
                    },
                    'social_embeds': embeds,
                    'social_meta_tags': {
                        'open_graph': bool(og_tags),
                        'twitter_cards': bool(twitter_tags)
                    },
                    'social_presence_score': self._calculate_social_presence_score(len(social_links), len(share_buttons), 
                                                                          sum(embeds.values()), bool(og_tags), bool(twitter_tags))
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing social presence: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_social_presence_score(self, profile_count, share_buttons_count, embed_count, has_og, has_twitter) -> float:
        """Calculate a score for social media presence"""
        score = 0
        
        # Social profiles
        if profile_count > 0:
            score += min(profile_count, 3)  # Up to 3 points for social profiles
        
        # Social sharing buttons
        if share_buttons_count > 0:
            score += min(share_buttons_count / 2, 2)  # Up to 2 points for share buttons
        
        # Social embeds
        if embed_count > 0:
            score += min(embed_count, 2)  # Up to 2 points for embeds
        
        # Social meta tags
        if has_og:
            score += 1.5
        if has_twitter:
            score += 1.5
        
        return max(0, min(10, score))
    
    async def _analyse_local_seo(self, url: str) -> Dict:
        """Analyze local SEO implementation"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check for structured data relevant to local SEO
                local_business_schema = None
                org_schema = None
                
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        data = json.loads(script.string)
                        if isinstance(data, dict):
                            if data.get('@type') == 'LocalBusiness':
                                local_business_schema = data
                            elif data.get('@type') == 'Organization':
                                org_schema = data
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if item.get('@type') == 'LocalBusiness':
                                        local_business_schema = item
                                    elif item.get('@type') == 'Organization':
                                        org_schema = item
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # Check for business address
                address_elements = soup.find_all(itemtype=lambda x: x and 'postaladdress' in str(x).lower())
                address_text = None
                if address_elements:
                    address_text = address_elements[0].get_text(strip=True)
                
                # Check for microdata address
                microdata_address = soup.find(itemtype=lambda x: x and 'postaladdress' in str(x).lower())
                
                # Check for phone number
                phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
                text_content = soup.get_text(strip=True)
                phone_matches = re.findall(phone_pattern, text_content)
                phone_number = phone_matches[0] if phone_matches else None
                
                # Check for maps embed
                maps_embed = soup.find('iframe', src=lambda s: s and ('maps.google' in s or 'maps.googleapis' in s))
                
                # Check for business hours
                hours_element = soup.find(class_=lambda c: c and ('hours' in str(c).lower() or 'schedule' in str(c).lower()))
                has_hours = hours_element is not None
                
                # Check for local keywords in content
                location_indicators = soup.find_all(string=lambda s: s and any(city in str(s).lower() for city in [
                    'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio', 
                    'san diego', 'dallas', 'san jose', 'austin', 'boston', 'seattle', 'denver', 'detroit'
                ]))
                
                # Results from the analysis
                return {
                    'local_business_schema': local_business_schema is not None,
                    'organization_schema': org_schema is not None,
                    'address': {
                        'text': address_text,
                        'structured': microdata_address is not None
                    },
                    'phone_number': phone_number is not None,
                    'maps_integration': maps_embed is not None,
                    'business_hours': has_hours,
                    'local_indicators': len(location_indicators),
                    'local_seo_score': self._calculate_local_seo_score(local_business_schema, org_schema, 
                                                               microdata_address, phone_number, maps_embed, has_hours)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing local SEO: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_local_seo_score(self, local_business_schema, org_schema, has_address, has_phone, has_map, has_hours) -> float:
        """Calculate a score for local SEO implementation"""
        score = 0
        
        # LocalBusiness schema is very important
        if local_business_schema is not None:
            score += 4
        # Organization schema is good but not as specific
        elif org_schema is not None:
            score += 2
        
        # Address information is critical
        if has_address is not None:
            score += 2
        
        # Phone number is important
        if has_phone is not None:
            score += 1
        
        # Maps integration
        if has_map is not None:
            score += 2
        
        # Business hours
        if has_hours:
            score += 1
        
        return max(0, min(10, score))
    
    async def _analyse_competitors(self, url: str) -> Dict:
        """
        Perform a basic competitor analysis.
        Note: This is a simplified implementation as true competitor analysis 
        would require more sophisticated tools and data sources.
        """
        try:
            domain = urlparse(url).netloc
            
            # This would typically use a service like SEMrush, Ahrefs, Moz, etc.
            # For this implementation, we'll do a limited analysis
            
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract external links that could be competitors
                external_domains = set()
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith(('http://', 'https://')) and domain not in href:
                        link_domain = urlparse(href).netloc
                        if '.' in link_domain and not any(pattern in link_domain for pattern in [
                            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
                            'youtube.com', 'google.com', 'pinterest.com', 'amazon.com'
                        ]):
                            external_domains.add(link_domain)
                
                # Extract potential industry keywords
                text_content = soup.get_text(separator=' ', strip=True)
                words = re.findall(r'\b\w+\b', text_content.lower())
                word_freq = Counter(words)
                
                # Remove common stop words
                stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                for stop_word in stop_words:
                    if stop_word in word_freq:
                        del word_freq[stop_word]
                
                # Get potential industry keywords
                industry_keywords = [word for word, count in word_freq.most_common(15)]
                
                return {
                    'potential_competitors': list(external_domains)[:10],  # Limit to top 10
                    'industry_keywords': industry_keywords,
                    'competitor_analysis': {
                        'note': 'Limited competitor analysis. For comprehensive analysis, use specialized SEO tools.',
                        'comparison_metrics': {
                            'domain_authority': 'Not available in this implementation',
                            'backlink_comparison': 'Not available in this implementation',
                            'keyword_overlap': 'Not available in this implementation'
                        }
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing competitors: {str(e)}")
            return {"error": str(e)}
    
    async def _analyse_authority(self, url: str) -> Dict:
        """
        Analyze website authority metrics.
        Note: True authority metrics require external data sources like Moz, Ahrefs, etc.
        """
        try:
            domain = urlparse(url).netloc
            
            # This is a simplified implementation
            # Real implementation would use APIs from Moz, Ahrefs, Majestic, etc.
            
            # Check for authority indicators on the page
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for trust indicators
                about_page_link = soup.find('a', href=lambda h: h and 'about' in h.lower())
                has_about_page = about_page_link is not None
                
                privacy_policy_link = soup.find('a', href=lambda h: h and 'privacy' in h.lower())
                has_privacy_policy = privacy_policy_link is not None
                
                terms_link = soup.find('a', href=lambda h: h and ('terms' in h.lower() or 'conditions' in h.lower()))
                has_terms = terms_link is not None
                
                # Check for author information
                author_elements = soup.find_all(class_=lambda c: c and 'author' in str(c).lower())
                has_author_info = len(author_elements) > 0
                
                # Check for publication dates
                date_elements = soup.find_all(class_=lambda c: c and ('date' in str(c).lower() or 'published' in str(c).lower()))
                has_dates = len(date_elements) > 0
                
                # Check for citations
                citation_elements = soup.find_all(['cite', 'blockquote'])
                has_citations = len(citation_elements) > 0
                
                # Check for quality outbound links
                quality_domains = ['edu', 'gov', 'org']
                quality_outbound_links = []
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith(('http://', 'https://')) and domain not in href:
                        link_domain = urlparse(href).netloc
                        if any(link_domain.endswith('.' + domain_type) for domain_type in quality_domains):
                            quality_outbound_links.append(href)
                
                return {
                    'authority_indicators': {
                        'has_about_page': has_about_page,
                        'has_privacy_policy': has_privacy_policy,
                        'has_terms': has_terms,
                        'has_author_info': has_author_info,
                        'has_publication_dates': has_dates,
                        'has_citations': has_citations,
                        'quality_outbound_links': len(quality_outbound_links)
                    },
                    'estimated_metrics': {
                        'note': 'These metrics would typically come from third-party tools.',
                        'domain_authority': 'Not available in this implementation',
                        'page_authority': 'Not available in this implementation',
                        'trust_flow': 'Not available in this implementation',
                        'citation_flow': 'Not available in this implementation'
                    },
                    'authority_score': self._calculate_authority_score(has_about_page, has_privacy_policy, 
                                                                has_terms, has_author_info, has_dates, 
                                                                has_citations, len(quality_outbound_links))
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing authority: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_authority_score(self, has_about, has_privacy, has_terms, has_author, has_dates, has_citations, quality_links_count) -> float:
        """Calculate a score for website authority indicators"""
        score = 0
        
        # Basic trust pages
        if has_about:
            score += 1
        if has_privacy:
            score += 1
        if has_terms:
            score += 1
        
        # Content quality indicators
        if has_author:
            score += 1.5
        if has_dates:
            score += 1
        if has_citations:
            score += 1.5
        
        # Quality outbound links
        if quality_links_count > 0:
            score += min(quality_links_count, 3)  # Up to 3 points
        
        return max(0, min(10, score))
    
    async def _analyse_trust_signals(self, url: str) -> Dict:
        """Analyze trust signals on the website"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check for SSL (HTTPS)
                is_https = urlparse(url).scheme == 'https'
                
                # Check for trust badges
                badge_keywords = ['secure', 'trust', 'verified', 'guarantee', 'certification', 'accredited', 'seal']
                trust_badges = []
                
                for img in soup.find_all('img'):
                    alt_text = img.get('alt', '').lower()
                    src = img.get('src', '').lower()
                    if any(keyword in alt_text or keyword in src for keyword in badge_keywords):
                        trust_badges.append({
                            'alt': img.get('alt', ''),
        print(f"[SEO AGENT] ERROR: An unexpected error occurred: {e}")
        print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))

async def run_analysis(api_key, seed_url, limit):
    async with ComprehensiveSEOAgent(api_key) as agent:
        print("[SEO AGENT] Agent initialized. Beginning website analysis...")
        analysis = await agent.analyse_website(seed_url, limit)
        print("[SEO AGENT] Website analysis complete. Generating recommendations...")
        recommendations = await agent.generate_recommendations(analysis)
        print("[SEO AGENT] Analysis complete!")
        print("\nTop 3 Recommendations:")
        for rec in recommendations[:3]:
            print(f"- {rec.get('recommendation', 'N/A')} (Priority: {rec.get('priority_score', 'N/A')})")

if __name__ == "__main__":
    asyncio.run(main())                            'src': img.get('src', '')
                        })
