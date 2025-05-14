import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("super_simple")

async def analyze_url(url: str):
    """Super simple SEO analyzer to debug recursion issues"""
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    logger.info(f"Analyzing URL: {url}")
    results = {
        "url": url,
        "title": None,
        "meta_description": None,
        "word_count": 0
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_tag = soup.find('title')
                    results["title"] = title_tag.get_text() if title_tag else None
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    results["meta_description"] = meta_desc.get('content') if meta_desc else None
                    
                    # Calculate word count
                    for script in soup(["script", "style"]):
                        script.extract()
                    text = soup.get_text(separator=' ', strip=True)
                    results["word_count"] = len(text.split())
                    
                    logger.info(f"Analysis completed successfully for {url}")
                else:
                    logger.error(f"Failed to fetch {url}: HTTP status {response.status}")
                    results["error"] = f"HTTP status {response.status}"
    except Exception as e:
        logger.error(f"Error analyzing {url}: {e}")
        results["error"] = str(e)
    
    return results

async def main():
    if len(sys.argv) < 2:
        logger.error("Please provide a URL to analyze")
        return
    
    url = sys.argv[1]
    results = await analyze_url(url)
    
    # Print results
    print("\nResults:")
    print(f"URL: {results['url']}")
    print(f"Title: {results['title']}")
    print(f"Meta description: {results['meta_description']}")
    print(f"Word count: {results['word_count']}")
    
    if "error" in results:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 