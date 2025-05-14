#!/usr/bin/env python3
"""
Simplified command-line interface for the SEO analyzer
"""
import asyncio
import json
import sys
import logging
import time
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from seo_analyzer import analyze_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

async def main():
    """Run the SEO analyzer."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_site.py [url] [max_pages] [--competitors competitor1.com competitor2.com ...]")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Parse max_pages if provided
    max_pages = 3  # Default
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        max_pages = int(sys.argv[2])
    
    # Parse competitors if provided
    competitor_urls = []
    if '--competitors' in sys.argv:
        competitor_idx = sys.argv.index('--competitors')
        competitor_urls = sys.argv[competitor_idx+1:]
    
    # Extract domain for the filename
    parsed_url = urlparse(url)
    domain = parsed_url.netloc or url.split('/')[0]
    
    # Log the extracted domain
    logger.info(f"Extracted domain: {domain}")
    
    # Run analysis
    start_time = time.time()
    logger.info(f"Starting SEO analysis of {url} (analyzing up to {max_pages} pages)")
    
    results = analyze_url(url, max_pages)
    
    # Print the time taken
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Output key metrics from the analysis
    logger.info(f"Technical SEO score: {results.get('technical_seo_score', 'N/A')}/10")
    logger.info(f"On-page SEO score: {results.get('on_page_seo_score', 'N/A')}/10")
    
    # Print findings
    print("\n=== SEO Analysis Results ===")
    print(f"URL: {url}")
    print(f"Pages analyzed: {results.get('pages_analyzed', 0)}")
    
    print("\n--- Technical SEO ---")
    tech_seo = results.get("technical_seo_findings", {})
    print(f"SSL/HTTPS implemented: {tech_seo.get('has_ssl', False)}")
    print(f"Average page load time: {tech_seo.get('avg_page_load_time', 0):.2f} seconds")
    print(f"Mobile-friendly score: {tech_seo.get('mobile_friendly_score', 0):.1f}/10")
    print(f"Robots.txt found: {tech_seo.get('has_robots_txt', False)}")
    print(f"Sitemap.xml found: {tech_seo.get('has_sitemap_xml', False)}")
    
    print("\n--- Performance Metrics ---")
    perf = results.get("performance_metrics", {})
    print(f"Total load time: {perf.get('total_load_time', 0):.2f} seconds")
    print(f"Total page size: {perf.get('total_page_size', 0):.2f} MB")
    print(f"Total requests: {perf.get('total_requests', 0)}")
    
    res = perf.get("resource_counts", {})
    print(f"Resource counts: {res.get('images', 0)} images, {res.get('scripts', 0)} scripts, "
          f"{res.get('stylesheets', 0)} stylesheets, {res.get('fonts', 0)} fonts")
    
    print("\n--- On-page SEO ---")
    onpage = results.get("on_page_seo_findings", {})
    print(f"Average word count: {onpage.get('avg_word_count', 0)} words")
    print(f"Pages with thin content: {onpage.get('thin_content_pages', 0)}")
    print(f"Pages missing titles: {onpage.get('missing_title_pages', 0)}")
    print(f"Pages missing meta descriptions: {onpage.get('missing_meta_desc_pages', 0)}")
    print(f"Pages missing H1 tags: {onpage.get('missing_h1_pages', 0)}")
    
    # If competitor analysis was performed
    if "competitor_analysis" in results:
        print("\n--- Competitor Analysis ---")
        for comp_url, comp_data in results["competitor_analysis"].items():
            print(f"\nCompetitor: {comp_url}")
            if isinstance(comp_data, dict):
                print(f"Load time: {comp_data.get('load_time_seconds', 0):.2f} seconds")
                print(f"Word count: {comp_data.get('word_count', 0)} words")
                print(f"Has schema markup: {comp_data.get('has_schema_markup', False)}")
                print(f"Meta tags present: {comp_data.get('has_meta_tags', False)}")
                print(f"Internal links: {comp_data.get('internal_link_count', 0)}")
            else:
                print(f"Error: {comp_data}")
    
    # Save to file with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{REPORTS_DIR}/{domain}_seo_analysis_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Complete analysis saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main()) 