import asyncio
import json
import sys
import logging
from seo_analyzer import analyze_url

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_simple_enhanced")

async def test_simple_enhanced(url: str, max_pages: int = 2):
    """Test the SEO analyzer with enhanced content analysis on a given URL"""
    logger.info(f"Starting simple enhanced SEO analysis for {url}")
    logger.info(f"Analyzing up to {max_pages} pages")
    
    try:
        # Run the analyzer
        results = await analyze_url(url, max_pages=max_pages)
        
        # Print summary results
        technical_score = results.get("technical_seo", {}).get("score", 0)
        onpage_score = results.get("on_page_seo", {}).get("score", 0)
        pages_analyzed = len(results.get("analyzed_pages", {}))
        
        logger.info(f"Analysis complete for {url}:")
        logger.info(f"- Technical SEO Score: {technical_score:.1f}/10")
        logger.info(f"- On-Page SEO Score: {onpage_score:.1f}/10")
        logger.info(f"- Pages analyzed: {pages_analyzed}")
        
        # Save results to a JSON file for inspection
        output_file = f"{url.replace('https://', '').replace('http://', '').split('/')[0]}_simple_enhanced.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nComplete analysis saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during SEO analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        asyncio.run(test_simple_enhanced(url, max_pages))
    else:
        logger.error("Please provide a URL to analyze. Usage: python test_simple_enhanced.py <url> [max_pages]") 