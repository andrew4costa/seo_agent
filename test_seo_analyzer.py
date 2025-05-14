import asyncio
import json
import sys
import logging
from seo_analyzer import analyze_url

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_seo_analyzer")

async def test_analyzer(url: str, max_pages: int = 3):
    """Test the SEO analyzer on a given URL"""
    logger.info(f"Starting SEO analysis test for {url}")
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
        
        # Technical SEO details
        tech_seo = results.get("technical_seo", {})
        logger.info("\nTechnical SEO Details:")
        logger.info(f"- SSL/HTTPS implemented: {tech_seo.get('security', {}).get('has_ssl', False)}")
        logger.info(f"- Avg page load time: {tech_seo.get('site_speed', {}).get('avg_load_time', 0):.2f} seconds")
        logger.info(f"- Mobile-friendly score: {tech_seo.get('mobile_optimisation', {}).get('score', 0):.1f}/10")
        logger.info(f"- Robots.txt found: {tech_seo.get('indexation', {}).get('robots_txt', {}).get('exists', False)}")
        logger.info(f"- Sitemap.xml found: {tech_seo.get('indexation', {}).get('sitemap', {}).get('exists', False)}")
        
        # On-page SEO details
        onpage_seo = results.get("on_page_seo", {})
        logger.info("\nOn-Page SEO Details:")
        logger.info(f"- Avg word count: {onpage_seo.get('content_quality', {}).get('avg_word_count', 0):.0f} words")
        logger.info(f"- Pages with thin content: {onpage_seo.get('content_quality', {}).get('thin_content_pages', 0)}")
        logger.info(f"- Pages missing title: {onpage_seo.get('meta_tags', {}).get('pages_without_title', 0)}")
        logger.info(f"- Pages missing meta desc: {onpage_seo.get('meta_tags', {}).get('pages_without_description', 0)}")
        logger.info(f"- Pages missing H1: {onpage_seo.get('heading_structure', {}).get('pages_without_h1', 0)}")
        
        # Schema markup types found
        schema_types = tech_seo.get('structured_data', {}).get('schema_types', [])
        if schema_types:
            logger.info(f"\nSchema Markup Types Found: {', '.join(schema_types)}")
        else:
            logger.info("\nNo schema markup found on analyzed pages")
        
        # Page-specific issues (first 3 pages only)
        logger.info("\nPage-Specific Issues (up to 3 pages):")
        for i, (url, page_data) in enumerate(results.get("analyzed_pages", {}).items()):
            if i >= 3:
                break
                
            logger.info(f"\nPage: {url}")
            logger.info(f"- Title: {page_data.get('title', 'No title')}")
            logger.info(f"- Word Count: {page_data.get('word_count', 0)}")
            logger.info(f"- Load Time: {page_data.get('load_time', 0):.2f} seconds")
            
            if page_data.get("issues"):
                logger.info("  Issues:")
                for issue in page_data.get("issues", []):
                    logger.info(f"  - {issue.get('description')} ({issue.get('severity')} severity)")
            else:
                logger.info("  No issues detected")
        
        # Save results to a JSON file for inspection
        output_file = f"{url.replace('https://', '').replace('http://', '').split('/')[0]}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nComplete analysis saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during SEO analysis: {e}")
        return None

if __name__ == "__main__":
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        asyncio.run(test_analyzer(url, max_pages))
    else:
        logger.error("Please provide a URL to analyze. Usage: python test_seo_analyzer.py <url> [max_pages]")