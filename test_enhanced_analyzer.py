import asyncio
import json
import sys
import logging
from seo_analyzer import analyze_url

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_enhanced_analyzer")

async def test_enhanced_analyzer(url: str, max_pages: int = 3, competitors: list = None):
    """Test the enhanced SEO analyzer on a given URL"""
    logger.info(f"Starting enhanced SEO analysis for {url}")
    logger.info(f"Analyzing up to {max_pages} pages")
    
    if competitors:
        logger.info(f"Including competitor analysis for: {', '.join(competitors)}")
    
    try:
        # Run the analyzer with competitor analysis if competitors are provided
        results = await analyze_url(
            url, 
            max_pages=max_pages,
            analyze_competitors=bool(competitors),
            competitors=competitors or []
        )
        
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
        
        # Performance details
        performance = tech_seo.get("performance", {})
        logger.info("\nPerformance Details:")
        logger.info(f"- Total load time: {performance.get('total_load_time_seconds', 0):.2f} seconds")
        logger.info(f"- Total page size: {performance.get('total_page_size_kb', 0)/1024:.2f} MB")
        logger.info(f"- Total requests: {performance.get('total_requests', 0)}")
        
        resource_counts = performance.get("resource_counts", {})
        logger.info(f"- Resource counts:")
        logger.info(f"  - Images: {resource_counts.get('images', 0)}")
        logger.info(f"  - Scripts: {resource_counts.get('scripts', 0)}")
        logger.info(f"  - Stylesheets: {resource_counts.get('stylesheets', 0)}")
        logger.info(f"  - Fonts: {resource_counts.get('fonts', 0)}")
        
        # Performance issues
        perf_issues = performance.get("performance_issues", [])
        if perf_issues:
            logger.info("\nPerformance Issues:")
            for issue in perf_issues:
                logger.info(f"- {issue.get('description')} ({issue.get('severity')} severity)")
        
        # Content analysis details
        content_analysis = results.get("content_analysis", {})
        if content_analysis:
            logger.info("\nContent Analysis:")
            logger.info(f"- Avg readability score: {content_analysis.get('avg_readability_score', 0):.2f}/100")
            
            # Keyword consistency
            keyword_consistency = content_analysis.get("keyword_consistency", {})
            top_keywords = keyword_consistency.get("top_site_keywords", [])
            if top_keywords:
                logger.info("\nTop Keywords Across Site:")
                for keyword, count in top_keywords[:5]:  # Show top 5
                    logger.info(f"- '{keyword}' (appears {count} times)")
        
        # On-page SEO details
        onpage_seo = results.get("on_page_seo", {})
        logger.info("\nOn-Page SEO Details:")
        logger.info(f"- Avg word count: {onpage_seo.get('content_quality', {}).get('avg_word_count', 0):.0f} words")
        logger.info(f"- Pages with thin content: {onpage_seo.get('content_quality', {}).get('thin_content_pages', 0)}")
        logger.info(f"- Pages missing title: {onpage_seo.get('meta_tags', {}).get('pages_without_title', 0)}")
        logger.info(f"- Pages missing meta desc: {onpage_seo.get('meta_tags', {}).get('pages_without_description', 0)}")
        logger.info(f"- Pages missing H1: {onpage_seo.get('heading_structure', {}).get('pages_without_h1', 0)}")
        
        # Competitor analysis
        competitor_analysis = results.get("competitor_analysis", {})
        if competitor_analysis:
            competitors_analyzed = competitor_analysis.get("competitors_analyzed", 0)
            competitor_data = competitor_analysis.get("competitor_data", {})
            
            if competitors_analyzed > 0:
                logger.info(f"\nCompetitor Analysis ({competitors_analyzed} competitors):")
                
                for competitor, data in competitor_data.items():
                    if "error" in data:
                        logger.info(f"- {competitor}: Error analyzing competitor - {data['error']}")
                        continue
                    
                    logger.info(f"\nCompetitor: {competitor}")
                    logger.info(f"- Load time: {data.get('load_time_seconds', 0):.2f} seconds")
                    logger.info(f"- Word count: {data.get('word_count', 0)} words")
                    logger.info(f"- Has schema markup: {data.get('has_schema', False)}")
                    logger.info(f"- Has meta tags: {data.get('meta_tags_present', False)}")
                    logger.info(f"- Internal links: {data.get('internal_links_count', 0)}")
        
        # Page-specific issues (first 2 pages only)
        logger.info("\nPage-Specific Details (up to 2 pages):")
        for i, (url, page_data) in enumerate(results.get("analyzed_pages", {}).items()):
            if i >= 2:
                break
                
            logger.info(f"\nPage: {url}")
            logger.info(f"- Title: {page_data.get('title', 'No title')}")
            logger.info(f"- Word Count: {page_data.get('word_count', 0)}")
            logger.info(f"- Load Time: {page_data.get('load_time', 0):.2f} seconds")
            
            # Content analysis for this page
            content_analysis = page_data.get("content_analysis", {})
            if content_analysis:
                # Readability
                readability = content_analysis.get("readability", {})
                if readability:
                    logger.info(f"- Readability score: {readability.get('flesch_reading_ease', 0):.2f}/100")
                    logger.info(f"- Grade level: {readability.get('grade_level', 'N/A')}")
                
                # Keywords
                keyword_analysis = content_analysis.get("keyword_analysis", {})
                top_keywords = keyword_analysis.get("top_keywords", [])
                if top_keywords:
                    logger.info("- Top keywords:")
                    for keyword, count in top_keywords[:3]:  # Show top 3
                        logger.info(f"  - '{keyword}' (appears {count} times)")
            
            # Issues
            if page_data.get("issues"):
                logger.info("- Issues:")
                for issue in page_data.get("issues", []):
                    logger.info(f"  - {issue.get('description')} ({issue.get('severity')} severity)")
            else:
                logger.info("- No issues detected")
        
        # Save results to a JSON file for inspection
        output_file = f"{url.replace('https://', '').replace('http://', '').split('/')[0]}_enhanced_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nComplete enhanced analysis saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during enhanced SEO analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        
        # Check for competitor URLs
        competitors = []
        if len(sys.argv) > 3:
            competitors = sys.argv[3].split(',')
        
        asyncio.run(test_enhanced_analyzer(url, max_pages, competitors))
    else:
        logger.error("Please provide a URL to analyze. Usage: python test_enhanced_analyzer.py <url> [max_pages] [competitor1,competitor2]") 