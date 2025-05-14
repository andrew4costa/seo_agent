import asyncio
import sys
from seo_analyzer import analyze_url
import json

async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analyzer.py <url> [max_pages]")
        return
    
    url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Starting SEO analysis of {url} (analyzing up to {max_pages} pages)...")
    
    try:
        results = await analyze_url(url, max_pages=max_pages)
        
        # Print basic results
        technical_score = results.get("technical_seo", {}).get("score", 0)
        onpage_score = results.get("on_page_seo", {}).get("score", 0)
        pages_analyzed = len(results.get("analyzed_pages", {}))
        
        print(f"\nAnalysis complete for {url}:")
        print(f"- Technical SEO Score: {technical_score:.1f}/10")
        print(f"- On-Page SEO Score: {onpage_score:.1f}/10")
        print(f"- Pages analyzed: {pages_analyzed}")
        
        # Save results to file
        output_file = f"{url.replace('https://', '').replace('http://', '').split('/')[0]}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nComplete analysis saved to {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 