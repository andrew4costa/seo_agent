import asyncio
import argparse
import os
import sys
import logging
import json
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seo_agent")

class SEOAgent:
    def __init__(self, api_key: str = "test_key"):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.current_url = None
    
    async def analyze_website(self, url: str, limit: int = 1):
        """Simple website analysis"""
        try:
            self.logger.info(f"Analyzing website: {url}")
            self.current_url = url
            
            # Customize analysis based on the URL
            technical_issues = ["Site speed could be improved"]
            technical_score = 7.5
            
            # Add SSL issue for non-https URLs
            if not url.startswith("https://"):
                technical_issues.append("Missing SSL certificate")
                technical_score -= 1.5
            
            # Customize on-page issues based on domain
            on_page_issues = []
            on_page_score = 6.8
            
            if "example.com" in url:
                on_page_issues = ["Missing meta descriptions", "Thin content on some pages"]
            elif "github" in url:
                on_page_issues = ["Inconsistent heading structure", "Thin content on repository pages"]
                on_page_score = 6.2
            elif "wordpress" in url:
                on_page_issues = ["Duplicate content on tag/category pages", "Image optimization needed"]
                on_page_score = 5.8
            elif "shopify" in url or "ecommerce" in url:
                on_page_issues = ["Product descriptions need improvement", "Missing structured data"]
                on_page_score = 6.0
            else:
                on_page_issues = ["Content could be more comprehensive", "Heading structure needs improvement"]
            
            # Customize off-page issues
            off_page_issues = ["Low backlink count"]
            off_page_score = 5.5
            
            if "blog" in url or "news" in url:
                off_page_issues.append("Few social shares")
            else:
                off_page_issues.append("Few social mentions")
            
            # Simulate analysis for demonstration
            analysis = {
                "technical_seo": {
                    "score": technical_score,
                    "issues": technical_issues
                },
                "on_page_seo": {
                    "score": on_page_score,
                    "issues": on_page_issues
                },
                "off_page_seo": {
                    "score": off_page_score,
                    "issues": off_page_issues
                }
            }
            
            self.logger.info("Analysis completed successfully")
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing website: {e}")
            raise
            
    async def generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate SEO recommendations based on analysis"""
        try:
            self.logger.info("Generating recommendations")
            
            # Base recommendations
            base_recommendations = []
            
            # Technical SEO recommendations based on issues
            technical_issues = analysis.get("technical_seo", {}).get("issues", [])
            technical_score = analysis.get("technical_seo", {}).get("score", 5.0)
            
            if "Site speed" in str(technical_issues) or technical_score < 7.0:
                base_recommendations.append({
                    "recommendation": "Improve site speed by optimizing images and implementing caching",
                    "priority_score": 8,
                    "impact": "high",
                    "implementation_difficulty": "medium",
                    "estimated_time": 4,
                    "category": "Technical SEO"
                })
            
            if "SSL certificate" in str(technical_issues):
                base_recommendations.append({
                    "recommendation": "Implement an SSL certificate for improved security and SEO ranking",
                    "priority_score": 9,
                    "impact": "high",
                    "implementation_difficulty": "easy",
                    "estimated_time": 1,
                    "category": "Technical SEO"
                })
            
            # On-page SEO recommendations
            on_page_issues = analysis.get("on_page_seo", {}).get("issues", [])
            on_page_score = analysis.get("on_page_seo", {}).get("score", 5.0)
            
            if "meta description" in str(on_page_issues).lower():
                base_recommendations.append({
                    "recommendation": "Add compelling meta descriptions to all pages",
                    "priority_score": 7,
                    "impact": "medium",
                    "implementation_difficulty": "easy",
                    "estimated_time": 2,
                    "category": "On-Page SEO"
                })
            
            if "thin content" in str(on_page_issues).lower() or on_page_score < 7.0:
                base_recommendations.append({
                    "recommendation": "Create a content strategy to address thin content issues",
                    "priority_score": 7,
                    "impact": "high",
                    "implementation_difficulty": "hard",
                    "estimated_time": 8,
                    "category": "On-Page SEO"
                })
            
            # Off-page SEO recommendations
            off_page_issues = analysis.get("off_page_seo", {}).get("issues", [])
            off_page_score = analysis.get("off_page_seo", {}).get("score", 5.0)
            
            if "backlink" in str(off_page_issues).lower() or off_page_score < 6.0:
                base_recommendations.append({
                    "recommendation": "Implement a link building campaign to improve backlink profile",
                    "priority_score": 8,
                    "impact": "high",
                    "implementation_difficulty": "hard",
                    "estimated_time": 10,
                    "category": "Off-Page SEO"
                })
            
            if "social" in str(off_page_issues).lower():
                base_recommendations.append({
                    "recommendation": "Develop a social media strategy to increase brand presence",
                    "priority_score": 6,
                    "impact": "medium",
                    "implementation_difficulty": "medium",
                    "estimated_time": 5,
                    "category": "Off-Page SEO"
                })
            
            # URL-specific recommendations (simulating different analysis for different URLs)
            if "example.com" in self.current_url:
                base_recommendations.append({
                    "recommendation": "Create more specific example content for demonstration purposes",
                    "priority_score": 5,
                    "impact": "medium",
                    "implementation_difficulty": "medium",
                    "estimated_time": 3,
                    "category": "Content"
                })
            elif "github.com" in self.current_url:
                base_recommendations.append({
                    "recommendation": "Optimize GitHub repository descriptions and READMEs for better discoverability",
                    "priority_score": 7,
                    "impact": "medium",
                    "implementation_difficulty": "easy",
                    "estimated_time": 2,
                    "category": "Technical SEO"
                })
            elif "wordpress" in self.current_url:
                base_recommendations.append({
                    "recommendation": "Install and configure Yoast SEO plugin for WordPress",
                    "priority_score": 8,
                    "impact": "high",
                    "implementation_difficulty": "easy",
                    "estimated_time": 1,
                    "category": "Technical SEO"
                })
            elif "shopify" in self.current_url or "ecommerce" in self.current_url:
                base_recommendations.append({
                    "recommendation": "Implement structured data markup for product pages",
                    "priority_score": 7,
                    "impact": "medium",
                    "implementation_difficulty": "medium",
                    "estimated_time": 4,
                    "category": "Technical SEO"
                })
                base_recommendations.append({
                    "recommendation": "Optimize product images with descriptive alt text and compressed file sizes",
                    "priority_score": 6,
                    "impact": "medium",
                    "implementation_difficulty": "easy",
                    "estimated_time": 3,
                    "category": "On-Page SEO"
                })
            else:
                # Generic recommendations for any site
                base_recommendations.append({
                    "recommendation": "Perform comprehensive keyword research to identify new content opportunities",
                    "priority_score": 8,
                    "impact": "high",
                    "implementation_difficulty": "medium",
                    "estimated_time": 6,
                    "category": "Content Strategy"
                })
                base_recommendations.append({
                    "recommendation": "Implement a regular content calendar to improve freshness signals",
                    "priority_score": 7,
                    "impact": "medium",
                    "implementation_difficulty": "medium",
                    "estimated_time": 4,
                    "category": "Content Strategy"
                })
            
            # Ensure we have at least 5 recommendations
            if len(base_recommendations) < 5:
                default_recs = [
                    {
                        "recommendation": "Optimize site navigation and internal linking structure",
                        "priority_score": 7,
                        "impact": "medium",
                        "implementation_difficulty": "medium",
                        "estimated_time": 5,
                        "category": "Technical SEO"
                    },
                    {
                        "recommendation": "Improve mobile responsiveness across all pages",
                        "priority_score": 8,
                        "impact": "high",
                        "implementation_difficulty": "medium",
                        "estimated_time": 6,
                        "category": "Technical SEO"
                    },
                    {
                        "recommendation": "Analyze and optimize page load times",
                        "priority_score": 8,
                        "impact": "high",
                        "implementation_difficulty": "medium",
                        "estimated_time": 5,
                        "category": "Technical SEO"
                    }
                ]
                
                for rec in default_recs:
                    if len(base_recommendations) < 5:
                        base_recommendations.append(rec)
            
            # Sort by priority score (highest first)
            recommendations = sorted(base_recommendations, key=lambda x: x.get("priority_score", 0), reverse=True)
            
            self.logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            raise

def save_report(url: str, recommendations: List[Dict], analysis: Dict, output_dir: Optional[str] = None):
    """Save results to a JSON file"""
    try:
        # Create a filename based on the domain and date
        from datetime import datetime
        import re
        
        # Extract domain from URL
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        domain = re.sub(r'[^\w\-_]', '_', domain)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Determine output directory
        if not output_dir:
            output_dir = os.path.expanduser("~/seo_reports")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = f"{domain}_seo_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Create report data
        report = {
            "url": url,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis": analysis,
            "recommendations": recommendations
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Report saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return None

async def run_analysis(api_key, url, limit, save_report_flag=False, output_dir=None):
    """Run the SEO analysis and generate recommendations"""
    try:
        agent = SEOAgent(api_key)
        
        # Analyze website
        analysis = await agent.analyze_website(url, limit)
        
        # Generate recommendations
        recommendations = await agent.generate_recommendations(analysis)
        
        # Save report if requested
        if save_report_flag:
            report_path = save_report(url, recommendations, analysis, output_dir)
            print(f"Report saved to: {report_path}")
        
        # Display top recommendations
        print("\nTop Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            priority = rec.get('priority_score', 'N/A')
            impact = rec.get('impact', 'N/A')
            difficulty = rec.get('implementation_difficulty', 'N/A')
            time = rec.get('estimated_time', 'N/A')
            
            print(f"{i}. {rec.get('recommendation', 'N/A')}")
            print(f"   Priority: {priority} | Impact: {impact} | Difficulty: {difficulty} | Est. Time: {time} hours")
            print("")
        
        return recommendations, analysis
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise

async def main():
    """Main entry point for the SEO Analysis tool"""
    parser = argparse.ArgumentParser(description='Simple SEO Analysis Tool')
    parser.add_argument('url', help='URL to analyze')
    parser.add_argument('-l', '--limit', type=int, default=1, help='Maximum number of pages to crawl (default: 1)')
    parser.add_argument('--save-report', action='store_true', help='Save report to file')
    parser.add_argument('--output-dir', help='Directory to save report (default: ~/seo_reports)')
    parser.add_argument('--key', default='test_key', help='API key (default: test_key)')
    
    args = parser.parse_args()
    
    # Normalize URL (add https:// if not present)
    if not args.url.startswith(('http://', 'https://')):
        url = f"https://{args.url}"
    else:
        url = args.url
    
    print(f"Analyzing {url}")
    
    try:
        await run_analysis(args.key, url, args.limit, args.save_report, args.output_dir)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 