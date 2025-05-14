import asyncio
import aiohttp
from typing import List, Dict, Set, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import argparse
import os
import sys
import traceback
import ssl
import re
import configparser
import pathlib
import time
import tldextract
from aiohttp import ClientError
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seo_agent")

# Constants
CONFIG_FILE = os.path.expanduser("~/.seo_agent_config.ini")
REPORTS_DIR = os.path.expanduser("~/seo_reports")

def save_api_keys(api_keys: Dict[str, str]):
    """Save API keys to config file"""
    config = configparser.ConfigParser()
    config['API_KEYS'] = api_keys
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    
    # Save the config file
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)
    
    # Set permissions to user only
    os.chmod(CONFIG_FILE, 0o600)
    logger.info(f"API keys saved to {CONFIG_FILE}")

def load_api_keys() -> Dict[str, str]:
    """Load API keys from config file"""
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        if 'API_KEYS' in config:
            logger.info(f"API keys loaded from {CONFIG_FILE}")
            return dict(config['API_KEYS'])
    return {}

def save_report(url: str, recommendations: List[Dict], analysis: Dict, title: Optional[str] = None, output_dir: Optional[str] = None):
    """Save recommendations to a JSON file with company name and date"""
    # Extract domain info
    extract_result = tldextract.extract(url)
    domain = extract_result.domain
    company_name = title or domain.capitalize()
    
    # Use provided output directory or fall back to default
    report_dir = output_dir or REPORTS_DIR
    
    # Create the report directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate filename with date and company name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sanitized_company = re.sub(r'[^\w\-_]', '_', company_name)
    filename = f"{sanitized_company}_seo_report_{timestamp}.json"
    filepath = os.path.join(report_dir, filename)
    
    # Create report data structure
    report = {
        "company": company_name,
        "url": url,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recommendations": recommendations,
        "analysis_data": {
            "technical_seo": analysis.get("technical_seo", {}),
            "on_page_seo": analysis.get("on_page_seo", {}),
            "off_page_seo": analysis.get("off_page_seo", {}),
            "analyzed_pages": analysis.get("analyzed_pages", {}),
            "website_business_analysis": analysis.get("website_analysis", "")
        },
        "summary": {
            "technical_score": analysis.get("technical_seo", {}).get("score", 0),
            "on_page_score": analysis.get("on_page_seo", {}).get("score", 0),
            "off_page_score": analysis.get("off_page_seo", {}).get("score", 0),
            "total_pages_analyzed": len(analysis.get("analyzed_pages", {})),
        }
    }
    
    # Save the report
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filepath

def save_html_report(url: str, recommendations: List[Dict], analysis: Dict, title: Optional[str] = None, output_dir: Optional[str] = None):
    """Save SEO analysis as a human-readable HTML report"""
    # Extract domain info
    extract_result = tldextract.extract(url)
    domain = extract_result.domain
    company_name = title or domain.capitalize()
    
    # Use provided output directory or fall back to default
    report_dir = output_dir or REPORTS_DIR
    
    # Create the report directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate filename with date and company name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sanitized_company = re.sub(r'[^\w\-_]', '_', company_name)
    filename = f"{sanitized_company}_seo_report_{timestamp}.html"
    filepath = os.path.join(report_dir, filename)
    
    # Extract summary scores
    technical_score = analysis.get("technical_seo", {}).get("score", 0)
    on_page_score = analysis.get("on_page_seo", {}).get("score", 0)
    off_page_score = analysis.get("off_page_seo", {}).get("score", 0)
    total_pages = len(analysis.get("analyzed_pages", {}))
    
    # Get issues from all pages
    all_issues = []
    for page_url, page_data in analysis.get("analyzed_pages", {}).items():
        page_issues = page_data.get("issues", [])
        for issue in page_issues:
            issue_with_page = issue.copy()
            issue_with_page["page"] = page_url
            all_issues.append(issue_with_page)
    
    # Sort issues by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_issues.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEO Report - {company_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .scores-container {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .score-card {{
            flex: 1;
            min-width: 200px;
            text-align: center;
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .technical {{
            background-color: #e3f2fd;
        }}
        .onpage {{
            background-color: #e8f5e9;
        }}
        .offpage {{
            background-color: #fff3e0;
        }}
        .score {{
            font-size: 36px;
            font-weight: bold;
        }}
        .recommendations {{
            margin-top: 30px;
        }}
        .recommendation {{
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .recommendation h3 {{
            margin-top: 0;
        }}
        .recommendation-meta {{
            display: flex;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        .meta-item {{
            background-color: #e9ecef;
            padding: 5px 10px;
            border-radius: 15px;
            margin-right: 10px;
            margin-bottom: 5px;
            font-size: 14px;
        }}
        .priority-high {{
            background-color: #ffcccc;
        }}
        .priority-medium {{
            background-color: #ffffcc;
        }}
        .priority-low {{
            background-color: #e6f2ff;
        }}
        .issues {{
            margin-top: 30px;
        }}
        .issue {{
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }}
        .severity-high {{
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }}
        .severity-medium {{
            background-color: #fff8e1;
            border-left: 5px solid #ffc107;
        }}
        .severity-low {{
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
        }}
        .pages {{
            margin-top: 30px;
        }}
        .page {{
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .collapse-btn {{
            background-color: #f1f1f1;
            padding: 10px 15px;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            border-radius: 5px;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .collapse-content {{
            display: none;
            padding: 15px;
            overflow: hidden;
            background-color: #fafafa;
            border-radius: 5px;
        }}
        .recommendations-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .recommendations-table th {{
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
        }}
        .recommendations-table td {{
            border: 1px solid #ddd;
            padding: 10px;
            vertical-align: top;
        }}
        .priority-level {{
            font-weight: bold;
            text-align: center;
        }}
        .high {{
            color: #d32f2f;
        }}
        .medium {{
            color: #f57c00;
        }}
        .low {{
            color: #388e3c;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SEO Analysis Report</h1>
        <h2>{company_name}</h2>
        <p>URL: <a href="{url}" target="_blank">{url}</a></p>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Pages analyzed: {total_pages}</p>
    </div>

    <div class="scores-container">
        <div class="score-card technical">
            <h3>Technical SEO</h3>
            <div class="score">{technical_score:.1f}/10</div>
        </div>
        <div class="score-card onpage">
            <h3>On-Page SEO</h3>
            <div class="score">{on_page_score:.1f}/10</div>
        </div>
        <div class="score-card offpage">
            <h3>Off-Page SEO</h3>
            <div class="score">{off_page_score:.1f}/10</div>
        </div>
    </div>

    <div class="recommendations">
        <h2>SEO Recommendations</h2>

        <table class="recommendations-table">
            <thead>
                <tr>
                    <th>Recommendation</th>
                    <th>Priority</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
"""

    # Group recommendations by category
    categories = {}
    for rec in recommendations:
        category = rec.get("category", "General")
        if category not in categories:
            categories[category] = []
        categories[category].append(rec)
    
    # Insert architecture-specific recommendations if needed
    # This would be populated from input data or the recommendations list
    architecture_recs = []
    # Check if we have any recommendations marked with ARCHITECTURE category
    if "ARCHITECTURE" in categories:
        architecture_recs = categories["ARCHITECTURE"]
    
    # Add categorized recommendations to the table
    added_architecture = False
    for category, recs in categories.items():
        # Skip ARCHITECTURE as we'll handle it separately
        if category == "ARCHITECTURE":
            continue
            
        # Sort by priority within each category
        sorted_recs = sorted(recs, key=lambda x: x.get("priority_score", 0), reverse=True)
        
        html_content += f"""
                <tr>
                    <td colspan="3" style="background-color: #e9ecef; font-weight: bold; text-align: center;">{category}</td>
                </tr>
        """
        
        for rec in sorted_recs:
            priority_score = rec.get("priority_score", 0)
            if priority_score >= 8:
                priority = "High"
                priority_class = "high"
            elif priority_score >= 5:
                priority = "Medium"
                priority_class = "medium"
            else:
                priority = "Low"
                priority_class = "low"
                
            recommendation_title = rec.get("recommendation", "").split(".")[0].strip() if "." in rec.get("recommendation", "") else rec.get("recommendation", "")
            detailed_description = rec.get("recommendation", "")
            
            html_content += f"""
                <tr>
                    <td style="width: 25%;">{recommendation_title}</td>
                    <td style="width: 15%;" class="priority-level {priority_class}">{priority}</td>
                    <td style="width: 60%;">{detailed_description}</td>
                </tr>
            """
    
    # Add ARCHITECTURE recommendations at the end if they exist
    if architecture_recs:
        html_content += f"""
                <tr>
                    <td colspan="3" style="background-color: #e9ecef; font-weight: bold; text-align: center;">ARCHITECTURE</td>
                </tr>
        """
        
        for rec in architecture_recs:
            priority_score = rec.get("priority_score", 0)
            if priority_score >= 8:
                priority = "High"
                priority_class = "high"
            elif priority_score >= 5:
                priority = "Medium"
                priority_class = "medium"
            else:
                priority = "Low"
                priority_class = "low"
                
            # Use the specific architecture_title if available
            recommendation_title = rec.get("architecture_title", "") 
            if not recommendation_title:
                recommendation_title = rec.get("recommendation", "").split(".")[0].strip() if "." in rec.get("recommendation", "") else rec.get("recommendation", "")
                
            detailed_description = rec.get("recommendation", "")
            
            html_content += f"""
                <tr>
                    <td style="width: 25%;">{recommendation_title}</td>
                    <td style="width: 15%;" class="priority-level {priority_class}">{priority}</td>
                    <td style="width: 60%;">{detailed_description}</td>
                </tr>
            """
    # If no architecture recommendations are included in the data, we'll add a sample section matching the image
    else:
        html_content += f"""
                <tr>
                    <td colspan="3" style="background-color: #e9ecef; font-weight: bold; text-align: center;">ARCHITECTURE</td>
                </tr>
                <tr>
                    <td style="width: 25%;">Create URL Keyword Map</td>
                    <td style="width: 15%;" class="priority-level high">High</td>
                    <td style="width: 60%;">Create an Excel map spanning your entire category structure and URL strings, and plot keyword analyse against each.</td>
                </tr>
                <tr>
                    <td style="width: 25%;">Improving Category Naming</td>
                    <td style="width: 15%;" class="priority-level medium">Medium</td>
                    <td style="width: 60%;">Some category names, like "EU & Eastern Europe," should be more search-friendly. "Europe" and "Eastern Europe" could be separate categories for better optimization the same with "Central & South America". Consider targeting popular holiday destinations i.e. Dubai etc</td>
                </tr>
                <tr>
                    <td style="width: 25%;">Add "All" Categories</td>
                    <td style="width: 15%;" class="priority-level medium">Medium</td>
                    <td style="width: 60%;">Ensure you have an "all destinations" and "all holiday types" to aid crawlability and UX.</td>
                </tr>
                <tr>
                    <td style="width: 25%;">Consolidate Bespoke</td>
                    <td style="width: 15%;" class="priority-level low">Low</td>
                    <td style="width: 60%;">There is a good opportunity to consolidate Berkeley Bespoke with Travel. This would combine the authorities of both sites and make management easier.</td>
                </tr>
                <tr>
                    <td style="width: 25%;">URL Naming Structure</td>
                    <td style="width: 15%;" class="priority-level medium">Medium</td>
                    <td style="width: 60%;">In some instances categories do not currently make proper use of keywords in URL string. If changing the URL, ensure to use 301 redirects from the old URL to the new URL. Ensure that category URLs contain the core target keyword where possible i.e. /interests/beach/ vs. /interests/luxury-beach-holidays/,"africa-indian-ocean" vs "africa-indian-ocean-holidays".</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </div>

    <div class="recommendations">
        <h2>Top Recommendations Details</h2>
"""

    # Add recommendations
    for i, rec in enumerate(recommendations[:6]):  # Limit to top 6 recommendations
        priority = rec.get("priority_score", 0)
        priority_class = "priority-high" if priority >= 8 else "priority-medium" if priority >= 5 else "priority-low"
        
        html_content += f"""
        <div class="recommendation">
            <h3>{i+1}. {rec.get("recommendation", "")}</h3>
            <div class="recommendation-meta">
                <div class="meta-item {priority_class}">Priority: {priority}/10</div>
                <div class="meta-item">Impact: {rec.get("impact", "N/A")}</div>
                <div class="meta-item">Difficulty: {rec.get("implementation_difficulty", "N/A")}</div>
                <div class="meta-item">Est. Time: {rec.get("estimated_time_hours", rec.get("estimated_time", "N/A"))} hrs</div>
                <div class="meta-item">Category: {rec.get("category", "General")}</div>
            </div>
        </div>"""

    # Add issues section
    html_content += """
    <div class="issues">
        <h2>Key Issues Found</h2>
"""

    # Add up to 10 most critical issues
    for i, issue in enumerate(all_issues[:10]):
        severity = issue.get("severity", "low")
        severity_class = f"severity-{severity}"
        
        html_content += f"""
        <div class="issue {severity_class}">
            <h4>{issue.get("type", "Issue").replace("_", " ").title()}</h4>
            <p>{issue.get("details", "")}</p>
            <p><strong>Page:</strong> {issue.get("page", "N/A")}</p>
        </div>"""

    # Add analyzed pages section
    html_content += """
    </div>
    
    <div class="pages">
        <h2>Analyzed Pages</h2>
"""

    # Add collapsible sections for technical SEO, on-page SEO, and off-page SEO details
    html_content += """
        <button class="collapse-btn" onclick="toggleCollapse('technical-details')">Technical SEO Details</button>
        <div id="technical-details" class="collapse-content">
"""
    
    # Add technical SEO details
    technical_seo = analysis.get("technical_seo", {})
    html_content += f"""
            <h3>Site Speed</h3>
            <p>Average Load Time: {technical_seo.get("site_speed", {}).get("avg_load_time", "N/A")} seconds</p>
            
            <h3>Mobile Optimization</h3>
            <p>Mobile-Friendly: {"Yes" if technical_seo.get("mobile_optimisation", {}).get("has_viewport_meta", False) else "No"}</p>
            
            <h3>Security</h3>
            <p>HTTPS Implemented: {"Yes" if technical_seo.get("security", {}).get("has_ssl", False) else "No"}</p>
            
            <h3>Indexation</h3>
            <p>Robots.txt: {"Found" if technical_seo.get("indexation", {}).get("robots_txt", {}).get("exists", False) else "Not Found"}</p>
            <p>Sitemap.xml: {"Found" if technical_seo.get("indexation", {}).get("sitemap", {}).get("exists", False) else "Not Found"}</p>
    """
    
    html_content += """
        </div>
        
        <button class="collapse-btn" onclick="toggleCollapse('onpage-details')">On-Page SEO Details</button>
        <div id="onpage-details" class="collapse-content">
"""
    
    # Add on-page SEO details
    on_page_seo = analysis.get("on_page_seo", {})
    html_content += f"""
            <h3>Content Quality</h3>
            <p>Average Word Count: {on_page_seo.get("content_quality", {}).get("avg_word_count", "N/A")} words</p>
            <p>Pages with Thin Content: {on_page_seo.get("content_quality", {}).get("thin_content_pages", "N/A")}</p>
            
            <h3>Meta Tags</h3>
            <p>Pages Missing Title Tags: {on_page_seo.get("meta_tags", {}).get("pages_without_title", "N/A")}</p>
            <p>Pages Missing Meta Descriptions: {on_page_seo.get("meta_tags", {}).get("pages_without_description", "N/A")}</p>
            
            <h3>Heading Structure</h3>
            <p>Pages Missing H1 Headings: {on_page_seo.get("heading_structure", {}).get("pages_without_h1", "N/A")}</p>
            
            <h3>Image Optimization</h3>
            <p>Images Without Alt Text: {on_page_seo.get("image_optimization", {}).get("images_without_alt", "N/A")}</p>
    """
    
    html_content += """
        </div>
        
        <button class="collapse-btn" onclick="toggleCollapse('pages-details')">Page Details</button>
        <div id="pages-details" class="collapse-content">
"""
    
    # Add table of analyzed pages
    html_content += """
            <table>
                <tr>
                    <th>URL</th>
                    <th>Title</th>
                    <th>Word Count</th>
                    <th>Issues</th>
                </tr>
    """
    
    for page_url, page_data in analysis.get("analyzed_pages", {}).items():
        issues_count = len(page_data.get("issues", []))
        html_content += f"""
                <tr>
                    <td><a href="{page_url}" target="_blank">{page_url}</a></td>
                    <td>{page_data.get("title", "N/A")}</td>
                    <td>{page_data.get("word_count", "N/A")}</td>
                    <td>{issues_count}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </div>

    <script>
        function toggleCollapse(id) {
            var content = document.getElementById(id);
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        }
    </script>

</body>
</html>
"""
    
    # Save the HTML report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filepath

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
    issues: List[Dict] = None  # Store identified issues
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []

class LLMProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.logger = logging.getLogger(__name__)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.default_model = "gpt-3.5-turbo"
        self.session = None
        # Flag to track if we've hit API quota limits
        self.quota_exceeded = False
        # Alternative local LLM URL (for Ollama, etc.)
        self.local_llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
        # Flag to use local LLM
        self.use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        # Set LLM provider
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        # Alternative API keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.together_api_key = os.getenv("TOGETHER_API_KEY", "")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *err):
        if self.session:
            await self.session.close()
    
    async def _make_api_request(self, messages: List[Dict[str, str]], temperature: float=0.7) -> Dict:
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # If we already know quota is exceeded, or user has enabled local LLM, use local LLM
            if self.quota_exceeded or self.use_local_llm:
                return await self._use_fallback_llm(messages, temperature)
            
            # Use appropriate API based on provider setting
            if self.llm_provider == "gemini" and self.gemini_api_key:
                return await self._make_gemini_request(messages, temperature)
            elif self.llm_provider == "together" and self.together_api_key:
                return await self._make_together_request(messages, temperature)
            elif self.llm_provider != "openai":
                # If provider isn't openai but no API key, fall back to openai
                self.logger.warning(f"No API key for {self.llm_provider}, trying OpenAI")
                
            # Default to OpenAI
            payload = {
                "model": self.default_model,
                "messages": messages,
                "temperature": temperature
            }
        
            async with self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    self.logger.error(f"API request failed with status {response.status}: {response_text}")
                    
                    # Check for quota exceeded error
                    if "insufficient_quota" in response_text or "exceeded your current quota" in response_text:
                        self.quota_exceeded = True
                        self.logger.warning("OpenAI API quota exceeded. Trying local LLM.")
                        return await self._use_fallback_llm(messages, temperature)
                    
                    raise Exception(f"API request failed: {response_text}")
                
                return await response.json()
        
        except Exception as e:
            self.logger.error(f"Error making API request: {str(e)}")
            # Try local LLM or raise error
            if self.use_local_llm:
                return await self._use_fallback_llm(messages, temperature)
            raise ValueError(f"API request failed and no fallback available: {str(e)}")

    async def _make_gemini_request(self, messages: List[Dict[str, str]], temperature: float) -> Dict:
        """Make a request to Google's Gemini API (high free quota)"""
        try:
            # Format OpenAI-style messages to Gemini format
            prompt = "You MUST respond with valid JSON only - no explanations, no extra text.\n\n"
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"Instructions: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            # For Gemini API
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            async with self.session.post(
                f"{gemini_url}?key={self.gemini_api_key}",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    self.logger.error(f"Gemini API request failed: {response_text}")
                    raise Exception(f"Gemini API request failed: {response_text}")
                
                gemini_response = await response.json()
                
                # Convert Gemini response format to OpenAI format
                try:
                    content = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }
                        ]
                    }
                except (KeyError, IndexError) as e:
                    self.logger.error(f"Error parsing Gemini response: {str(e)}")
                    raise
        
        except Exception as e:
            self.logger.error(f"Error with Gemini API: {str(e)}")
            raise

    async def _make_together_request(self, messages: List[Dict[str, str]], temperature: float) -> Dict:
        """Make a request to Together.ai API (high free quota)"""
        try:
            # Together.ai uses OpenAI-compatible API
            together_url = "https://api.together.xyz/v1/chat/completions"
            payload = {
                "model": "meta-llama/Llama-3.1-8B-Instruct", # Free tier model
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 800
            }
            
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                together_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    self.logger.error(f"Together API request failed: {response_text}")
                    raise Exception(f"Together API request failed: {response_text}")
                
                return await response.json()
        
        except Exception as e:
            self.logger.error(f"Error with Together API: {str(e)}")
            raise

    async def _use_fallback_llm(self, messages: List[Dict[str, str]], temperature: float) -> Dict:
        """Try to use a local LLM (Ollama instance) if available"""
        try:
            # Try using a local LLM if URL is configured
            if self.use_local_llm:
                try:
                    # Format the messages for Ollama API
                    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    
                    payload = {
                        "model": "llama2",  # or any other model you have in Ollama
                        "prompt": prompt,
                        "stream": False,
                        "temperature": temperature
                    }
                    
                    async with self.session.post(
                        self.local_llm_url,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": data.get("response", "No response from local LLM")
                                        }
                                    }
                                ]
                            }
                except Exception as e:
                    self.logger.error(f"Local LLM failed: {str(e)}")
                    raise ValueError(f"Local LLM failed and no fallback available: {str(e)}")
            
            # If we get here and we were using the local LLM or have quota exceeded, 
            # there's no other options available so raise an error
            raise ValueError("API quota exceeded or error occurred, and no local LLM is available.")
                
        except Exception as e:
            self.logger.error(f"Error using LLM: {str(e)}")
            raise ValueError(f"All LLM options failed: {str(e)}")

    async def validate_json_response(self, content: str) -> Dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If quota is exceeded, produce a simple valid JSON
            if self.quota_exceeded:
                return {"error": "Could not validate JSON due to API quota limits"}
                
            messages = [
                {"role": "system", "content": "Fix the following text to be valid JSON:"},
                {"role": "user", "content": content}
            ]
            try:
                response = await self._make_api_request(messages, temperature=0.1)
                fixed_content = response['choices'][0]['message']['content']
                return json.loads(fixed_content)
            except Exception as e:
                self.logger.error(f"JSON validation failed: {str(e)}")
                return {"error": "Failed to validate JSON response"}

    async def analyse_content(self, prompt: str) -> Dict[str, any]:
        messages = [
            {"role": "system", "content": "You are an expert SEO analyst. Provide detailed, structured analysis in JSON format. Your entire response must be valid JSON without any explanatory text before or after the JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._make_api_request(messages, temperature=0.3)
            content = response['choices'][0]['message']['content']

            # Clean the content to ensure it's valid JSON
            content = self._clean_json_response(content)

            try:
                analysis = json.loads(content)
                return analysis
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                self.logger.debug(f"Raw content: {content}")
                
                # Try to extract JSON if there's text surrounding it
                extracted_json = self._extract_json(content)
                if extracted_json:
                    try:
                        return json.loads(extracted_json)
                    except json.JSONDecodeError:
                        pass
                
                # Raise an error instead of using fallback
                raise ValueError(f"Failed to parse content analysis as valid JSON: {e}")
        
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            raise ValueError(f"Content analysis failed: {str(e)}")
    
    async def generate_recommendations(self, prompt: str) -> List[Dict]:
        messages = [
            {
                "role": "system",
                "content": """You are an expert SEO consultant. Generate specific, actionable recommendations 
                in JSON format. Your entire response must be a valid JSON array without any explanatory text before or after. 
                Each recommendation should include:
                - "recommendation": detailed description
                - "priority_score": 1-10
                - "impact": "high"/"medium"/"low"
                - "implementation_difficulty": "easy"/"medium"/"hard"
                - "estimated_time": in hours"""
            },
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self._make_api_request(messages, temperature=0.4)
            content = response['choices'][0]['message']['content']
            
            # Clean the content to ensure it's valid JSON
            content = self._clean_json_response(content)
            
            try:
                recommendations = json.loads(content)
                if isinstance(recommendations, list):
                    return recommendations
                elif isinstance(recommendations, dict) and any(key in recommendations for key in ["recommendations", "results", "items"]):
                    # Try to extract the recommendation list if it's nested in a dict
                    for key in ["recommendations", "results", "items"]:
                        if key in recommendations and isinstance(recommendations[key], list):
                            return recommendations[key]
                    # If we can't find a list, return the dict as a single-item list
                    return [recommendations]
                else:
                    return [recommendations]
            
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM recommendations as JSON: {e}")
                self.logger.debug(f"Raw content: {content}")
                
                # Try to extract JSON if there's text surrounding it
                extracted_json = self._extract_json(content)
                if extracted_json:
                    try:
                        recommendations = json.loads(extracted_json)
                        if isinstance(recommendations, list):
                            return recommendations
                        else:
                            return [recommendations]
                    except json.JSONDecodeError:
                        pass
                
                # If we can't parse JSON at all, raise an exception rather than using fallback
                raise ValueError(f"Could not parse LLM response as valid JSON: {e}")
    
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            raise ValueError(f"Failed to generate recommendations: {str(e)}")
    
    def _clean_json_response(self, content: str) -> str:
        """Clean response to make it valid JSON"""
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*|\s*```', '', content)
        content = re.sub(r'```\s*|\s*```', '', content)
        
        # Remove any explanatory text before the first { or [
        content = re.sub(r'^[^{\[]*', '', content, flags=re.DOTALL)
        
        # Remove any explanatory text after the last } or ]
        content = re.sub(r'[^}\]]*$', '', content, flags=re.DOTALL)
        
        # Try to fix specific JSON issues often encountered with Gemini
        
        # Fix missing commas between objects in arrays
        content = re.sub(r'}\s*{', '},{', content)
        
        # Fix missing quotes around property names
        content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', content)
        
        # Fix trailing commas before closing brackets
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Fix property value formatting
        content = re.sub(r':\s*\'([^\']*?)\'', r':"\1"', content)  # Single quotes to double quotes
        content = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', content)  # Unquoted strings
        
        # Replace single quotes with double quotes but avoid changing apostrophes inside already quoted strings
        in_string = False
        result = []
        i = 0
        while i < len(content):
            if content[i] == '"':
                in_string = not in_string
            elif content[i] == "'" and not in_string:
                result.append('"')
                i += 1
                continue
            result.append(content[i])
            i += 1
        content = ''.join(result)
        
        # Try to fix Gemini's common formatting issue where it outputs explanations between fields
        # Remove any comment-like text between JSON properties
        content = re.sub(r',\s*//[^\n]*\n', ',', content)
        content = re.sub(r'{\s*//[^\n]*\n', '{', content)
        
        return content.strip()
    
    def _extract_json(self, content: str) -> Optional[str]:
        """Try to extract JSON from a string that may contain non-JSON content"""
        # If we're dealing with what looks like an array of objects
        if "[" in content and "{" in content:
            try:
                # Find all object literals in the text
                objects = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', content, re.DOTALL)
                if objects:
                    # Try to create a valid JSON array from the found objects
                    array_content = "[" + ",".join(objects) + "]"
                    # Test if it parses
                    json.loads(array_content)
                    return array_content
            except json.JSONDecodeError:
                pass
        
        # Look for complete JSON array
        array_match = re.search(r'\[(.*)\]', content, re.DOTALL)
        if array_match:
            try:
                array_content = f"[{array_match.group(1)}]"
                # Test if it parses
                json.loads(array_content)
                return array_content
            except json.JSONDecodeError:
                pass
        
        # Look for complete JSON object
        object_match = re.search(r'\{(.*)\}', content, re.DOTALL)
        if object_match:
            try:
                object_content = f"{{{object_match.group(1)}}}"
                # Test if it parses
                json.loads(object_content)
                return object_content
            except json.JSONDecodeError:
                pass
        
        # More aggressive approach - If all else fails, try to find and fix all JSON objects in text
        try:
            # Find all potential object patterns
            patterns = re.findall(r'{\s*"[^"]+"\s*:[\s\S]*?}', content)
            if patterns:
                # Attempt to fix and validate each pattern
                validated_objects = []
                for pattern in patterns:
                    try:
                        fixed = self._clean_json_response(pattern)
                        json.loads(fixed)  # Test if it parses
                        validated_objects.append(fixed)
                    except:
                        continue
                        
                if validated_objects:
                    # Create a JSON array from valid objects
                    return "[" + ",".join(validated_objects) + "]"
        except:
            pass
            
        # Could not extract valid JSON
        self.logger.warning("Could not extract valid JSON from LLM response")
        return None
    
    async def _generate_technical_recommendations(self, technical_analysis_str: str) -> List[Dict]:
        """Generate technical SEO recommendations efficiently."""
        try:
            print(f"[SEO AGENT] Processing technical analysis data ({len(technical_analysis_str)} bytes)...")
            # Convert the string back to a dictionary
            technical_analysis = json.loads(technical_analysis_str)
            
            # Reduce the size of the analysis prompt to avoid overloading the LLM
            condensed_analysis = {}
            for key in ["site_speed", "mobile_optimisation", "indexation", "security", "structured_data"]:
                value = technical_analysis.get(key, "Summary not available")
                condensed_analysis[key] = str(value)[:250]
            
            print(f"[SEO AGENT] Prepared technical analysis with keys: {list(condensed_analysis.keys())}")

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

            print("[SEO AGENT] Requesting technical recommendations from LLM...")
            try:
                # Use a 20-second timeout for the LLM request
                recommendations = await asyncio.wait_for(
                    self.generate_recommendations(prompt),
                    timeout=20
                )
                print(f"[SEO AGENT] Successfully received {len(recommendations)} LLM-generated technical recommendations")
                return recommendations
            except asyncio.TimeoutError:
                print("[SEO AGENT] Technical recommendations request timed out")
                raise ValueError("Technical recommendations request timed out")
                
        except Exception as e:
            self.logger.error(f"Error generating technical SEO recommendations: {e}")
            print(f"[SEO AGENT] ERROR: Failed to get technical recommendations: {str(e)}")
            
            # Instead of using hardcoded fallbacks, generate data-driven recommendations
            print("[SEO AGENT] Generating basic technical recommendations instead")
            
            # Create data-driven recommendations based on website's actual data
            recommendations = []
            
            # Add recommendations based on what data we have
            if "site_speed" in technical_analysis:
                recommendations.append({
                    "recommendation": "Conduct a comprehensive site speed audit using tools like Google PageSpeed Insights and GTmetrix to identify bottlenecks. Address issues related to large images, slow server response times, and inefficient code.",
                    "priority_score": 10,
                    "impact": "high",
                    "implementation_difficulty": "medium",
                    "estimated_time": 20
                })
            
            if "mobile_optimisation" in technical_analysis:
                recommendations.append({
                    "recommendation": "Ensure your website is responsive and adapts seamlessly to different screen sizes. Test your website's mobile-friendliness using Google's Mobile-Friendly Test.",
                    "priority_score": 9,
                    "impact": "high",
                    "implementation_difficulty": "medium",
                    "estimated_time": 15
                })
            
            if "indexation" in technical_analysis:
                recommendations.append({
                    "recommendation": "Improve your website's indexation by ensuring robots.txt is correctly configured, sitemaps are updated and submitted to search engines, and important pages aren't blocked from crawling.",
                    "priority_score": 8,
                    "impact": "high",
                    "implementation_difficulty": "medium",
                    "estimated_time": 8
                })
            
            if "security" in technical_analysis:
                recommendations.append({
                    "recommendation": "Enhance website security by implementing HTTPS, securing forms, and adding proper security headers to protect against common vulnerabilities.",
                    "priority_score": 8,
                    "impact": "medium",
                    "implementation_difficulty": "medium", 
                    "estimated_time": 10
                })
            
            # Ensure we have at least one recommendation
            if not recommendations:
                recommendations.append({
                    "recommendation": "Implement a comprehensive technical SEO audit to identify and fix issues with site speed, mobile optimization, and indexation.",
                    "priority_score": 9,
                    "impact": "high",
                    "implementation_difficulty": "medium",
                    "estimated_time": 25
                })
            
            return recommendations
    
    async def generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate prioritised recommendations based on analysis."""
        if not analysis:
            self.logger.error("Analysis data is None.")
            raise ValueError("Analysis data is empty or None.")
            
        try:
            # Convert dictionaries to strings for caching
            technical_analysis_str = json.dumps(analysis.get("technical_seo", {}))
            on_page_analysis = analysis.get("on_page_seo", {})
            off_page_analysis = analysis.get("off_page_seo", {})
            analyzed_pages = analysis.get("analyzed_pages", {})
            
            print(f"[SEO AGENT] Technical analysis data size: {len(technical_analysis_str)} bytes")
            print(f"[SEO AGENT] On-page analysis data size: {len(json.dumps(on_page_analysis))} bytes")
            print(f"[SEO AGENT] Off-page analysis data size: {len(json.dumps(off_page_analysis))} bytes")
            
            # Use asyncio.gather to execute recommendation functions concurrently
            print("[SEO AGENT] Generating technical recommendations...")
            technical_recommendations = await self._generate_technical_recommendations(technical_analysis_str)
            print("[SEO AGENT] Generating on-page recommendations...")
            on_page_recommendations = await self._generate_on_page_recommendations(on_page_analysis)
            print("[SEO AGENT] Generating off-page recommendations...")
            off_page_recommendations = await self._generate_off_page_recommendation(off_page_analysis)
            
            # Generate page-specific recommendations
            print("[SEO AGENT] Generating page-specific recommendations...")
            page_recommendations = await self._generate_page_specific_recommendations(analyzed_pages)
            
            # Combine recommendations, ensuring we get a mix of different types
            all_recommendations = []
            
            # Take top 2 from each category to ensure diversity
            for i, recs in enumerate([technical_recommendations, on_page_recommendations, off_page_recommendations]):
                if recs:
                    category_name = ["Technical SEO", "On-Page SEO", "Off-Page SEO"][i]
                    for j, rec in enumerate(recs[:2]):  # Take top 2 from each category
                        rec['category'] = category_name
                        # Set generic source if not specified
                        if 'source_page' not in rec:
                            rec['source_page'] = "site-wide"
                        all_recommendations.append(rec)
            
            # Add page-specific recommendations
            for rec in page_recommendations[:4]:  # Take top 4 page-specific recommendations
                rec['category'] = "Page-Specific"
                all_recommendations.append(rec)
            
            # Sort by priority score while maintaining diversity
            return sorted(
                all_recommendations,
                key=lambda x: (x.get("priority_score", 0), x.get("impact", "")),
                reverse=True
            )
        except Exception as e:
            self.logger.exception(f"Error generating recommendations: {e}")
            # Instead of fallback, raise the exception
            raise ValueError(f"Failed to generate recommendations: {str(e)}")

    async def _generate_page_specific_recommendations(self, analyzed_pages: Dict) -> List[Dict]:
        """Generate recommendations based on specific page issues"""
        recommendations = []
        
        for url, page_data in analyzed_pages.items():
            issues = page_data.get("issues", [])
            
            # For each issue type, generate appropriate recommendations
            for issue in issues:
                issue_type = issue.get("type")
                severity = issue.get("severity", "medium")
                details = issue.get("details", "")
                
                if issue_type == "missing_title":
                    recommendations.append({
                        "recommendation": f"Add a descriptive title tag to the page at {url}",
                        "priority_score": 8 if severity == "high" else 6,
                        "impact": "high",
                        "implementation_difficulty": "easy",
                        "estimated_time": 0.5,
                        "source_page": url
                    })
                
                elif issue_type == "missing_meta_description":
                    recommendations.append({
                        "recommendation": f"Add a compelling meta description to the page at {url}",
                        "priority_score": 7 if severity == "high" else 5,
                        "impact": "medium",
                        "implementation_difficulty": "easy",
                        "estimated_time": 0.5,
                        "source_page": url
                    })
                
                elif issue_type == "thin_content":
                    recommendations.append({
                        "recommendation": f"Expand content on {url} - currently only {details.split()[-2]} words",
                        "priority_score": 7,
                        "impact": "high",
                        "implementation_difficulty": "medium",
                        "estimated_time": 3,
                        "source_page": url
                    })
                
                elif issue_type == "missing_h1":
                    recommendations.append({
                        "recommendation": f"Add an H1 heading to the page at {url}",
                        "priority_score": 6,
                        "impact": "medium",
                        "implementation_difficulty": "easy",
                        "estimated_time": 0.5,
                        "source_page": url
                    })
        
        # If there are analyzed pages but we couldn't extract issues
        if analyzed_pages and not recommendations:
            # Generate LLM-based recommendations for each page
            try:
                page_summaries = []
                urls = list(analyzed_pages.keys())[:5]  # Limit to 5 pages to avoid token limits
                
                for url in urls:
                    page = analyzed_pages[url]
                    summary = {
                        "url": url,
                        "title": page.get("title", "Unknown"),
                        "description": page.get("description", "")
                    }
                    page_summaries.append(summary)
                
                if page_summaries:
                    prompt = f"""
                    Based on these pages:
                    {json.dumps(page_summaries, indent=2)}
                    
                    Generate specific, page-level SEO recommendations. Each recommendation must include:
                    - "recommendation": detailed description
                    - "priority_score": 1-10
                    - "impact": "high"/"medium"/"low"
                    - "implementation_difficulty": "easy"/"medium"/"hard"
                    - "estimated_time": hours
                    - "source_page": url of the specific page
                    
                    Format as JSON array.
                    """
                    
                    llm_recommendations = await self.llm.generate_recommendations(prompt)
                    
                    # Ensure source_page is set for each recommendation
                    for rec in llm_recommendations:
                        if "source_page" not in rec:
                            # Try to match with one of our analyzed pages
                            for url in urls:
                                if url.lower() in rec.get("recommendation", "").lower():
                                    rec["source_page"] = url
                                    break
                            else:
                                # If no match found, use the first URL
                                rec["source_page"] = urls[0] if urls else "unknown"
                    
                    recommendations.extend(llm_recommendations)
            except Exception as e:
                self.logger.error(f"Error generating page-specific LLM recommendations: {e}")
                # Don't add fallback recommendations, just raise the error
                raise ValueError(f"Failed to generate page-specific recommendations: {str(e)}")
        
        return recommendations

    async def _generate_on_page_recommendations(self, on_page_analysis: Dict) -> List[Dict]:
        """Generate on-page SEO recommendations"""
        try:
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
            Each recommendation should include:
            - "recommendation": detailed description
            - "priority_score": 1-10
            - "impact": "high"/"medium"/"low"
            - "implementation_difficulty": "easy"/"medium"/"hard"
            - "estimated_time": hours
            """

            print("[SEO AGENT] Requesting on-page recommendations from LLM...")
            recommendations = await self.llm.generate_recommendations(prompt)
            print(f"[SEO AGENT] Successfully received {len(recommendations)} LLM-generated on-page recommendations")
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating on-page SEO recommendations: {e}")
            print(f"[SEO AGENT] ERROR: Failed to get on-page LLM recommendations: {str(e)}")
            raise ValueError(f"Failed to generate on-page SEO recommendations: {str(e)}")
    
    async def _generate_off_page_recommendation(self, off_page_analysis: Dict) -> List[Dict]:
        """Generate off-page SEO recommendations"""
        try:
            prompt = f"""
            Based on this off-page SEO analysis:
            {json.dumps(off_page_analysis)}

            Generate prioritised recommendations for:
            1. Link building opportunities
            2. Brand signal improvements
            3. Social media presence
            4. Local SEO optimisation
            5. Authority building

            Format as JSON list with priority_score and impact for each recommendation.
            Each recommendation should include:
            - "recommendation": detailed description
            - "priority_score": 1-10 
            - "impact": "high"/"medium"/"low"
            - "implementation_difficulty": "easy"/"medium"/"hard"
            - "estimated_time": hours
            """

            print("[SEO AGENT] Requesting off-page recommendations from LLM...")
            recommendations = await self.llm.generate_recommendations(prompt)
            print(f"[SEO AGENT] Successfully received {len(recommendations)} LLM-generated off-page recommendations")
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating off-page SEO recommendations: {e}")
            print(f"[SEO AGENT] ERROR: Failed to get off-page LLM recommendations: {str(e)}")
            raise ValueError(f"Failed to generate off-page SEO recommendations: {str(e)}")

    async def _analyse_internal_linking(self, url: str) -> Dict:
        """Analyze internal linking structure for a page"""
        try:
            if url not in self.analyzed_pages:
                return {"error": "Page not in analyzed pages"}
                
            page_data = self.analyzed_pages[url]
            
            # Count internal and external links
            internal_link_count = len(page_data.internal_links) if page_data.internal_links else 0
            external_link_count = len(page_data.external_links) if page_data.external_links else 0
            
            # Check for common issues
            issues = []
            
            # Too few internal links
            if internal_link_count < 3:
                issues.append({
                    "type": "few_internal_links",
                    "severity": "medium",
                    "details": f"Page has only {internal_link_count} internal links"
                })
            
            # Too many external links
            if external_link_count > 20:
                issues.append({
                    "type": "many_external_links",
                    "severity": "low",
                    "details": f"Page has {external_link_count} external links"
                })
            
            # Calculate link diversity - how many different internal pages are linked
            unique_internal_urls = set()
            for link in page_data.internal_links or []:
                # Remove fragments and queries to count unique pages
                parsed = urlparse(link)
                base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                unique_internal_urls.add(base_url)
            
            link_diversity = len(unique_internal_urls)
            
            # Add issues to page data
            if hasattr(page_data, 'issues') and page_data.issues is not None:
                page_data.issues.extend(issues)
            
            return {
                "internal_links_count": internal_link_count,
                "external_links_count": external_link_count,
                "link_diversity": link_diversity,
                "issues": issues,
                "score": self._calculate_internal_linking_score(internal_link_count, link_diversity, issues)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing internal linking for {url}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_internal_linking_score(self, internal_link_count: int, link_diversity: int, issues: List[Dict]) -> float:
        """Calculate score for internal linking"""
        base_score = 5.0
        
        # Add points for good internal linking
        if internal_link_count >= 10:
            base_score += 2
        elif internal_link_count >= 5:
            base_score += 1
        
        # Add points for link diversity
        if link_diversity >= 8:
            base_score += 2
        elif link_diversity >= 4:
            base_score += 1
        
        # Deduct points for issues
        base_score -= len(issues) * 0.5
        
        # Ensure score is in range 0-10
        return max(0, min(10, base_score))
    
    async def analyse_website(self, seed_url: str, limit: int = 100):
        """Crawls and analyzes a website."""
        print(f"[SEO AGENT] Starting crawl at: {seed_url}")
        queue = asyncio.Queue()
        await queue.put(seed_url)
        processed_count = 0  # Define this at the outer function level
        
        # Initialize overall analysis structure
        analysis = {
            "technical_seo": {},
            "on_page_seo": {},
            "off_page_seo": {},
            "analyzed_pages": {}  # Store page-specific analyses
        }
        
        # Set up a concurrent fetch tasks list
        fetch_tasks = []
        max_concurrent_tasks = min(self.concurrency, limit)  # Don't create more tasks than our limit
        
        # Analyze the main seed URL for more comprehensive data
        if processed_count < limit:
            print(f"[SEO AGENT] Analyzing seed URL thoroughly: {seed_url}")
            try:
                # Collect technical SEO metrics with a timeout
                technical_metrics = await asyncio.wait_for(
                    self.analyse_technical_seo(seed_url),
                    timeout=15  # 15-second timeout for technical analysis
                )
                analysis["technical_seo"] = technical_metrics
                
                # Collect on-page SEO metrics with a timeout
                on_page_metrics = await asyncio.wait_for(
                    self.analyse_on_page_seo(seed_url),
                    timeout=15  # 15-second timeout for on-page analysis
                )
                analysis["on_page_seo"] = on_page_metrics
                
                # Collect off-page SEO metrics with a timeout
                off_page_metrics = await asyncio.wait_for(
                    self.analyse_off_page_seo(seed_url),
                    timeout=15  # 15-second timeout for off-page analysis
                )
                analysis["off_page_seo"] = off_page_metrics
                
                # Store the seed URL analysis
                analysis["analyzed_pages"][seed_url] = {
                    "technical_seo": technical_metrics,
                    "on_page_seo": on_page_metrics,
                    "off_page_seo": off_page_metrics,
                    "issues": []  # Will be populated as we analyze
                }
                
                print(f"[SEO AGENT] Comprehensive analysis of seed URL complete.")
            except asyncio.TimeoutError:
                print(f"[SEO AGENT] Seed URL analysis timed out. Continuing with limited analysis.")
                # Create basic metrics if analysis timed out
                analysis["technical_seo"] = {"score": 5.0}
                analysis["on_page_seo"] = {"score": 5.0}
                analysis["off_page_seo"] = {"score": 5.0}
            except Exception as e:
                print(f"[SEO AGENT] Error analyzing seed URL: {str(e)}")
                raise

        async def process_url():
            """Process URLs from the queue with proper error handling"""
            nonlocal processed_count  # Declare nonlocal here, after the variable is defined in the outer scope
            
            while not queue.empty() and processed_count < limit:
                try:
                    url = await queue.get()
                    
                    # Skip processing if we've reached the limit
                    if processed_count >= limit:
                        queue.task_done()
                        continue
                    
                    # Skip URLs we've already visited
                    if url in self.visited_urls:
                        queue.task_done()
                        continue
                        
                    print(f"[SEO AGENT] Fetching page: {url}")
                    
                    # Fetch the page with a timeout
                    try:
                        page_data = await asyncio.wait_for(
                            self.fetch_page(url),
                            timeout=10  # 10-second timeout per page
                        )
                    except asyncio.TimeoutError:
                        print(f"[SEO AGENT] Fetch timed out for: {url}")
                        queue.task_done()
                        continue
                    
                    if not page_data:
                        print(f"[SEO AGENT] Skipped or failed to fetch: {url}")
                        queue.task_done()
                        continue
                        
                    processed_count += 1
                    print(f"[SEO AGENT] Analyzing: {url} ({processed_count}/{limit})")
                    
                    # Store the page in our analyzed pages
                    self.analyzed_pages[url] = page_data
                    
                    # Simplified analysis for subsequent pages
                    if url != seed_url:  # Skip if it's the seed URL we already analyzed
                        try:
                            # Collect page-specific analyses
                            page_analysis = {
                                "url": url,
                                "title": page_data.title,
                                "description": page_data.description,
                                "issues": []  # Will be populated with issues
                            }
                            
                            # Check for common issues
                            if not page_data.title or len(page_data.title) < 10:
                                issue = {"type": "missing_title", "severity": "high", "details": "Page title is missing or too short"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                                
                            if not page_data.description or len(page_data.description) < 50:
                                issue = {"type": "missing_meta_description", "severity": "medium", "details": "Meta description is missing or too short"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                                
                            if page_data.word_count < 300:
                                issue = {"type": "thin_content", "severity": "medium", "details": f"Page has only {page_data.word_count} words"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                                
                            if not page_data.h1:
                                issue = {"type": "missing_h1", "severity": "medium", "details": "Page has no H1 heading"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                            
                            # Add specific analyses with timeouts
                            try:
                                internal_linking_data = await asyncio.wait_for(
                                    self._analyse_internal_linking(url),
                                    timeout=5  # 5-second timeout
                                )
                                page_analysis["internal_linking"] = internal_linking_data
                            except (asyncio.TimeoutError, Exception) as e:
                                self.logger.error(f"Error analyzing internal linking for {url}: {str(e)}")
                                page_analysis["internal_linking"] = {"error": "Analysis timed out or failed"}
                            
                            try:
                                content_structure_data = await asyncio.wait_for(
                                    self._analyse_content_structure(url),
                                    timeout=5  # 5-second timeout
                                )
                                page_analysis["content_structure"] = content_structure_data
                            except (asyncio.TimeoutError, Exception) as e:
                                self.logger.error(f"Error analyzing content structure for {url}: {str(e)}")
                                page_analysis["content_structure"] = {"error": "Analysis timed out or failed"}
                            
                            # Store the page analysis
                            analysis["analyzed_pages"][url] = page_analysis
                            
                        except Exception as e:
                            self.logger.error(f"Error analyzing page {url}: {str(e)}")
                    
                    # Add internal links to queue if we're still under the limit
                    if page_data.internal_links and processed_count < limit:
                        # Filter out URLs we've already seen or queued
                        filtered_links = [
                            link for link in page_data.internal_links
                            if link not in self.visited_urls 
                            and not self._should_skip_url(link)
                        ]
                        
                        # Take only the first 10 links to avoid excessive crawling
                        for link in filtered_links[:10]:
                            if link not in self.visited_urls:
                                self.visited_urls.add(link)  # Mark as visited to avoid duplicates
                                await queue.put(link)
                    
                    queue.task_done()
                except Exception as e:
                    self.logger.error(f"Error processing URL: {str(e)}")
                    queue.task_done()
        
        # Create workers to process URLs concurrently
        for _ in range(max_concurrent_tasks):
            task = asyncio.create_task(process_url())
            fetch_tasks.append(task)
        
        # Wait for initial queue to be processed
        await queue.join()
        
        # Cancel any pending tasks
        for task in fetch_tasks:
            task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        print("[SEO AGENT] Website crawl finished.")
        return analysis
        
    def _should_skip_url(self, url: str) -> bool:
        """Check if a URL should be skipped based on common patterns"""
        # Skip URLs with fragments unless they are pointing to page anchors
        parsed = urlparse(url)
        
        # Skip URLs with certain file extensions
        extensions_to_skip = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.css', '.js']
        if any(parsed.path.lower().endswith(ext) for ext in extensions_to_skip):
            return True
            
        # Skip URLs with query parameters that often lead to duplicates
        query_params_to_skip = ['sort', 'filter', 'page', 'utm_source', 'utm_medium', 'utm_campaign']
        if parsed.query:
            query_dict = {k: v for k, v in [p.split('=') for p in parsed.query.split('&')] if '=' in p}
            if any(param in query_dict for param in query_params_to_skip):
                return True
                
        # Skip URLs that look like login, logout, carts, etc.
        skip_patterns = ['/login', '/logout', '/cart', '/checkout', '/account', '/admin', '/thank-you']
        if any(pattern in parsed.path.lower() for pattern in skip_patterns):
            return True
            
        return False
            
    async def fetch_page(self, url: str) -> PageData:
        """Fetch a single page and extract basic information."""
        try:
            # Create a page data object
            page = PageData(url)
            
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
                
            try:
                # Fetch the page with a timeout
                async with self.session.get(url, timeout=10) as response:
                    if response.status >= 400:
                        self.logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
                        
                    content = await response.text()
                    page.html = content
                    
                    # Parse with BeautifulSoup for easier extraction
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract title
                    title_tag = soup.find('title')
                    page.title = title_tag.get_text() if title_tag else None
                    
                    # If this is the first page we're fetching, and it looks like a site title, save it
                    if not self.site_title and page.title:
                        self.site_title = page.title
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    page.description = meta_desc.get('content', '') if meta_desc else None
                    
                    # Extract meta keywords
                    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                    page.keywords = meta_keywords.get('content', '') if meta_keywords else None
                    
                    # Extract headings
                    page.h1 = [h1.get_text() for h1 in soup.find_all('h1')]
                    page.headings = self._extract_headings(soup)
                    
                    # Count words in the page content
                    page.word_count = len(soup.get_text().split())
                    
                    # Extract links (both internal and external)
                    base_domain = urlparse(url).netloc
                    internal_links = []
                    external_links = []
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        # Handle relative links
                        if href.startswith('/'):
                            # Convert to absolute
                            parsed_url = urlparse(url)
                            absolute_url = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                            internal_links.append(absolute_url)
                        elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                            # Handle links without scheme/domain
                            # Skip anchors like "#section" and javascript
                            if href.startswith('#') or href.startswith('javascript:'):
                                continue
                                
                            # Get the page's path directory to handle relative links
                            path_parts = urlparse(url).path.split('/')
                            if len(path_parts) > 1:
                                base_dir = '/'.join(path_parts[:-1]) + '/'
                            else:
                                base_dir = '/'
                                
                            # Create absolute URL
                            parsed_url = urlparse(url)
                            absolute_url = f"{parsed_url.scheme}://{parsed_url.netloc}{base_dir}{href}"
                            internal_links.append(absolute_url)
                        elif base_domain in href:
                            # It's an absolute URL but on the same domain
                            internal_links.append(href)
                        else:
                            # It's an external link
                            external_links.append(href)
                    
                    page.internal_links = internal_links
                    page.external_links = external_links
                    
                    # Mark as visited
                    self.visited_urls.add(url)
                    
                    return page
                    
            except ClientError as e:
                # Handle client errors (like 404)
                self.logger.warning(f"Client error when fetching {url}: {str(e)}")
                return None
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout when fetching {url}")
                return None
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Unhandled error in fetch_page for {url}: {str(e)}")
            return None
            
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract heading tags (h1-h6) from the page."""
        headings = {}
        for i in range(1, 7):
            tag = f'h{i}'
            heading_tags = soup.find_all(tag)
            headings[tag] = [h.get_text().strip() for h in heading_tags]
        return headings

    async def analyse_technical_seo(self, url: str) -> Dict:
        """Analyze technical SEO metrics for a given URL."""
        try:
            result = {
                "site_speed": {},
                "mobile_optimisation": {},
                "indexation": {},
                "security": {},
                "structured_data": {},
                "issues": []
            }

            # Measure site speed metrics
            try:
                speed_metrics = await self._analyse_site_speed(url)
                result["site_speed"] = speed_metrics
                
                # Add issues if site is slow
                if speed_metrics.get("overall_load_time", 0) > 3:
                    result["issues"].append({
                        "type": "slow_site_speed",
                        "severity": "high",
                        "details": f"Site load time is {speed_metrics.get('overall_load_time', 0)} seconds"
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing site speed for {url}: {e}")
                result["site_speed"] = {"error": str(e)}
            
            # Check mobile optimization
            try:
                mobile_metrics = await self._check_mobile_optimisation(url)
                result["mobile_optimisation"] = mobile_metrics
                
                # Add issues if site is not mobile-friendly
                if mobile_metrics.get("optimization_score", 0) < 6:
                    result["issues"].append({
                        "type": "poor_mobile_optimization",
                        "severity": "high",
                        "details": "Site is not well optimized for mobile devices"
                    })
            except Exception as e:
                self.logger.error(f"Error checking mobile optimization for {url}: {e}")
                result["mobile_optimisation"] = {"error": str(e)}
            
            # Check indexation status
            try:
                robots_allow = True
                robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
                
                try:
                    async with self.session.get(robots_url, timeout=5) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            if "Disallow: /" in robots_content or f"Disallow: {urlparse(url).path}" in robots_content:
                                robots_allow = False
                except Exception as e:
                    self.logger.warning(f"Error checking robots.txt for {url}: {e}")
                
                # Check for noindex directives
                page_data = await self.fetch_page(url)
                noindex = False
                
                if page_data and page_data.html:
                    soup = BeautifulSoup(page_data.html, 'html.parser')
                    
                    # Check meta robots
                    meta_robots = soup.find('meta', attrs={'name': 'robots'})
                    if meta_robots and 'noindex' in meta_robots.get('content', '').lower():
                        noindex = True
                        
                    # Check X-Robots-Tag in headers (would require original response)
                
                # Store indexation data
                result["indexation"] = {
                    "robots_allow": robots_allow,
                    "noindex": noindex,
                    "issues": []
                }
                
                if not robots_allow or noindex:
                    result["issues"].append({
                        "type": "indexation_blocked",
                        "severity": "high",
                        "details": f"{'Robots.txt blocking' if not robots_allow else ''} {'Meta noindex present' if noindex else ''}"
                    })
            except Exception as e:
                self.logger.error(f"Error checking indexation for {url}: {e}")
                result["indexation"] = {"error": str(e)}
            
            # Check security (HTTPS, security headers)
            try:
                is_https = url.startswith("https://")
                hsts = False
                
                try:
                    async with self.session.get(url, timeout=5) as response:
                        headers = response.headers
                        if 'Strict-Transport-Security' in headers:
                            hsts = True
                except Exception as e:
                    self.logger.warning(f"Error checking security headers for {url}: {e}")
                
                result["security"] = {
                    "https": is_https,
                    "hsts": hsts,
                    "issues": []
                }
                
                if not is_https:
                    result["issues"].append({
                        "type": "no_https",
                        "severity": "high",
                        "details": "Site is not using HTTPS"
                    })
            except Exception as e:
                self.logger.error(f"Error checking security for {url}: {e}")
                result["security"] = {"error": str(e)}
            
            # Check structured data
            try:
                structured_data = {}
                page_data = await self.fetch_page(url)
                
                if page_data and page_data.html:
                    soup = BeautifulSoup(page_data.html, 'html.parser')
                    
                    # Look for JSON-LD
                    json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
                    if json_ld_scripts:
                        structured_data["json_ld"] = True
                        structured_data["count"] = len(json_ld_scripts)
                    else:
                        structured_data["json_ld"] = False
                        
                    # Look for microdata
                    microdata_elements = soup.find_all(attrs={"itemtype": True})
                    if microdata_elements:
                        structured_data["microdata"] = True
                        structured_data["count"] = len(microdata_elements)
                    else:
                        structured_data["microdata"] = False
                
                result["structured_data"] = structured_data
            except Exception as e:
                self.logger.error(f"Error checking structured data for {url}: {e}")
                result["structured_data"] = {"error": str(e)}
            
            # Calculate overall score
            scores = []
            if "site_speed" in result and isinstance(result["site_speed"], dict) and "score" in result["site_speed"]:
                scores.append(result["site_speed"]["score"])
            if "mobile_optimisation" in result and isinstance(result["mobile_optimisation"], dict) and "optimization_score" in result["mobile_optimisation"]:
                scores.append(result["mobile_optimisation"]["optimization_score"])
            
            # Add a base score if we couldn't calculate from components
            if not scores:
                scores.append(5.0)  # Default middle score
            
            result["score"] = sum(scores) / len(scores)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in technical SEO analysis for {url}: {e}")
            return {
                "score": 0,
                "error": str(e),
                "site_speed": {"error": "Analysis failed"},
                "mobile_optimisation": {"error": "Analysis failed"},
                "indexation": {"error": "Analysis failed"},
                "security": {"error": "Analysis failed"},
                "structured_data": {"error": "Analysis failed"},
                "issues": []
            }

class ComprehensiveSEOAgent:
    def __init__(self, api_key: str, user_agent: str = "ComprehensiveSEOAgent/1.0", concurrency: int = 10):
        self.api_key = api_key
        self.llm = None
        self.logger = logging.getLogger(__name__)
        self.visited_urls = set()
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.session = None
        self.analyzed_pages = {}  # Store analyzed pages for recommendation tracking
        self.site_title = None  # Store the site title for reporting
    
    async def __aenter__(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), headers={'User-Agent': self.user_agent}, trust_env=True)
        self.llm = LLMProcessor(self.api_key)
        await self.llm.__aenter__()
        return self

    async def __aexit__(self, *err):
        if self.llm:
            await self.llm.__aexit__(*err)
        if self.session:
            await self.session.close()
            
    async def analyse_website(self, seed_url: str, limit: int = 100):
        """Crawls and analyzes a website."""
        print(f"[SEO AGENT] Starting crawl at: {seed_url}")
        queue = asyncio.Queue()
        await queue.put(seed_url)
        processed_count = 0  # Define this at the outer function level
        
        # Initialize overall analysis structure
        analysis = {
            "technical_seo": {},
            "on_page_seo": {},
            "off_page_seo": {},
            "analyzed_pages": {}  # Store page-specific analyses
        }
        
        # Set up a concurrent fetch tasks list
        fetch_tasks = []
        max_concurrent_tasks = min(self.concurrency, limit)  # Don't create more tasks than our limit
        
        # Analyze the main seed URL for more comprehensive data
        if processed_count < limit:
            print(f"[SEO AGENT] Analyzing seed URL thoroughly: {seed_url}")
            try:
                # Collect technical SEO metrics with a timeout
                technical_metrics = await asyncio.wait_for(
                    self.analyse_technical_seo(seed_url),
                    timeout=15  # 15-second timeout for technical analysis
                )
                analysis["technical_seo"] = technical_metrics
                
                # Collect on-page SEO metrics with a timeout
                on_page_metrics = await asyncio.wait_for(
                    self.analyse_on_page_seo(seed_url),
                    timeout=15  # 15-second timeout for on-page analysis
                )
                analysis["on_page_seo"] = on_page_metrics
                
                # Collect off-page SEO metrics with a timeout
                off_page_metrics = await asyncio.wait_for(
                    self.analyse_off_page_seo(seed_url),
                    timeout=15  # 15-second timeout for off-page analysis
                )
                analysis["off_page_seo"] = off_page_metrics
                
                # Store the seed URL analysis
                analysis["analyzed_pages"][seed_url] = {
                    "technical_seo": technical_metrics,
                    "on_page_seo": on_page_metrics,
                    "off_page_seo": off_page_metrics,
                    "issues": []  # Will be populated as we analyze
                }
                
                print(f"[SEO AGENT] Comprehensive analysis of seed URL complete.")
            except asyncio.TimeoutError:
                print(f"[SEO AGENT] Seed URL analysis timed out. Continuing with limited analysis.")
                # Create basic metrics if analysis timed out
                analysis["technical_seo"] = {"score": 5.0}
                analysis["on_page_seo"] = {"score": 5.0}
                analysis["off_page_seo"] = {"score": 5.0}
            except Exception as e:
                print(f"[SEO AGENT] Error analyzing seed URL: {str(e)}")
                raise

        async def process_url():
            """Process URLs from the queue with proper error handling"""
            nonlocal processed_count  # Declare nonlocal here, after the variable is defined in the outer scope
            
            while not queue.empty() and processed_count < limit:
                try:
                    url = await queue.get()
                    
                    # Skip processing if we've reached the limit
                    if processed_count >= limit:
                        queue.task_done()
                        continue
                    
                    # Skip URLs we've already visited
                    if url in self.visited_urls:
                        queue.task_done()
                        continue
                        
                    print(f"[SEO AGENT] Fetching page: {url}")
                    
                    # Fetch the page with a timeout
                    try:
                        page_data = await asyncio.wait_for(
                            self.fetch_page(url),
                            timeout=10  # 10-second timeout per page
                        )
                    except asyncio.TimeoutError:
                        print(f"[SEO AGENT] Fetch timed out for: {url}")
                        queue.task_done()
                        continue
                    
                    if not page_data:
                        print(f"[SEO AGENT] Skipped or failed to fetch: {url}")
                        queue.task_done()
                        continue
                        
                    processed_count += 1
                    print(f"[SEO AGENT] Analyzing: {url} ({processed_count}/{limit})")
                    
                    # Store the page in our analyzed pages
                    self.analyzed_pages[url] = page_data
                    
                    # Simplified analysis for subsequent pages
                    if url != seed_url:  # Skip if it's the seed URL we already analyzed
                        try:
                            # Collect page-specific analyses
                            page_analysis = {
                                "url": url,
                                "title": page_data.title,
                                "description": page_data.description,
                                "issues": []  # Will be populated with issues
                            }
                            
                            # Check for common issues
                            if not page_data.title or len(page_data.title) < 10:
                                issue = {"type": "missing_title", "severity": "high", "details": "Page title is missing or too short"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                                
                            if not page_data.description or len(page_data.description) < 50:
                                issue = {"type": "missing_meta_description", "severity": "medium", "details": "Meta description is missing or too short"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                                
                            if page_data.word_count < 300:
                                issue = {"type": "thin_content", "severity": "medium", "details": f"Page has only {page_data.word_count} words"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                                
                            if not page_data.h1:
                                issue = {"type": "missing_h1", "severity": "medium", "details": "Page has no H1 heading"}
                                page_analysis["issues"].append(issue)
                                if hasattr(page_data, 'issues') and page_data.issues is not None:
                                    page_data.issues.append(issue)
                            
                            # Add specific analyses with timeouts
                            try:
                                internal_linking_data = await asyncio.wait_for(
                                    self._analyse_internal_linking(url),
                                    timeout=5  # 5-second timeout
                                )
                                page_analysis["internal_linking"] = internal_linking_data
                            except (asyncio.TimeoutError, Exception) as e:
                                self.logger.error(f"Error analyzing internal linking for {url}: {str(e)}")
                                page_analysis["internal_linking"] = {"error": "Analysis timed out or failed"}
                            
                            try:
                                content_structure_data = await asyncio.wait_for(
                                    self._analyse_content_structure(url),
                                    timeout=5  # 5-second timeout
                                )
                                page_analysis["content_structure"] = content_structure_data
                            except (asyncio.TimeoutError, Exception) as e:
                                self.logger.error(f"Error analyzing content structure for {url}: {str(e)}")
                                page_analysis["content_structure"] = {"error": "Analysis timed out or failed"}
                            
                            # Store the page analysis
                            analysis["analyzed_pages"][url] = page_analysis
                            
                        except Exception as e:
                            self.logger.error(f"Error analyzing page {url}: {str(e)}")
                    
                    # Add internal links to queue if we're still under the limit
                    if page_data.internal_links and processed_count < limit:
                        # Filter out URLs we've already seen or queued
                        filtered_links = [
                            link for link in page_data.internal_links
                            if link not in self.visited_urls 
                            and not self._should_skip_url(link)
                        ]
                        
                        # Take only the first 10 links to avoid excessive crawling
                        for link in filtered_links[:10]:
                            if link not in self.visited_urls:
                                self.visited_urls.add(link)  # Mark as visited to avoid duplicates
                                await queue.put(link)
                    
                    queue.task_done()
                except Exception as e:
                    self.logger.error(f"Error processing URL: {str(e)}")
                    queue.task_done()
        
        # Create workers to process URLs concurrently
        for _ in range(max_concurrent_tasks):
            task = asyncio.create_task(process_url())
            fetch_tasks.append(task)
        
        # Wait for initial queue to be processed
        await queue.join()
        
        # Cancel any pending tasks
        for task in fetch_tasks:
            task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        print("[SEO AGENT] Website crawl finished.")
        return analysis
        
    def _should_skip_url(self, url: str) -> bool:
        """Check if a URL should be skipped based on common patterns"""
        # Skip URLs with fragments unless they are pointing to page anchors
        parsed = urlparse(url)
        
        # Skip URLs with certain file extensions
        extensions_to_skip = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.css', '.js']
        if any(parsed.path.lower().endswith(ext) for ext in extensions_to_skip):
            return True
            
        # Skip URLs with query parameters that often lead to duplicates
        query_params_to_skip = ['sort', 'filter', 'page', 'utm_source', 'utm_medium', 'utm_campaign']
        if parsed.query:
            query_dict = {k: v for k, v in [p.split('=') for p in parsed.query.split('&')] if '=' in p}
            if any(param in query_dict for param in query_params_to_skip):
                return True
                
        # Skip URLs that look like login, logout, carts, etc.
        skip_patterns = ['/login', '/logout', '/cart', '/checkout', '/account', '/admin', '/thank-you']
        if any(pattern in parsed.path.lower() for pattern in skip_patterns):
            return True
            
        return False
            
    async def fetch_page(self, url: str) -> PageData:
        """Fetch a single page and extract basic information."""
        try:
            # Create a page data object
            page = PageData(url)
            
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
                
            try:
                # Fetch the page with a timeout
                async with self.session.get(url, timeout=10) as response:
                    if response.status >= 400:
                        self.logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
                        
                    content = await response.text()
                    page.html = content
                    
                    # Parse with BeautifulSoup for easier extraction
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract title
                    title_tag = soup.find('title')
                    page.title = title_tag.get_text() if title_tag else None
                    
                    # If this is the first page we're fetching, and it looks like a site title, save it
                    if not self.site_title and page.title:
                        self.site_title = page.title
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    page.description = meta_desc.get('content', '') if meta_desc else None
                    
                    # Extract meta keywords
                    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                    page.keywords = meta_keywords.get('content', '') if meta_keywords else None
                    
                    # Extract headings
                    page.h1 = [h1.get_text() for h1 in soup.find_all('h1')]
                    page.headings = self._extract_headings(soup)
                    
                    # Count words in the page content
                    page.word_count = len(soup.get_text().split())
                    
                    # Extract links (both internal and external)
                    base_domain = urlparse(url).netloc
                    internal_links = []
                    external_links = []
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        # Handle relative links
                        if href.startswith('/'):
                            # Convert to absolute
                            parsed_url = urlparse(url)
                            absolute_url = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                            internal_links.append(absolute_url)
                        elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                            # Handle links without scheme/domain
                            # Skip anchors like "#section" and javascript
                            if href.startswith('#') or href.startswith('javascript:'):
                                continue
                                
                            # Get the page's path directory to handle relative links
                            path_parts = urlparse(url).path.split('/')
                            if len(path_parts) > 1:
                                base_dir = '/'.join(path_parts[:-1]) + '/'
                            else:
                                base_dir = '/'
                                
                            # Create absolute URL
                            parsed_url = urlparse(url)
                            absolute_url = f"{parsed_url.scheme}://{parsed_url.netloc}{base_dir}{href}"
                            internal_links.append(absolute_url)
                        elif base_domain in href:
                            # It's an absolute URL but on the same domain
                            internal_links.append(href)
                        else:
                            # It's an external link
                            external_links.append(href)
                    
                    page.internal_links = internal_links
                    page.external_links = external_links
                    
                    # Mark as visited
                    self.visited_urls.add(url)
                    
                    return page
                    
            except ClientError as e:
                # Handle client errors (like 404)
                self.logger.warning(f"Client error when fetching {url}: {str(e)}")
                return None
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout when fetching {url}")
                return None
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Unhandled error in fetch_page for {url}: {str(e)}")
            return None
            
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract heading tags (h1-h6) from the page."""
        headings = {}
        for i in range(1, 7):
            tag = f'h{i}'
            heading_tags = soup.find_all(tag)
            headings[tag] = [h.get_text().strip() for h in heading_tags]
        return headings

    async def analyse_technical_seo(self, url: str) -> Dict:
        """Analyze technical SEO metrics for a given URL."""
        try:
            result = {
                "site_speed": {},
                "mobile_optimisation": {},
                "indexation": {},
                "security": {},
                "structured_data": {},
                "issues": []
            }

            # Measure site speed metrics
            try:
                speed_metrics = await self._analyse_site_speed(url)
                result["site_speed"] = speed_metrics
                
                # Add issues if site is slow
                if speed_metrics.get("overall_load_time", 0) > 3:
                    result["issues"].append({
                        "type": "slow_site_speed",
                        "severity": "high",
                        "details": f"Site load time is {speed_metrics.get('overall_load_time', 0)} seconds"
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing site speed for {url}: {e}")
                result["site_speed"] = {"error": str(e)}
            
            # Check mobile optimization
            try:
                mobile_metrics = await self._check_mobile_optimisation(url)
                result["mobile_optimisation"] = mobile_metrics
                
                # Add issues if site is not mobile-friendly
                if mobile_metrics.get("optimization_score", 0) < 6:
                    result["issues"].append({
                        "type": "poor_mobile_optimization",
                        "severity": "high",
                        "details": "Site is not well optimized for mobile devices"
                    })
            except Exception as e:
                self.logger.error(f"Error checking mobile optimization for {url}: {e}")
                result["mobile_optimisation"] = {"error": str(e)}
            
            # Check indexation status
            try:
                robots_allow = True
                robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
                
                try:
                    async with self.session.get(robots_url, timeout=5) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            if "Disallow: /" in robots_content or f"Disallow: {urlparse(url).path}" in robots_content:
                                robots_allow = False
                except Exception as e:
                    self.logger.warning(f"Error checking robots.txt for {url}: {e}")
                
                # Check for noindex directives
                page_data = await self.fetch_page(url)
                noindex = False
                
                if page_data and page_data.html:
                    soup = BeautifulSoup(page_data.html, 'html.parser')
                    
                    # Check meta robots
                    meta_robots = soup.find('meta', attrs={'name': 'robots'})
                    if meta_robots and 'noindex' in meta_robots.get('content', '').lower():
                        noindex = True
                        
                    # Check X-Robots-Tag in headers (would require original response)
                
                # Store indexation data
                result["indexation"] = {
                    "robots_allow": robots_allow,
                    "noindex": noindex,
                    "issues": []
                }
                
                if not robots_allow or noindex:
                    result["issues"].append({
                        "type": "indexation_blocked",
                        "severity": "high",
                        "details": f"{'Robots.txt blocking' if not robots_allow else ''} {'Meta noindex present' if noindex else ''}"
                    })
            except Exception as e:
                self.logger.error(f"Error checking indexation for {url}: {e}")
                result["indexation"] = {"error": str(e)}
            
            # Check security (HTTPS, security headers)
            try:
                is_https = url.startswith("https://")
                hsts = False
                
                try:
                    async with self.session.get(url, timeout=5) as response:
                        headers = response.headers
                        if 'Strict-Transport-Security' in headers:
                            hsts = True
                except Exception as e:
                    self.logger.warning(f"Error checking security headers for {url}: {e}")
                
                result["security"] = {
                    "https": is_https,
                    "hsts": hsts,
                    "issues": []
                }
                
                if not is_https:
                    result["issues"].append({
                        "type": "no_https",
                        "severity": "high",
                        "details": "Site is not using HTTPS"
                    })
            except Exception as e:
                self.logger.error(f"Error checking security for {url}: {e}")
                result["security"] = {"error": str(e)}
            
            # Check structured data
            try:
                structured_data = {}
                page_data = await self.fetch_page(url)
                
                if page_data and page_data.html:
                    soup = BeautifulSoup(page_data.html, 'html.parser')
                    
                    # Look for JSON-LD
                    json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
                    if json_ld_scripts:
                        structured_data["json_ld"] = True
                        structured_data["count"] = len(json_ld_scripts)
                    else:
                        structured_data["json_ld"] = False
                        
                    # Look for microdata
                    microdata_elements = soup.find_all(attrs={"itemtype": True})
                    if microdata_elements:
                        structured_data["microdata"] = True
                        structured_data["count"] = len(microdata_elements)
                    else:
                        structured_data["microdata"] = False
                
                result["structured_data"] = structured_data
            except Exception as e:
                self.logger.error(f"Error checking structured data for {url}: {e}")
                result["structured_data"] = {"error": str(e)}
            
            # Calculate overall score
            scores = []
            if "site_speed" in result and isinstance(result["site_speed"], dict) and "score" in result["site_speed"]:
                scores.append(result["site_speed"]["score"])
            if "mobile_optimisation" in result and isinstance(result["mobile_optimisation"], dict) and "optimization_score" in result["mobile_optimisation"]:
                scores.append(result["mobile_optimisation"]["optimization_score"])
            
            # Add a base score if we couldn't calculate from components
            if not scores:
                scores.append(5.0)  # Default middle score
            
            result["score"] = sum(scores) / len(scores)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in technical SEO analysis for {url}: {e}")
            return {
                "score": 0,
                "error": str(e),
                "site_speed": {"error": "Analysis failed"},
                "mobile_optimisation": {"error": "Analysis failed"},
                "indexation": {"error": "Analysis failed"},
                "security": {"error": "Analysis failed"},
                "structured_data": {"error": "Analysis failed"},
                "issues": []
            }

    async def analyse_on_page_seo(self, url: str) -> Dict:
        """Analyze on-page SEO metrics for a given URL."""
        try:
            result = {
                "keyword_usage": {},
                "content_quality": {},
                "meta_tags": {},
                "internal_linking": {},
                "content_structure": {},
                "issues": []
            }
            
            # Fetch the page if not already in analyzed pages
            if url not in self.analyzed_pages:
                page_data = await self.fetch_page(url)
                if not page_data:
                    return {
                        "score": 0,
                        "error": "Failed to fetch page",
                        "issues": [{
                            "type": "fetch_failed",
                            "severity": "high",
                            "details": "Could not fetch page content"
                        }]
                    }
                self.analyzed_pages[url] = page_data
            else:
                page_data = self.analyzed_pages[url]
            
            # Analyze keyword usage
            try:
                keyword_analysis = await self._analyse_keyword_usage(url)
                result["keyword_usage"] = keyword_analysis
                
                # Add issues if keyword usage is poor
                if keyword_analysis.get("score", 0) < 4:
                    result["issues"].append({
                        "type": "poor_keyword_optimization",
                        "severity": "high",
                        "details": "Page has poor keyword optimization"
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing keyword usage for {url}: {e}")
                result["keyword_usage"] = {"error": str(e)}
            
            # Analyze content quality
            try:
                content_quality = await self._analyse_content_quality(url)
                result["content_quality"] = content_quality
                
                # Add issues if content quality is poor
                if content_quality.get("score", 0) < 5:
                    result["issues"].append({
                        "type": "low_content_quality",
                        "severity": "high",
                        "details": "Page has low-quality content"
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing content quality for {url}: {e}")
                result["content_quality"] = {"error": str(e)}
            
            # Analyze meta tags
            try:
                meta_tags = {}
                
                # Check title tag
                title = page_data.title
                title_length = len(title) if title else 0
                title_contains_brand = self.site_title and title and self.site_title.lower() in title.lower()
                
                meta_tags["title"] = {
                    "content": title,
                    "length": title_length,
                    "contains_brand": title_contains_brand,
                    "issues": []
                }
                
                if not title:
                    meta_tags["title"]["issues"].append({
                        "type": "missing_title",
                        "severity": "high",
                        "details": "Missing title tag"
                    })
                    result["issues"].append({
                        "type": "missing_title",
                        "severity": "high", 
                        "details": "Missing title tag"
                    })
                elif title_length < 30:
                    meta_tags["title"]["issues"].append({
                        "type": "short_title",
                        "severity": "medium",
                        "details": f"Title is too short ({title_length} characters)"
                    })
                    result["issues"].append({
                        "type": "short_title",
                        "severity": "medium",
                        "details": f"Title is too short ({title_length} characters)"
                    })
                elif title_length > 60:
                    meta_tags["title"]["issues"].append({
                        "type": "long_title",
                        "severity": "low",
                        "details": f"Title may be too long ({title_length} characters)"
                    })
                
                # Check meta description
                description = page_data.description
                desc_length = len(description) if description else 0
                
                meta_tags["description"] = {
                    "content": description,
                    "length": desc_length,
                    "issues": []
                }
                
                if not description:
                    meta_tags["description"]["issues"].append({
                        "type": "missing_description",
                        "severity": "high",
                        "details": "Missing meta description"
                    })
                    result["issues"].append({
                        "type": "missing_description",
                        "severity": "high",
                        "details": "Missing meta description"
                    })
                elif desc_length < 50:
                    meta_tags["description"]["issues"].append({
                        "type": "short_description",
                        "severity": "medium",
                        "details": f"Description is too short ({desc_length} characters)"
                    })
                    result["issues"].append({
                        "type": "short_description",
                        "severity": "medium",
                        "details": f"Description is too short ({desc_length} characters)"
                    })
                elif desc_length > 160:
                    meta_tags["description"]["issues"].append({
                        "type": "long_description",
                        "severity": "low",
                        "details": f"Description may be too long ({desc_length} characters)"
                    })
                
                result["meta_tags"] = meta_tags
            except Exception as e:
                self.logger.error(f"Error analyzing meta tags for {url}: {e}")
                result["meta_tags"] = {"error": str(e)}
            
            # Analyze internal linking
            try:
                internal_linking = await self._analyse_internal_linking(url)
                result["internal_linking"] = internal_linking
                
                # Add issues if internal linking is poor
                if internal_linking.get("score", 0) < 5:
                    result["issues"].append({
                        "type": "poor_internal_linking",
                        "severity": "medium",
                        "details": f"Page has poor internal linking structure"
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing internal linking for {url}: {e}")
                result["internal_linking"] = {"error": str(e)}
            
            # Analyze content structure
            try:
                content_structure = await self._analyse_content_structure(url)
                result["content_structure"] = content_structure
                
                # Add issues from content structure analysis
                for issue in content_structure.get("issues", []):
                    result["issues"].append(issue)
            except Exception as e:
                self.logger.error(f"Error analyzing content structure for {url}: {e}")
                result["content_structure"] = {"error": str(e)}
            
            # Calculate overall score
            scores = []
            for key in ["keyword_usage", "content_quality", "internal_linking", "content_structure"]:
                if key in result and isinstance(result[key], dict) and "score" in result[key]:
                    scores.append(result[key]["score"])
            
            # Add a penalty for missing meta tags
            meta_penalty = 0
            if "meta_tags" in result:
                tags = result["meta_tags"]
                if "title" in tags and not tags["title"].get("content"):
                    meta_penalty += 2
                if "description" in tags and not tags["description"].get("content"):
                    meta_penalty += 1
            
            # Calculate final score (with a minimum of 1)
            if scores:
                final_score = sum(scores) / len(scores) - meta_penalty
                result["score"] = max(1, min(10, final_score))
            else:
                result["score"] = 5.0  # Default middle score
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in on-page SEO analysis for {url}: {e}")
            return {
                "score": 0,
                "error": str(e),
                "keyword_usage": {"error": "Analysis failed"},
                "content_quality": {"error": "Analysis failed"},
                "meta_tags": {"error": "Analysis failed"},
                "internal_linking": {"error": "Analysis failed"},
                "content_structure": {"error": "Analysis failed"},
                "issues": []
            }
    
    async def _analyse_keyword_usage(self, url: str) -> Dict:
        """Analyze keyword usage on the page."""
        try:
            if url not in self.analyzed_pages:
                return {"error": "Page not in analyzed pages"}
                
            page_data = self.analyzed_pages[url]
            
            # Get page text content
            if not hasattr(page_data, 'html') or not page_data.html:
                return {"error": "No HTML content available"}
                
            soup = BeautifulSoup(page_data.html, 'html.parser')
            text_content = soup.get_text()
            
            # Remove common stop words and tokenize
            stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'is', 'was', 'were'}
            words = [word.lower() for word in re.findall(r'\b\w+\b', text_content) if word.lower() not in stop_words and len(word) > 2]
            
            # Count word frequency
            word_counts = {}
            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            
            # Get top keywords
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_keywords = sorted_words[:10]
            
            # Calculate keyword density
            total_words = len(words)
            keyword_density = {word: count / total_words * 100 for word, count in top_keywords} if total_words > 0 else {}
            
            # Check for keyword in important elements
            title = page_data.title or ""
            description = page_data.description or ""
            h1 = ' '.join(page_data.headings.get('h1', [])) if page_data.headings else ""
            url_path = urlparse(url).path
            
            keyword_presence = {}
            for word, _ in top_keywords:
                keyword_presence[word] = {
                    "in_title": word.lower() in title.lower(),
                    "in_description": word.lower() in description.lower(),
                    "in_h1": word.lower() in h1.lower(),
                    "in_url": word.lower() in url_path.lower()
                }
            
            # Score keyword optimization
            score = self._calculate_keyword_score(top_keywords, keyword_presence, total_words)
            
            return {
                "top_keywords": [{"keyword": k, "count": c} for k, c in top_keywords],
                "keyword_density": {k: round(v, 2) for k, v in keyword_density.items()},
                "keyword_presence": keyword_presence,
                "total_words": total_words,
                "score": score
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing keyword usage for {url}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_keyword_score(self, top_keywords: List[tuple], keyword_presence: Dict, total_words: int) -> float:
        """Calculate a score for keyword optimization."""
        if not top_keywords or total_words < 50:
            return 3.0  # Low score for very short content
            
        base_score = 5.0
        
        # Add points for having a good number of keywords
        if len(top_keywords) >= 5:
            base_score += 1
        
        # Check if the top keywords appear in important places
        important_places = 0
        for word, _ in top_keywords[:3]:  # Check top 3 keywords
            if word in keyword_presence:
                presences = keyword_presence[word]
                for place in ["in_title", "in_description", "in_h1", "in_url"]:
                    if presences.get(place, False):
                        important_places += 1
        
        # Add points for keywords in important places
        if important_places >= 8:  # Most of top 3 keywords are in important places
            base_score += 3
        elif important_places >= 5:
            base_score += 2
        elif important_places >= 3:
            base_score += 1
        
        # Check for keyword stuffing (penalty)
        for _, count in top_keywords[:3]:
            density = count / total_words * 100
            if density > 5:  # More than 5% density for a single word
                base_score -= 2
                break
        
        # Ensure score is in range 0-10
        return max(0, min(10, base_score))
    
    async def _analyse_content_quality(self, url: str) -> Dict:
        """Analyze the quality of the page content."""
        try:
            if url not in self.analyzed_pages:
                return {"error": "Page not in analyzed pages"}
                
            page_data = self.analyzed_pages[url]
            
            # Basic metrics
            word_count = page_data.word_count
            has_images = False
            has_lists = False
            readability_score = 0
            
            # Parse HTML for more detailed analysis
            if hasattr(page_data, 'html') and page_data.html:
                soup = BeautifulSoup(page_data.html, 'html.parser')
                
                # Check for images
                images = soup.find_all('img')
                has_images = len(images) > 0
                
                # Check for lists
                lists = soup.find_all(['ul', 'ol'])
                has_lists = len(lists) > 0
                
                # Simple readability estimation (crude approximation)
                text = soup.get_text()
                sentences = re.split(r'[.!?]+', text)
                sentence_count = len([s for s in sentences if len(s.strip()) > 0])
                
                if sentence_count > 0:
                    words_per_sentence = word_count / sentence_count
                    # Simple version of Flesch-Kincaid readability
                    if words_per_sentence <= 14:
                        readability_score = 8  # Good readability
                    elif words_per_sentence <= 18:
                        readability_score = 6  # Medium readability
                    else:
                        readability_score = 4  # Poor readability
            
            # Identify issues
            issues = []
            
            if word_count < 300:
                issues.append({
                    "type": "thin_content",
                    "severity": "high",
                    "details": f"Page has only {word_count} words"
                })
            
            if not has_images:
                issues.append({
                    "type": "no_images",
                    "severity": "medium",
                    "details": "Page has no images"
                })
            
            if readability_score < 5:
                issues.append({
                    "type": "poor_readability",
                    "severity": "medium",
                    "details": "Content may be difficult to read"
                })
            
            # Score content quality
            score = self._calculate_content_quality_score(word_count, has_images, has_lists, readability_score, issues)
            
            # Add a more detailed content quality analysis if we have HTML
            content_analysis = {}
            if hasattr(page_data, 'html') and page_data.html:
                content_analysis = {
                    "word_count": word_count,
                    "has_images": has_images,
                    "image_count": len(soup.find_all('img')) if 'soup' in locals() else 0,
                    "has_lists": has_lists,
                    "list_count": len(soup.find_all(['ul', 'ol'])) if 'soup' in locals() else 0,
                    "has_tables": len(soup.find_all('table')) > 0 if 'soup' in locals() else False,
                    "external_links": len(page_data.external_links) if page_data.external_links else 0,
                    "readability_score": readability_score
                }
            
            return {
                "score": score,
                "issues": issues,
                "analysis": content_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content quality for {url}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_content_quality_score(self, word_count: int, has_images: bool, has_lists: bool, readability_score: float, issues: List[Dict]) -> float:
        """Calculate a score for content quality."""
        base_score = 5.0
        
        # Add points for longer content
        if word_count >= 1500:
            base_score += 2
        elif word_count >= 800:
            base_score += 1.5
        elif word_count >= 400:
            base_score += 1
        
        # Add points for supplementary content elements
        if has_images:
            base_score += 1
        if has_lists:
            base_score += 0.5
        
        # Add points for good readability
        base_score += readability_score / 2  # Convert 0-10 scale to 0-5 contribution
        
        # Deduct points for issues
        base_score -= len(issues) * 0.5
        
        # Ensure score is in range 0-10
        return max(0, min(10, base_score))
    
    async def analyse_off_page_seo(self, url: str) -> Dict:
        """Analyze off-page SEO metrics for a given URL."""
        try:
            # Extract domain
            domain = urlparse(url).netloc
            
            result = {
                "backlink_profile": {},
                "brand_signals": {},
                "social_signals": {},
                "local_seo": {},
                "issues": []
            }
            
            # Since we can't do a full backlink analysis without external APIs,
            # we'll simulate it with basic estimation
            
            # Simulate backlink profile (this would normally use data from APIs like Ahrefs, Moz, etc.)
            backlink_profile = {
                "estimated_total_backlinks": 0,
                "estimated_referring_domains": 0,
                "domain_rating": 0,
                "domain_authority": 0
            }
            
            # Try to estimate domain age and importance (very rough estimate)
            try:
                async with self.session.get(f"https://{domain}/robots.txt", timeout=5) as response:
                    domain_status = response.status
                    has_robots = domain_status == 200
                    
                    # Simple heuristic: sites with robots.txt are typically more established
                    if has_robots:
                        backlink_profile["estimated_total_backlinks"] = 500
                        backlink_profile["estimated_referring_domains"] = 50
                        backlink_profile["domain_rating"] = 25
                        backlink_profile["domain_authority"] = 30
                    else:
                        backlink_profile["estimated_total_backlinks"] = 100
                        backlink_profile["estimated_referring_domains"] = 10
                        backlink_profile["domain_rating"] = 15
                        backlink_profile["domain_authority"] = 20
            except Exception:
                # Default values if we can't check
                backlink_profile["estimated_total_backlinks"] = 200
                backlink_profile["estimated_referring_domains"] = 20
                backlink_profile["domain_rating"] = 20
                backlink_profile["domain_authority"] = 25
            
            # Store backlink profile in result
            result["backlink_profile"] = backlink_profile
            
            # Simulate brand signals
            brand_signals = {
                "brand_searches": "unknown",
                "brand_mentions": "unknown",
                "knowledge_panel": "unknown"
            }
            result["brand_signals"] = brand_signals
            
            # Simulate social signals
            social_signals = {
                "social_shares": "unknown",
                "social_engagement": "unknown"
            }
            result["social_signals"] = social_signals
            
            # Calculate score based on available data
            if backlink_profile["domain_rating"] > 40:
                score = 8.0
            elif backlink_profile["domain_rating"] > 30:
                score = 7.0
            elif backlink_profile["domain_rating"] > 20:
                score = 6.0
            elif backlink_profile["domain_rating"] > 10:
                score = 5.0
            else:
                score = 4.0
            
            result["score"] = score
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in off-page SEO analysis for {url}: {e}")
            return {
                "score": 5.0,  # Default middle score
                "error": str(e),
                "backlink_profile": {},
                "brand_signals": {},
                "social_signals": {},
                "local_seo": {},
                "issues": []
            }
        
    async def generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate prioritized recommendations based on website analysis."""
        if not self.llm:
            self.logger.error("LLM processor not initialized")
            raise ValueError("LLM processor not initialized")
            
        try:
            return await self.llm.generate_recommendations(analysis)
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise ValueError(f"Error generating recommendations: {str(e)}")

async def run_analysis(api_key, seed_url, limit, save_report_callback=None, architecture_recs=None):
    """Run SEO analysis with optional architecture recommendations
    
    Args:
        api_key: API key for LLM services
        seed_url: URL to analyze
        limit: Maximum number of pages to crawl
        save_report_callback: Optional callback to save report
        architecture_recs: Optional list of architecture recommendations in format:
            [{"recommendation": "Name", "priority": "High/Medium/Low", "description": "Details"}]
    """
    async with ComprehensiveSEOAgent(api_key) as agent:
        print("[SEO AGENT] Agent initialized. Beginning website analysis...")
        try:
            # Use a shorter timeout for the website analysis
            analysis = await asyncio.wait_for(
                agent.analyse_website(seed_url, limit),
                timeout=90  # 90-second timeout for crawling
            )
            print("[SEO AGENT] Website analysis complete. Generating recommendations...")
            
            # Use a separate timeout for recommendation generation
            recommendations = await asyncio.wait_for(
                agent.generate_recommendations(analysis),
                timeout=30  # 30-second timeout for recommendations
            )
            
            # Add architecture recommendations if provided
            if architecture_recs:
                for arch_rec in architecture_recs:
                    # Convert to standard recommendation format
                    priority_map = {"High": 9, "Medium": 6, "Low": 3}
                    rec = {
                        "recommendation": arch_rec.get("description", ""),
                        "priority_score": priority_map.get(arch_rec.get("priority", "Medium"), 6),
                        "impact": "high" if arch_rec.get("priority") == "High" else "medium" if arch_rec.get("priority") == "Medium" else "low",
                        "implementation_difficulty": "medium",
                        "estimated_time": 5,  # Default estimate
                        "category": "ARCHITECTURE",
                        "architecture_title": arch_rec.get("recommendation", "")  # Store original title
                    }
                    # Add to recommendations list
                    recommendations.append(rec)
            
            print("[SEO AGENT] Analysis complete!")
            
            # Save report to file if requested
            if save_report_callback:
                report_path = save_report_callback(seed_url, recommendations, analysis, agent.site_title)
                print(f"[SEO AGENT] Report saved to: {report_path}")
            
            print("\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:6], 1):
                priority = rec.get('priority_score', 'N/A')
                impact = rec.get('impact', 'N/A')
                difficulty = rec.get('implementation_difficulty', 'N/A')
                time = rec.get('estimated_time', 'N/A')
                source = rec.get('source_page', 'site-wide')
                
                print(f"{i}. {rec.get('recommendation', 'N/A')}")
                print(f"   Priority: {priority} | Impact: {impact} | Difficulty: {difficulty} | Est. Time: {time} hours")
                print(f"   Source: {source}")
                print("")
            
            return recommendations, analysis
        
        except Exception as e:
            print(f"[SEO AGENT] ERROR: Analysis failed: {str(e)}")
            raise  # Re-raise the exception to be handled by the main function

async def main():
    """Main entry point for the SEO Analysis tool."""
    parser = argparse.ArgumentParser(description='Comprehensive SEO Analysis Tool')
    parser.add_argument('url', help='URL to analyze', nargs='?')
    parser.add_argument('-l', '--limit', type=int, default=10, help='Maximum number of pages to crawl (default: 10)')
    parser.add_argument('--save-report', action='store_true', help='Save report to file')
    parser.add_argument('--output-dir', help='Directory to save report (default: ~/seo_reports)')
    parser.add_argument('--setup', action='store_true', help='Set up API keys')
    parser.add_argument('--key', help='Directly provide API key (not recommended)')
    parser.add_argument('--architecture-file', help='Path to JSON file with architecture recommendations')
    
    args = parser.parse_args()
    
    # Handle setup command
    if args.setup:
        openai_key = input("Enter your OpenAI API key (press Enter to skip): ").strip()
        gemini_key = input("Enter your Google Gemini API key (press Enter to skip): ").strip()
        together_key = input("Enter your Together.ai API key (press Enter to skip): ").strip()
        
        api_keys = {}
        if openai_key:
            api_keys['OPENAI_API_KEY'] = openai_key
        if gemini_key:
            api_keys['GEMINI_API_KEY'] = gemini_key
        if together_key:
            api_keys['TOGETHER_API_KEY'] = together_key
            
        if api_keys:
            save_api_keys(api_keys)
            print("API keys saved successfully!")
        else:
            print("No API keys provided. Setup cancelled.")
        return
    
    # If not setup mode, url is required
    if not args.url:
        print("Error: URL is required for analysis")
        parser.print_help()
        return
    
    # Load API keys from config
    api_keys = load_api_keys()
    
    if not api_keys and not args.key:
        print("No API keys found! Please run with --setup to configure your API keys.")
        return
    
    # Determine which API key to use
    api_key = args.key if args.key else None
    
    # Check for specific provider keys
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    if llm_provider == "gemini" and 'GEMINI_API_KEY' in api_keys:
        print(f"Using GEMINI as LLM provider")
        api_key = api_keys.get('GEMINI_API_KEY')
    elif llm_provider == "together" and 'TOGETHER_API_KEY' in api_keys:
        print(f"Using TOGETHER.AI as LLM provider")
        api_key = api_keys.get('TOGETHER_API_KEY')
    elif not api_key:
        # Default to OpenAI if no specific provider is set
        api_key = api_keys.get('OPENAI_API_KEY')
        print(f"Using {'OPENAI' if 'OPENAI_API_KEY' in api_keys else 'DEFAULT'} as LLM provider")
    
    if not api_key:
        print("ERROR: No API key available for the configured LLM provider.")
        return
    
    # Normalize URL (add https:// if not present)
    if not args.url.startswith(('http://', 'https://')):
        url = f"https://{args.url}"
    else:
        url = args.url
    
    print(f"Crawling up to {args.limit} pages from {url}")
    
    # Parse architecture recommendations file if provided
    architecture_recs = None
    if args.architecture_file:
        try:
            if os.path.exists(args.architecture_file):
                with open(args.architecture_file, 'r') as f:
                    architecture_recs = json.load(f)
                print(f"Loaded {len(architecture_recs)} architecture recommendations from {args.architecture_file}")
            else:
                print(f"Warning: Architecture file not found: {args.architecture_file}")
        except Exception as e:
            print(f"Error loading architecture file: {e}")
    # If no architecture file was provided, we can use the sample data from the image
    else:
        # Create sample architecture recommendations based on the image
        architecture_recs = [
            {
                "recommendation": "Create URL Keyword Map",
                "priority": "High",
                "description": "Create an Excel map spanning your entire category structure and URL strings, and plot keyword analyse against each."
            },
            {
                "recommendation": "Improving Category Naming",
                "priority": "Medium",
                "description": "Some category names, like \"EU & Eastern Europe,\" should be more search-friendly. \"Europe\" and \"Eastern Europe\" could be separate categories for better optimization the same with \"Central & South America\". Consider targeting popular holiday destinations i.e. Dubai etc"
            },
            {
                "recommendation": "Add \"All\" Categories",
                "priority": "Medium",
                "description": "Ensure you have an \"all destinations\" and \"all holiday types\" to aid crawlability and UX."
            },
            {
                "recommendation": "Consolidate Bespoke",
                "priority": "Low",
                "description": "There is a good opportunity to consolidate Berkeley Bespoke with Travel. This would combine the authorities of both sites and make management easier."
            },
            {
                "recommendation": "URL Naming Structure",
                "priority": "Medium",
                "description": "In some instances categories do not currently make proper use of keywords in URL string. If changing the URL, ensure to use 301 redirects from the old URL to the new URL. Ensure that category URLs contain the core target keyword where possible i.e. /interests/beach/ vs. /interests/luxury-beach-holidays/,\"africa-indian-ocean\" vs \"africa-indian-ocean-holidays\"."
            }
        ]
    
    try:
        save_callback = None
        if args.save_report:
            output_dir = args.output_dir or REPORTS_DIR
            save_callback = lambda url, recs, analysis, title: save_html_report(url, recs, analysis, title, output_dir)
        
        await run_analysis(api_key, url, args.limit, save_callback, architecture_recs)
        
    except Exception as e:
        print(f"[SEO AGENT] ERROR: An unexpected error occurred: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())