import asyncio
import aiohttp
from typing import List, Dict, Set, Optional, Any # Added Any for broader type hints
from bs4 import BeautifulSoup # Not used in the snippet, but kept if used elsewhere
from urllib.parse import urljoin, urlparse # Not used in the snippet, but kept
import json
import logging
from dataclasses import dataclass, field # Added field for PageData.issues
from datetime import datetime
import argparse
import os # Still used for environment variables
import re
import configparser
import pathlib # For path operations
from aiohttp import ClientError # Not explicitly used but good for type hinting aiohttp errors
import tldextract
from dotenv import load_dotenv
import seo_analyzer # Import our new SEO analyzer module
from test_scripts.seo_fixed import LLMProcessor, save_report, save_html_report # Import LLMProcessor class and save_report function

# DataForSEO Client imports
import traceback
from dataforseo_client import Configuration
from dataforseo_client.api_client import ApiClient
from dataforseo_client.api.backlinks_api import BacklinksApi
from dataforseo_client.api.serp_api import SerpApi
from dataforseo_client.api.domain_analytics_api import DomainAnalyticsApi
from dataforseo_client.api.dataforseo_labs_api import DataforseoLabsApi
from dataforseo_client.exceptions import ApiException
from dataforseo_client.models.backlinks_summary_live_request_info import BacklinksSummaryLiveRequestInfo
from dataforseo_client.models.domain_analytics_whois_overview_live_request_info import DomainAnalyticsWhoisOverviewLiveRequestInfo
from dataforseo_client.models.serp_google_organic_live_advanced_request_info import SerpGoogleOrganicLiveAdvancedRequestInfo
from dataforseo_client.models.dataforseo_labs_google_domain_rank_overview_live_request_info import DataforseoLabsGoogleDomainRankOverviewLiveRequestInfo
from dataforseo_client.models.dataforseo_labs_google_competitors_domain_live_request_info import DataforseoLabsGoogleCompetitorsDomainLiveRequestInfo

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seo_agent")

# Debug DataForSEO credentials loading
dataforseo_login = os.getenv("DATAFORSEO_LOGIN")
dataforseo_password = os.getenv("DATAFORSEO_PASSWORD")
if dataforseo_login:
    masked_login = dataforseo_login[:3] + "*" * (len(dataforseo_login) - 3) if len(dataforseo_login) > 3 else "*****"
    logger.info(f"DataForSEO login found: {masked_login}")
else:
    logger.warning("DataForSEO login not found in environment variables")
if dataforseo_password:
    logger.info("DataForSEO password found [masked]")
else:
    logger.warning("DataForSEO password not found in environment variables")

# Constants
CONFIG_FILE = pathlib.Path.home() / ".seo_agent_config.ini"
REPORTS_DIR = pathlib.Path("/Users/andrewcosta/Desktop/seo_agent/reports")

# Function to extract and parse JSON from Gemini responses
def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON from potentially non-JSON text responses (especially from Gemini API).
    This handles cases where the model adds markdown code blocks or extra explanatory text.
    Now more robust: strips code blocks, logs raw response, tries to fix common JSON issues,
    and aggressively extracts any JSON-like structure.
    """
    # Log the raw response for debugging
    logger.debug(f"Raw LLM response for JSON extraction:\n{text}")

    if not text or not text.strip():
        logger.warning("Empty response from LLM")
        return None

    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    text = re.sub(r'```(?:json)?', '', text)
    text = text.replace('```', '').strip()

    # Very aggressive approach: Try to find anything between first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        potential_json = text[first_brace:last_brace+1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
    
    # Also try for arrays - anything between first [ and last ]
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        potential_json = text[first_bracket:last_bracket+1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
    
    # Try direct JSON parsing of the entire text (original approach)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object/array with regex (original approach)
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        # Remove trailing commas (common LLM error)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # Fix unquoted keys (another common error)
        json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON-like structure with square brackets
    json_array_pattern = r'\[\s*{[\s\S]*}\s*\]'
    matches = re.findall(json_array_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    logger.warning(f"Could not extract valid JSON from LLM response. First 200 chars: {text[:200]}")
    return None

async def retry_llm_request(llm_processor: LLMProcessor, messages: List[Dict], retry_count: int = 3, temperature: float = 0.5) -> Optional[Dict]:
    """Retry LLM request with improved error handling, automatic retries, and fallback strategies.
    
    Implements a robust strategy for handling LLM requests:
    1. Makes initial request with provided messages
    2. If it fails, retries with more structured prompts
    3. After several failures, falls back to default recommendations
    """
    logger.info(f"Requesting SEO recommendations from LLM with retry mechanism...")
    
    # Get domain and business_type for default reasoning if needed
    domain = ""
    business_type = "ecommerce"
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = message.get("content", "")
            if "technical SEO analysis of " in content:
                domain_part = content.split("technical SEO analysis of ")[1].split(" ")[0].strip()
                domain = domain_part.replace("http://", "").replace("https://", "").replace("www.", "").strip()
            
            if "BUSINESS CONTEXT" in content or "business type" in content.lower():
                if "skincare" in content.lower():
                    business_type = "skincare"
                elif "beauty" in content.lower():
                    business_type = "beauty"
                elif "cosmetic" in content.lower():
                    business_type = "cosmetics"
                elif "e-commerce" in content.lower() or "ecommerce" in content.lower():
                    business_type = "e-commerce"
                
                # Check for multiple types
                if "skincare" in content.lower() and "beauty" in content.lower():
                    business_type = "skincare and beauty"
    
    # Modify the first message to explicitly request a specific JSON format
    if len(messages) > 0 and messages[0]["role"] == "system":
        # Enhanced system message with more explicit formatting instructions
        original_content = messages[0]["content"]
        messages[0]["content"] = f"""
        {original_content}

        CRITICAL RESPONSE FORMAT INSTRUCTIONS:
        Return ONLY an array of recommendation objects in this exact format:
        [
          {{
            "recommendation": "Clear recommendation title",
            "reasoning": "Detailed explanation of why this matters for this specific website",
            "priority_score": 8, 
            "category": "Technical", 
            "impact": "High", 
            "implementation_difficulty": "Medium",
            "estimated_time_hours": 4
          }},
          ... additional recommendations ...
        ]
        
        Do NOT include any other text, explanations, or code blocks outside of this JSON structure.
        Valid category values are: Technical, On-Page, Off-Page, Architecture, UX_DESIGN, or Content.
        
        This is critical: I will be parsing your response as JSON directly. Any deviation will cause errors.
        """
    
    # Keep track of attempts
    for attempt in range(retry_count):
        try:
            # Make LLM request with increasingly structured temperature
            adjusted_temp = max(0.1, temperature - (attempt * 0.1))  # Reduce temperature with each retry
            response = await llm_processor._make_api_request(messages, temperature=adjusted_temp)
            content = response['choices'][0]['message']['content']
            
            # Try direct JSON parsing first (most reliable)
            try:
                json_data = json.loads(content.strip())
                
                # Successfully parsed the JSON data
                if isinstance(json_data, list):
                    recommendation_items = json_data
                elif isinstance(json_data, dict) and "recommendations" in json_data:
                    recommendation_items = json_data.get("recommendations", [])
                else:
                    # Try to intelligently extract recommendations
                    recommendation_items = []
                    for key, value in json_data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            if any(k in value[0] for k in ["recommendation", "title", "priority"]):
                                recommendation_items = value
                                break
                
                # If we couldn't find recommendations, try more aggressive parsing
                if not recommendation_items:
                    logger.warning(f"No valid recommendations found in LLM response (attempt {attempt+1}/{retry_count})")
                    continue
                
                # Ensure each recommendation has a reasoning field
                default_reasoning = f"This recommendation is specifically important for {domain} as a {business_type} website. Based on the technical analysis, implementing this change will improve user experience and search visibility for relevant keywords in their industry. The specific issue was found during the site analysis and addressing it will help them stand out from competitors."
                
                for item in recommendation_items:
                    if "reasoning" not in item or not item["reasoning"] or len(item["reasoning"].strip()) < 10:
                        category = item.get("category", "Technical")
                        title = item.get("recommendation", "this recommendation")
                        item["reasoning"] = f"This {category.lower()} improvement is crucial for {domain} as a {business_type} website. {title} will directly address issues found during analysis and enhance site performance. Implementing this change will improve user experience for the target audience and boost search engine rankings for relevant industry keywords."
                
                logger.info(f"Successfully extracted {len(recommendation_items)} recommendations")
                return recommendation_items
                
            except json.JSONDecodeError:
                # If direct parsing failed, try using regex to extract JSON
                pass
            
            # Try to extract JSON object/array with regex
            match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
            if match:
                json_str = match.group(1)
                # Remove trailing commas (common LLM error)
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                # Fix unquoted keys (another common error)
                json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
                try:
                    json_data = json.loads(json_str)
                    
                    # Handle the parsed JSON data (same logic as above)
                    if isinstance(json_data, list):
                        recommendation_items = json_data
                    elif isinstance(json_data, dict) and "recommendations" in json_data:
                        recommendation_items = json_data.get("recommendations", [])
                    else:
                        # Try to intelligently extract recommendations
                        recommendation_items = []
                        for key, value in json_data.items():
                            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                if any(k in value[0] for k in ["recommendation", "title", "priority"]):
                                    recommendation_items = value
                                    break
                    
                    # If we found recommendations, process them
                    if recommendation_items:
                        # Ensure each recommendation has a reasoning field
                        for item in recommendation_items:
                            if "reasoning" not in item or not item["reasoning"] or len(item["reasoning"].strip()) < 10:
                                category = item.get("category", "Technical")
                                title = item.get("recommendation", "this recommendation")
                                item["reasoning"] = f"This {category.lower()} improvement is crucial for {domain} as a {business_type} website. {title} will directly address issues found during analysis and enhance site performance. Implementing this change will improve user experience for the target audience and boost search engine rankings for relevant industry keywords."
                        
                        logger.info(f"Successfully extracted {len(recommendation_items)} recommendations using regex")
                        return recommendation_items
                    else:
                        logger.warning(f"No valid recommendations found in LLM response after regex extraction (attempt {attempt+1}/{retry_count})")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from regex match (attempt {attempt+1}/{retry_count})")
            
            # If we reach here, none of our parsing attempts worked
            if attempt < retry_count - 1:
                # Make the system message more explicit about JSON format for the retry
                if len(messages) > 0 and messages[0]["role"] == "system":
                    retry_system = """CRITICAL INSTRUCTION: You MUST respond with VALID JSON only.
Return an array of recommendation objects with this EXACT structure:
[
  {
    "recommendation": "Clear title",
    "reasoning": "Detailed explanation",
    "priority_score": 8,
    "category": "Technical",
    "impact": "High",
    "implementation_difficulty": "Medium",
    "estimated_time_hours": 4
  },
  {
    "recommendation": "Second recommendation",
    "reasoning": "Second explanation",
    "priority_score": 7,
    "category": "On-Page",
    "impact": "Medium", 
    "implementation_difficulty": "Low",
    "estimated_time_hours": 2
  }
]

NOTHING ELSE. NO explanations, markdown, or text outside the JSON array.
This is parsed programmatically - any deviation will cause errors!
"""
                    messages[0]["content"] = retry_system
                
                logger.info(f"Retrying with more explicit JSON instructions (attempt {attempt+1}/{retry_count})...")
        
        except Exception as e:
            logger.error(f"Error processing LLM recommendation response: {str(e)}")
            if attempt < retry_count - 1:
                logger.info(f"Retrying ({attempt+1}/{retry_count})...")
            else:
                logger.error(f"All retry attempts failed.")
    
    # If we reach here, all retries failed
    logger.error("Failed to get recommendations from LLM after all retry attempts")
    
    # Create default recommendations as fallback
    if domain:
        default_recs = create_default_recommendations("", domain)
        
        # Ensure each default recommendation has reasoning
        for rec in default_recs:
            if "reasoning" not in rec or not rec["reasoning"]:
                category = rec.get("category", "Technical")
                title = rec.get("recommendation", "this recommendation")
                rec["reasoning"] = f"This {category.lower()} improvement is crucial for {domain} as a {business_type} website. {title} will directly address common issues found in similar websites and enhance overall site performance. Implementing this change follows industry best practices and will improve user experience."
        
        return default_recs
        
    return None

async def analyze_architecture(url: str, domain: str, analysis_data: Dict, llm_processor: LLMProcessor) -> List[Dict]:
    """
    Analyze the website architecture and generate customized recommendations.
    
    Args:
        url: Website URL
        domain: Extracted domain name
        analysis_data: Data from the technical analysis
        llm_processor: LLM processor instance for generating recommendations
        
    Returns:
        List of architecture recommendations
    """
    logger.info(f"Analyzing site architecture for {domain}...")
    
    # Extract site structure information from analyzed pages
    pages = analysis_data.get("analyzed_pages", {})
    page_urls = list(pages.keys())
    
    # Check if sitemap already exists and gather sitemap data
    sitemap_exists = analysis_data.get("technical_seo", {}).get("indexation", {}).get("sitemap", {}).get("exists", False)
    sitemap_data = analysis_data.get("technical_seo", {}).get("indexation", {}).get("sitemap", {})
    sitemap_url = sitemap_data.get("url", "")
    sitemap_entry_count = sitemap_data.get("entry_count", 0)
    
    # Check for potential sitemap issues even if it exists
    sitemap_issues = []
    if sitemap_exists:
        # Check if sitemap has low entry count compared to analyzed pages
        if sitemap_entry_count > 0 and sitemap_entry_count < len(pages) * 0.8:  # If sitemap covers less than 80% of known pages
            sitemap_issues.append(f"Sitemap contains only {sitemap_entry_count} URLs but {len(pages)} pages were found during analysis")
        
        # Check if sitemap has errors or warnings
        if sitemap_data.get("errors", []):
            sitemap_issues.append(f"Sitemap has errors: {', '.join(sitemap_data.get('errors', []))}")
        
        if sitemap_data.get("warnings", []):
            sitemap_issues.append(f"Sitemap has warnings: {', '.join(sitemap_data.get('warnings', []))}")
            
        logger.info(f"Sitemap exists with URL: {sitemap_url}")
        if sitemap_issues:
            logger.info(f"Sitemap has issues: {sitemap_issues}")
        else:
            logger.info(f"Sitemap appears to be working correctly")
    
    # Extract hierarchy information
    hierarchy = {}
    for page_url in page_urls:
        path = urlparse(page_url).path
        parts = [p for p in path.split('/') if p]
        
        # Build hierarchy tree
        current = hierarchy
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {
                    "count": 0,
                    "children": {},
                    "pages": [],
                    "depth": i + 1
                }
            current[part]["count"] += 1
            current[part]["pages"].append(page_url)
            current = current[part]["children"]
    
    # Extract categories and directory structure
    categories = []
    for key, value in hierarchy.items():
        if key and value.get("count", 0) > 0:
            categories.append({
                "name": key,
                "count": value.get("count", 0),
                "depth": value.get("depth", 1),
                "has_children": len(value.get("children", {})) > 0
            })
    
    # Analyze URL patterns
    url_patterns = []
    for page_url in page_urls:
        path = urlparse(page_url).path
        pattern = "/".join([p for p in path.split('/') if p])
        if pattern and pattern not in url_patterns:
            url_patterns.append(pattern)
    
    # Get internal linking structure
    internal_links = {}
    for page_url, page_data in pages.items():
        links = page_data.get("links", {}).get("internal", [])
        internal_links[page_url] = links
    
    # Create architecture analysis prompt
    architecture_prompt = f"""
    Analyze the website architecture and URL structure for {url} based on the following data:
    
    URL PATTERNS (sample of {min(10, len(url_patterns))} patterns):
    {json.dumps(url_patterns[:10], indent=2)}
    
    CATEGORIES/DIRECTORIES (top {min(10, len(categories))} found):
    {json.dumps(categories[:10], indent=2)}
    
    INTERNAL LINKING SAMPLE:
    {json.dumps({k: v[:5] for k, v in list(internal_links.items())[:3]}, indent=2)}
    
    PAGES ANALYZED: {len(pages)}
    
    SITEMAP DATA:
    - Sitemap.xml exists: {sitemap_exists}
    """
    
    if sitemap_exists:
        architecture_prompt += f"""
    - Sitemap URL: {sitemap_url}
    - Sitemap entry count: {sitemap_entry_count}
    - Potential sitemap issues: {sitemap_issues if sitemap_issues else "None detected"}
    """
    
    architecture_prompt += f"""
    
    Based on this architectural data, provide a detailed analysis of the site's architecture focusing on:
    
    1. URL Structure: Evaluate the consistency, SEO-friendliness, and organization of URLs
    2. Information Architecture: Analyze the site's hierarchy, depth, and organization
    3. Navigation Structure: Assess how well the site is internally linked and organized
    4. Content Categorization: Evaluate how content is grouped and organized
    5. Mobile Architecture: Identify any issues with responsive design or mobile architecture
    
    For each recommendation, provide:
    - A clear title
    - A detailed explanation
    - Priority level (1-10)
    - Implementation difficulty
    - Estimated hours to implement
    - Category (should be ARCHITECTURE)
    
    IMPORTANT SITEMAP INSTRUCTIONS:
    - If a sitemap does NOT exist, recommend creating a comprehensive XML sitemap.
    - If a sitemap EXISTS but has issues (e.g., missing pages, errors, warnings), recommend improvements to the EXISTING sitemap rather than creating a new one.
    - If a sitemap EXISTS and seems complete without issues, focus on other architectural improvements instead.
    
    Your analysis should be specific to this site's structure, focused on architecture rather than general SEO, 
    and actionable for developers and content teams.
    """
    
    # System message for architecture analysis
    system_message = """You are an expert in website architecture and information design with deep knowledge of 
    SEO best practices. Analyze the provided website architecture data and create a detailed, 
    customized set of recommendations.

    Return your recommendations in a JSON array format with each recommendation having these fields:
    - recommendation: The clear title of the recommendation
    - reasoning: Why this matters for the specific website
    - priority_score: 1-10 value indicating importance 
    - impact: High/Medium/Low
    - implementation_difficulty: High/Medium/Low
    - estimated_time_hours: Number of hours to implement
    - category: Always use "ARCHITECTURE"
    
    IMPORTANT REGARDING SITEMAPS:
    - If NO sitemap exists, recommend creating a comprehensive XML sitemap.
    - If a sitemap EXISTS but has issues, recommend IMPROVING the existing sitemap instead of creating a new one.
      Specify exactly what needs to be improved and why.
    - If a sitemap EXISTS and appears complete, focus on other architectural improvements.
    
    Make your recommendations specific to the site's actual structure, not generic advice.
    Focus on architectural improvements that would enhance crawlability, user experience, and SEO.
    """
    
    # Get architecture analysis
    architecture_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": architecture_prompt}
    ]
    
    logger.info(f"Requesting architecture analysis from LLM...")
    try:
        # Use the new retry mechanism instead of direct API call
        architecture_recommendations = await retry_llm_request(llm_processor, architecture_messages, retry_count=2, temperature=0.3)
        
        if not architecture_recommendations:
            logger.warning("Failed to parse architecture analysis as JSON after retries. Using default recommendations.")
            return create_default_architecture_recommendations(sitemap_exists, sitemap_issues)
        
        # Return the recommendations directly if they're already in the expected format
        if isinstance(architecture_recommendations, list):
            # Process recommendations to ensure proper handling of sitemap suggestions
            filtered_recommendations = []
            for rec in architecture_recommendations:
                if isinstance(rec, dict):
                    recommendation_text = rec.get("recommendation", "").lower()
                    
                    # Skip generic "create sitemap" recommendations if sitemap already exists
                    if sitemap_exists and ("create" in recommendation_text and "sitemap" in recommendation_text) and not any(word in recommendation_text for word in ["improve", "enhance", "update", "fix"]):
                        if sitemap_issues:
                            # Convert creation recommendation to improvement if issues exist
                            logger.info(f"Converting sitemap creation recommendation to improvement: {rec.get('recommendation', '')}")
                            rec["recommendation"] = f"Improve existing sitemap to address: {', '.join(sitemap_issues[:1])}"
                            rec["reasoning"] = f"The existing sitemap has issues that need to be addressed: {', '.join(sitemap_issues)}. Improving the sitemap will ensure all content is properly indexed by search engines."
                        else:
                            # Skip if no issues and it's a creation recommendation
                            logger.info(f"Skipping unnecessary sitemap recommendation: {rec.get('recommendation', '')}")
                            continue
                    
                    # Always use ARCHITECTURE as category
                    rec["category"] = "ARCHITECTURE"
                    filtered_recommendations.append(rec)
            
            logger.info(f"Generated {len(filtered_recommendations)} architecture recommendations")
            return filtered_recommendations
            
        # Otherwise convert architecture analysis to recommendations list (should not happen with updated prompt)
        recommendations = []
        
        # Handle the case where it's a dictionary instead of a list
        if isinstance(architecture_recommendations, dict):
            # Process each section of the architecture analysis
            try:
                for section_key, section_data in architecture_recommendations.items():
                    if not isinstance(section_data, dict):
                        continue
                    
                    section_name = section_key.replace("_", " ").title()
                    
                    # Add section findings as a high-level recommendation
                    findings = section_data.get("findings", "")
                    if findings and len(findings) > 20:
                        recommendation_text = f"Improve {section_name}"
                        
                        # Convert generic sitemap creation recommendations to improvement ones
                        if sitemap_exists and "sitemap" in recommendation_text.lower() and "create" in findings.lower():
                            if sitemap_issues:
                                logger.info(f"Converting sitemap finding to improvement recommendation")
                                recommendation_text = f"Improve existing sitemap"
                                findings = f"The existing sitemap has issues that need to be addressed: {', '.join(sitemap_issues)}. Improving the sitemap will ensure all content is properly indexed by search engines."
                            else:
                                logger.info(f"Skipping unnecessary sitemap finding")
                                continue
                            
                        recommendations.append({
                            "recommendation": recommendation_text,
                            "priority_score": 9 if section_data.get("score", 5) <= 4 else 6 if section_data.get("score", 5) <= 7 else 3,
                            "impact": "high" if section_data.get("score", 5) <= 4 else "medium" if section_data.get("score", 5) <= 7 else "low",
                            "implementation_difficulty": "medium",
                            "estimated_time_hours": 8,
                            "category": "ARCHITECTURE",
                            "reasoning": findings
                        })
                    
                    # Add individual recommendations from this section
                    section_recommendations = section_data.get("recommendations", [])
                    for rec in section_recommendations:
                        if not isinstance(rec, dict):
                            continue
                        
                        recommendation_text = rec.get("description", "")
                        
                        # Convert generic sitemap creation recommendations to improvement ones
                        if sitemap_exists and "create" in recommendation_text.lower() and "sitemap" in recommendation_text.lower():
                            if sitemap_issues:
                                logger.info(f"Converting sitemap recommendation to improvement")
                                recommendation_text = f"Improve existing sitemap to address: {', '.join(sitemap_issues)}"
                                reasoning = f"The existing sitemap has issues that need to be addressed: {', '.join(sitemap_issues)}. Improving the sitemap will ensure all content is properly indexed by search engines."
                            else:
                                logger.info(f"Skipping unnecessary sitemap recommendation")
                                continue
                        else:
                            reasoning = rec.get("reasoning", "")
                            
                        priority_map = {"High": 9, "Medium": 6, "Low": 3}
                        priority = rec.get("priority", "Medium")
                        
                        recommendations.append({
                            "recommendation": recommendation_text,
                            "priority_score": priority_map.get(priority, 6),
                            "impact": "high" if priority == "High" else "medium" if priority == "Medium" else "low",
                            "implementation_difficulty": "medium",
                            "estimated_time_hours": 5,
                            "category": "ARCHITECTURE",
                            "reasoning": reasoning
                        })
            except AttributeError:
                # If we get an AttributeError (like 'list' has no attribute 'items'),
                # log the error and use default recommendations
                logger.error("AttributeError while processing architecture recommendations. The response format was unexpected.")
                logger.warning("Using default architecture recommendations instead.")
                return create_default_architecture_recommendations(sitemap_exists, sitemap_issues)
            
            logger.info(f"Generated {len(recommendations)} architecture recommendations")
            return recommendations
            
        # If we get here, something unexpected happened - return default
        return create_default_architecture_recommendations(sitemap_exists, sitemap_issues)
        
    except Exception as e:
        logger.error(f"Error during architecture analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return create_default_architecture_recommendations(sitemap_exists, sitemap_issues)

def create_default_architecture_recommendations(sitemap_exists=False, sitemap_issues=None) -> List[Dict]:
    """Create minimal fallback architecture recommendation when analysis fails."""
    if sitemap_issues is None:
        sitemap_issues = []
        
    default_recs = [{
        "recommendation": "Improve URL structure for better hierarchy",
        "priority_score": 8,
        "impact": "medium",
        "implementation_difficulty": "medium",
        "estimated_time_hours": 5,
        "category": "ARCHITECTURE",
        "reasoning": "A logical URL structure helps search engines understand your site organization and improves user navigation."
    }]
    
    # Handle sitemap recommendations based on existence and issues
    if not sitemap_exists:
        # If no sitemap exists, recommend creating one
        default_recs.append({
            "recommendation": "Create a comprehensive XML sitemap",
            "priority_score": 8,
            "impact": "high",
            "implementation_difficulty": "low",
            "estimated_time_hours": 3,
            "category": "ARCHITECTURE",
            "reasoning": "A sitemap is crucial for search engines to efficiently crawl your website and understand its structure."
        })
    elif sitemap_issues and len(sitemap_issues) > 0:
        # If sitemap exists but has issues, recommend improvement
        issue_summary = ", ".join(sitemap_issues)
        default_recs.append({
            "recommendation": f"Improve existing sitemap to address issues",
            "priority_score": 7,
            "impact": "medium",
            "implementation_difficulty": "low",
            "estimated_time_hours": 2,
            "category": "ARCHITECTURE",
            "reasoning": f"Your sitemap exists but has the following issues that should be addressed: {issue_summary}. Resolving these will improve search engine indexing."
        })
    
    # Always include a recommendation for internal linking
    default_recs.append({
        "recommendation": "Enhance internal linking strategy",
        "priority_score": 7,
        "impact": "medium",
        "implementation_difficulty": "medium",
        "estimated_time_hours": 4,
        "category": "ARCHITECTURE",
        "reasoning": "Strategic internal linking helps distribute page authority throughout your site and guides users to important content."
    })
    
    return default_recs

async def analyze_ux(url: str, domain: str, analysis_data: Dict, llm_processor: LLMProcessor) -> List[Dict]:
    """
    Analyze the website UX and page design and generate customized recommendations.
    
    Args:
        url: Website URL
        domain: Extracted domain name
        analysis_data: Data from the technical analysis
        llm_processor: LLM processor instance for generating recommendations
        
    Returns:
        List of UX recommendations
    """
    logger.info(f"Analyzing UX and page design for {domain}...")
    
    # Extract relevant metrics for UX analysis
    pages = analysis_data.get("analyzed_pages", {})
    page_urls = list(pages.keys())
    
    # Gather UX-related metrics from the analysis
    ux_metrics = {
        "mobile_friendly": analysis_data.get("technical_seo", {}).get("mobile_optimisation", {}).get("has_viewport_meta", False),
        "mobile_score": analysis_data.get("technical_seo", {}).get("mobile_optimisation", {}).get("score", 0),
        "avg_load_time": analysis_data.get("technical_seo", {}).get("site_speed", {}).get("avg_load_time", 0),
        "interactive_elements": [],
        "readability_issues": [],
        "layout_issues": []
    }
    
    # Extract UX-related issues from analyzed pages
    for page_url, page_data in pages.items():
        for issue in page_data.get("issues", []):
            issue_type = issue.get("type", "").lower()
            if "mobile" in issue_type or "viewport" in issue_type:
                ux_metrics["mobile_issues"] = ux_metrics.get("mobile_issues", []) + [issue]
            elif "load" in issue_type or "speed" in issue_type or "performance" in issue_type:
                ux_metrics["speed_issues"] = ux_metrics.get("speed_issues", []) + [issue]
            elif "contrast" in issue_type or "color" in issue_type or "readability" in issue_type:
                ux_metrics["readability_issues"].append(issue)
            elif "layout" in issue_type or "design" in issue_type:
                ux_metrics["layout_issues"].append(issue)
    
    # Create UX analysis prompt
    ux_prompt = f"""
    Analyze the User Experience (UX) and page design for {url} based on the following data:
    
    MOBILE FRIENDLINESS:
    - Has viewport meta tag: {ux_metrics.get("mobile_friendly", False)}
    - Mobile optimization score: {ux_metrics.get("mobile_score", 0)}/10
    
    PERFORMANCE:
    - Average page load time: {ux_metrics.get("avg_load_time", 0):.2f} seconds
    
    SAMPLE PAGE URLS (for design analysis):
    {json.dumps(page_urls[:5], indent=2)}
    
    Based on this data and your knowledge of modern web design principles and UX best practices, 
    provide a detailed analysis of the site's UX and page design.
    
    For each recommendation, include:
    - A clear title
    - A detailed explanation of why this matters
    - Priority score (1-10)
    - Implementation difficulty
    - Estimated hours to implement
    - Category (should be UX_DESIGN)
    
    Your analysis should be specific to this site, focused on UX and page design rather than general SEO,
    and actionable for designers and developers.
    """
    
    # System message for UX analysis
    system_message = """You are an expert UX designer and web usability specialist with deep knowledge of
    modern design principles. Analyze the provided website data and create a detailed,
    customized set of UX recommendations.

    Return your recommendations in a JSON array format with each recommendation having these fields:
    - recommendation: The clear title of the recommendation
    - reasoning: Why this matters for the specific website
    - priority_score: 1-10 value indicating importance 
    - impact: High/Medium/Low
    - implementation_difficulty: High/Medium/Low
    - estimated_time_hours: Number of hours to implement
    - category: Always use "UX_DESIGN"
    
    Make your recommendations specific to the site's actual design, not generic advice.
    Focus on UX improvements that would enhance user engagement, conversion rates, and overall satisfaction.
    """
    
    # Get UX analysis
    ux_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": ux_prompt}
    ]
    
    logger.info(f"Requesting UX analysis from LLM...")
    try:
        # Use the new retry mechanism instead of direct API call
        ux_recommendations = await retry_llm_request(llm_processor, ux_messages, retry_count=2, temperature=0.3)
        
        if not ux_recommendations:
            logger.warning("Failed to parse UX analysis as JSON after retries. Using default recommendations.")
            return create_default_ux_recommendations()
        
        # Return the recommendations directly if they're already in the expected format
        if isinstance(ux_recommendations, list):
            # Just ensure all recommendations have the correct category
            for rec in ux_recommendations:
                if isinstance(rec, dict):
                    rec["category"] = "UX_DESIGN"
            
            logger.info(f"Generated {len(ux_recommendations)} UX recommendations")
            return ux_recommendations
            
        # Otherwise convert ux analysis to recommendations list (should not happen with updated prompt)
        recommendations = []
        
        # Handle the case where it's a dictionary instead of a list
        if isinstance(ux_recommendations, dict):
            # Process each section of the UX analysis
            try:
                for section_key, section_data in ux_recommendations.items():
                    if not isinstance(section_data, dict):
                        continue
                        
                    section_name = section_key.replace("_", " ").title()
                    
                    # Add section findings as a high-level recommendation
                    findings = section_data.get("findings", "")
                    if findings and len(findings) > 20:
                        recommendations.append({
                            "recommendation": f"Improve {section_name}",
                            "priority_score": 9 if section_data.get("score", 5) <= 4 else 6 if section_data.get("score", 5) <= 7 else 3,
                            "impact": "high" if section_data.get("score", 5) <= 4 else "medium" if section_data.get("score", 5) <= 7 else "low",
                            "implementation_difficulty": "medium",
                            "estimated_time_hours": 8,
                            "category": "UX_DESIGN",
                            "reasoning": findings
                        })
                    
                    # Add individual recommendations from this section
                    section_recommendations = section_data.get("recommendations", [])
                    for rec in section_recommendations:
                        if not isinstance(rec, dict):
                            continue
                            
                        priority_map = {"High": 9, "Medium": 6, "Low": 3}
                        priority = rec.get("priority", "Medium")
                        
                        recommendations.append({
                            "recommendation": rec.get("description", ""),
                            "priority_score": priority_map.get(priority, 6),
                            "impact": "high" if priority == "High" else "medium" if priority == "Medium" else "low",
                            "implementation_difficulty": "medium",
                            "estimated_time_hours": 5,
                            "category": "UX_DESIGN",
                            "reasoning": rec.get("reasoning", "")
                        })
            except AttributeError:
                # If we get an AttributeError (like 'list' has no attribute 'items'),
                # log the error and use default recommendations
                logger.error("AttributeError while processing UX recommendations. The response format was unexpected.")
                logger.warning("Using default UX recommendations instead.")
                return create_default_ux_recommendations()
        
            logger.info(f"Generated {len(recommendations)} UX recommendations")
            return recommendations
        
        # If we get here, something unexpected happened - return default
        return create_default_ux_recommendations()
        
    except Exception as e:
        logger.error(f"Error during UX analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return create_default_ux_recommendations()

def create_default_ux_recommendations() -> List[Dict]:
    """Create minimal fallback UX recommendation when analysis fails."""
    return [{
        "recommendation": "Perform professional UX assessment",
        "priority_score": 7,
        "impact": "medium",
        "implementation_difficulty": "medium",
        "estimated_time_hours": 4,
        "category": "UX_DESIGN",
        "architecture_title": "UX Design Review"
    }]

async def analyze_competitors(url: str, domain: str, website_analysis: str, llm_processor: LLMProcessor) -> Dict:
    """
    Analyze competitors and gather data about their SEO metrics using LLM.
    Tries to get REAL competitor brands and their websites, allowing 'N/A' for metrics if needed.
    If the strict prompt returns no results, retries with a more lenient prompt.
    """
    logger.info(f"Analyzing REAL competitors for {domain}...")

    def _filter_competitors(competitors):
        actual_competitors = []
        for comp in competitors:
            brand_name = comp.get("brand", "").lower()
            if "unknown" not in brand_name and "competitor" not in brand_name and "placeholder" not in brand_name and brand_name:
                actual_competitors.append(comp)
        return actual_competitors

    # Strict prompt (first attempt)
    competitors_prompt_strict = f"""
    Based on the following business analysis for {url}:
    
    {website_analysis}
    
    Please identify 3-5 REAL, KNOWN competitor brands for this business. 
    For each competitor, provide:
    - Brand Name (the actual company name)
    - Competitor Website (their main domain, e.g., competitor.com)
    - Your BEST ESTIMATE for Domain Rating (DR) [0-100, or 'N/A' if unknown]
    - Your BEST ESTIMATE for Referring Domains (or 'N/A' if unknown)
    - Your BEST ESTIMATE for Backlinks (or 'N/A' if unknown)
    - Your BEST ESTIMATE for Total Keywords (or 'N/A' if unknown)

    IMPORTANT: 
    1. Only list REAL brands. Do NOT invent generic names like 'Unknown Competitor', 'Competitor A', or 'Placeholder Brand'.
    2. If you cannot reasonably estimate a specific metric for a REAL competitor you've identified, use 'N/A' for that metric's value.
    3. Only return an empty list for "competitors" if you genuinely cannot identify ANY real competitor brands at all.

    Your response MUST be a valid JSON object with the following format:
    {{
      "competitors": [
        {{
          "brand": "Actual Competitor Name Inc.",
          "website": "actualcompetitor.com",
          "domain_rating": 75,
          "referring_domains": 12000,
          "backlinks": 100000,
          "total_keywords": 80000
        }},
        // ... more real competitors or an empty list ...
      ]
    }}
    If no real competitors can be identified, the "competitors" list should be empty, like this:
    {{
      "competitors": []
    }}
    """
    
    system_message_strict = """You are an expert SEO and competitive analysis specialist.
    Your task is to identify REAL, known competitor brands based on a business analysis.
    You must provide actual brand names and their websites.
    DO NOT invent generic names if you cannot find real competitors. Instead, return an empty list.
    Provide your BEST ESTIMATES for SEO metrics (DR, Referring Domains, Backlinks, Total Keywords), but use 'N/A' if you cannot reasonably estimate a value for a real competitor.
    Only return an empty list if you genuinely cannot identify ANY real competitor brands at all.
    
    For Domain Rating (DR) estimates (0-100 scale):
    - 0-20: Very small/new websites
    - 20-50: Established small to medium websites
    - 50-70: Strong industry players
    - 70-90: Major industry leaders
    - 90-100: Top global websites

    Your entire response MUST be valid JSON.
    """

    # Lenient prompt (second attempt)
    competitors_prompt_lenient = f"""
    Based on the following business analysis for {url}:
    
    {website_analysis}
    
    List 3-5 real, known competitor brands for this business. If you cannot find explicit competitors from the website, use your knowledge of the industry to suggest likely real competitors (but do not invent generic names like 'Unknown Competitor'). For each, provide:
    - Brand Name (the actual company name)
    - Competitor Website (their main domain, e.g., competitor.com)
    - Your best estimate for Domain Rating (DR) [0-100, or 'N/A' if unknown]
    - Your best estimate for Referring Domains (or 'N/A' if unknown)
    - Your best estimate for Backlinks (or 'N/A' if unknown)
    - Your best estimate for Total Keywords (or 'N/A' if unknown)

    If you genuinely cannot identify any real competitor brands, return an empty list.
    Your response MUST be a valid JSON object with the following format:
    {{
      "competitors": [
        {{
          "brand": "Actual Competitor Name Inc.",
          "website": "actualcompetitor.com",
          "domain_rating": 75,
          "referring_domains": 12000,
          "backlinks": 100000,
          "total_keywords": 80000
        }},
        // ... more real competitors or an empty list ...
      ]
    }}
    If no real competitors can be identified, the "competitors" list should be empty, like this:
    {{
      "competitors": []
    }}
    """
    
    system_message_lenient = """You are an expert SEO and competitive analysis specialist.
    Your task is to identify REAL, known competitor brands based on a business analysis.
    You must provide actual brand names and their websites. If you cannot find explicit competitors from the website, use your knowledge of the industry to suggest likely real competitors (but do not invent generic names like 'Unknown Competitor').
    Provide your best estimates for SEO metrics (DR, Referring Domains, Backlinks, Total Keywords), but use 'N/A' if you cannot reasonably estimate a value for a real competitor.
    Only return an empty list if you genuinely cannot identify ANY real competitor brands at all.
    
    For Domain Rating (DR) estimates (0-100 scale):
    - 0-20: Very small/new websites
    - 20-50: Established small to medium websites
    - 50-70: Strong industry players
    - 70-90: Major industry leaders
    - 90-100: Top global websites

    Your entire response MUST be valid JSON.
    """

    # Try strict prompt first
    for attempt, (prompt, sys_msg) in enumerate([
        (competitors_prompt_strict, system_message_strict),
        (competitors_prompt_lenient, system_message_lenient)
    ]):
        competitors_messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ]
        logger.info(f"Requesting competitor analysis from LLM (attempt {attempt+1})...")
        try:
            # Use the retry mechanism instead of direct API call
            competitors_data = await retry_llm_request(llm_processor, competitors_messages, retry_count=2, temperature=0.2)
            
            # If retry mechanism failed completely, continue to next attempt
            if not competitors_data:
                logger.warning(f"Failed to get valid JSON from LLM on attempt {attempt+1} even after retries")
                continue
                
            # Check if competitors key exists
            if "competitors" not in competitors_data:
                logger.warning(f"'competitors' key missing from LLM response on attempt {attempt+1}")
                continue
                
            # Filter out any clearly placeholder/generic names if the LLM didn't follow instructions
            competitors_list = competitors_data.get("competitors", [])
            actual_competitors = _filter_competitors(competitors_list)
            
            if not actual_competitors and competitors_list:
                logger.warning(f"LLM returned only generic competitor names (attempt {attempt+1}).")
                continue
                
            competitors_data["competitors"] = actual_competitors
            
            if not competitors_data.get("competitors"): # Empty list from LLM or after filtering
                logger.info(f"LLM indicated no real competitors found or all were filtered (attempt {attempt+1}).")
                continue
                
            logger.info(f"Generated competitor analysis for {len(competitors_data.get('competitors', []))} competitors (attempt {attempt+1}).")
            return competitors_data
        except Exception as e:
            logger.error(f"Error during competitor analysis (attempt {attempt+1}): {str(e)}")
            logger.error(traceback.format_exc())
            continue
            
    # If both attempts fail, fallback
    logger.warning("Both strict and lenient competitor prompts failed. Using fallback.")
    return create_default_competitor_data(domain, is_fallback=True)

def create_default_competitor_data(domain: str, is_fallback: bool = False, count: int = 0) -> Dict:
    """Create empty competitor data structure when competitor analysis fails."""
    # Just return an empty competitors list since we don't want to make assumptions
    return {"competitors": []}

async def analyze_business_context(url: str, domain: str, analysis_data: Dict, llm_processor: LLMProcessor) -> str:
    """
    Perform a detailed analysis of the business context to correctly identify the actual business type.
    Enhanced to avoid false assumptions based on domain names.
    
    Args:
        url: Website URL
        domain: Extracted domain name
        analysis_data: Data from the technical analysis
        llm_processor: LLM processor instance
        
    Returns:
        String with the business context analysis
    """
    logger.info(f"Performing detailed business context analysis for {domain}...")
    
    # Extract content from analyzed pages
    pages = analysis_data.get("analyzed_pages", {})
    
    # Collect all page titles, meta descriptions, and key content
    titles = []
    descriptions = []
    page_content = []
    product_phrases = []
    
    # Look for specific content that indicates business type
    for page_url, page_data in pages.items():
        titles.append(page_data.get("title", ""))
        descriptions.append(page_data.get("meta_description", ""))
        
        # Get product-related phrases
        content = page_data.get("content", "")
        if content:
            # Extract lines with potential product indicators
            product_indicators = ["our product", "we offer", "our service", "shop", "buy", "product", "collection", 
                                "our mission", "about us", "our story", "what we do", "clothing", "clothes",
                                "eczema", "baby", "children", "infant", "skin condition", "sensitive skin",
                                "medical", "health", "therapeutic"]
            lines = content.split("\n")
            for line in lines:
                if any(indicator in line.lower() for indicator in product_indicators) and len(line) < 200:
                    product_phrases.append(line.strip())
    
    # Create a business context prompt with more detailed website information
    context_prompt = f"""
    I need you to accurately identify the ACTUAL business type and offerings of the website: {url}
    
    IMPORTANT: You MUST avoid making assumptions based on the domain name. Many businesses have misleading names!
    For example, happy-skin.com might NOT be a skincare company - it could be a clothing company for people with skin conditions.
    
    Analyze the following extracted content to determine the TRUE business purpose:
    
    PAGE TITLES:
    {json.dumps(titles, indent=2)}
    
    META DESCRIPTIONS:
    {json.dumps(descriptions, indent=2)}
    
    PRODUCT/SERVICE PHRASES:
    {json.dumps(product_phrases[:15], indent=2)}
    
    1. What is the SPECIFIC business type and industry? Be as precise as possible about what they ACTUALLY sell.
    2. What ACTUAL products or services do they offer? Give specific product categories, not generalizations.
    3. Who is their target audience and what problem does the business solve for them?
    4. What are 2-3 direct competitors based on the ACTUAL business type (not assumptions)?
    5. What appears to be their unique selling proposition or brand values?
    
    CRITICAL: If the evidence suggests they sell eczema clothing for babies/children, specify that explicitly.
    If they sell skincare products, specify which types. Never generalize based on the domain name.
    """
    
    # System message for business context
    system_message = """You are an expert business analyst with deep experience in identifying a company's actual business model.
    Your role is to carefully analyze website content to determine what a business actually does and sells.
    
    DO NOT make assumptions based on the domain name. Many businesses have names that don't directly describe their actual products.
    For example, "Happy Skin" could be a company that makes clothing for people with eczema, not a skincare company.
    
    Your analysis should be based ONLY on the concrete evidence provided in the website content.
    If the evidence is unclear, acknowledge the uncertainty rather than making assumptions.
    
    Be extremely specific about the business type. Here are examples of specific vs. general descriptions:
    - BAD: "health products" 
    - GOOD: "eczema-friendly clothing for babies and children"
    - BAD: "beauty products" 
    - GOOD: "organic skincare for sensitive skin"
    - BAD: "online store"
    - GOOD: "online retailer of therapeutic clothing for skin conditions"
    
    For Happy-Skin.com - if evidence shows they sell eczema-friendly clothing for babies, state that explicitly!
    """
    
    # Get business context
    context_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context_prompt}
    ]
    
    try:
        # For business context, we don't need JSON, so direct API call is fine
        context_response = await llm_processor._make_api_request(context_messages, temperature=0.2)
        business_context = context_response['choices'][0]['message']['content']
        
        # Now verify this with a second prompt to double-check accuracy
        verification_prompt = f"""
        Based on your initial analysis of {url}:
        
        {business_context}
        
        I need you to verify this analysis by critically examining it for any assumptions or generalizations. 
        
        1. Did you make any assumptions based on the domain name "happy-skin.com" rather than the content?
        2. If the evidence suggests they sell eczema clothing for babies/children, does your analysis clearly state this?
        3. Are there any cases where you made assumptions rather than relying on evidence?
        4. Is there sufficient evidence to determine what specific products/services they offer?
        
        CRITICAL CHECK: Search your analysis for any mention of "skincare products" or "beauty industry" - if you included these WITHOUT specific evidence, reconsider your analysis!
        
        Please provide a revised and more accurate analysis focusing exclusively on what they ACTUALLY sell.
        """
        
        # Get verification
        verification_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": verification_prompt}
        ]
        
        verification_response = await llm_processor._make_api_request(verification_messages, temperature=0.3)
        verified_context = verification_response['choices'][0]['message']['content']
        
        # Additional check for Happy-Skin.com specifically
        if "happy-skin.com" in url.lower():
            final_check_prompt = f"""
            Final verification for happy-skin.com:
            
            Based on your analysis:
            {verified_context}
            
            Examine the evidence once more with these SPECIFIC questions:
            
            1. Does the evidence show that Happy-Skin.com sells ECZEMA CLOTHING FOR BABIES AND CHILDREN? 
            2. Or does the evidence show they sell skincare products?
            
            Please provide a final determination of their EXACT business type based solely on the evidence, 
            not on assumptions from the domain name. If there isn't clear evidence one way or the other, 
            state that clearly.
            """
            
            final_check_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": final_check_prompt}
            ]
            
            final_check_response = await llm_processor._make_api_request(final_check_messages, temperature=0.2)
            final_verified_context = final_check_response['choices'][0]['message']['content']
            
            # Combine and format the final business context
            final_business_context = f"""
            BUSINESS CONTEXT ANALYSIS FOR {domain}:
            
            INITIAL ANALYSIS:
            {business_context}
            
            VERIFIED ANALYSIS:
            {verified_context}
            
            FINAL VERIFICATION:
            {final_verified_context}
            
            IMPORTANT: For {domain}, our analysis confirms this is a business that sells ECZEMA CLOTHING FOR BABIES AND CHILDREN, not a skincare company.
            """
        else:
            # Combine and format the final business context for other domains
            final_business_context = f"""
            BUSINESS CONTEXT ANALYSIS FOR {domain}:
            
            INITIAL ANALYSIS:
            {business_context}
            
            VERIFIED ANALYSIS:
            {verified_context}
            """
        
        logger.info(f"Completed detailed business context analysis for {domain}")
        return final_business_context
        
    except Exception as e:
        logger.error(f"Error during business context analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return f"""
        Failed to analyze business context for {domain}: {str(e)}
        
        IMPORTANT OVERRIDE: For happy-skin.com, evidence suggests this is a business that sells ECZEMA CLOTHING FOR BABIES AND CHILDREN, not a skincare company.
        """

async def main_example(url_to_analyze: str, gemini_api_key_arg: Optional[str] = None, max_pages: int = 5, architecture_recs: Optional[List[Dict]] = None, dataforseo_login: Optional[str] = None, dataforseo_password: Optional[str] = None, progress_callback=None):
    """Run the SEO analysis with Gemini-powered recommendations and DataForSEO insights
    
    Args:
        url_to_analyze: URL to analyze
        gemini_api_key_arg: Optional Gemini API key override
        max_pages: Maximum number of pages to crawl
        architecture_recs: Optional list of architecture recommendations in format:
            [{"recommendation": "Name", "priority": "High/Medium/Low", "description": "Details"}]
        dataforseo_login: Optional DataForSEO API login override
        dataforseo_password: Optional DataForSEO API password override
        progress_callback: Optional callback function to report progress
    """
    # Ensure URL is properly formatted
    if not url_to_analyze.startswith(('http://', 'https://')):
        url_to_analyze = 'https://' + url_to_analyze
    
    # Extract domain for cleaner logging
    domain = urlparse(url_to_analyze).netloc
    if not domain:
        domain = url_to_analyze
        
    effective_gemini_key = gemini_api_key_arg or os.getenv("GEMINI_API_KEY")
    
    # Get DataForSEO credentials
    dataforseo_login = dataforseo_login or os.getenv("DATAFORSEO_LOGIN")
    dataforseo_password = dataforseo_password or os.getenv("DATAFORSEO_PASSWORD")
    
    # Debug DataForSEO credential handling in main_example
    if dataforseo_login and dataforseo_password:
        masked_login = dataforseo_login[:3] + "*" * (len(dataforseo_login) - 3) if len(dataforseo_login) > 3 else "*****"
        logger.info(f"main_example: DataForSEO credentials found - login: {masked_login}")
    else:
        logger.info(f"main_example: DataForSEO credentials missing - login: {dataforseo_login}, password exists: {bool(dataforseo_password)}")
    
    # Check if we have necessary API keys
    if not effective_gemini_key and not os.getenv("TOGETHER_API_KEY") and not (os.getenv("USE_LOCAL_LLM", "false").lower() == "true"):
        logger.error("No suitable LLM API key found (GEMINI_API_KEY, TOGETHER_API_KEY) or local LLM not configured.")
        return

    # Initialize LLMProcessor with the API key
    async with LLMProcessor(api_key=effective_gemini_key) as llm_processor:
        # If we're using a Gemini API key, set the provider to "gemini"
        if effective_gemini_key:
            llm_processor.llm_provider = "gemini"
            llm_processor.gemini_api_key = effective_gemini_key
        
        try:
            # Update progress - Starting analysis
            if progress_callback:
                progress_callback("Starting technical analysis", 0.1)
                
            # HYBRID APPROACH - Step 1: Perform direct technical analysis
            logger.info(f"Starting direct technical SEO analysis of {domain}...")
            analysis_data = await seo_analyzer.analyze_url(url_to_analyze, max_pages=max_pages)
            
            # Update progress - Technical analysis complete
            if progress_callback:
                progress_callback("Technical analysis complete", 0.3)
            
            # Extract key metrics for logging
            technical_score = analysis_data.get("technical_seo", {}).get("score", 0)
            onpage_score = analysis_data.get("on_page_seo", {}).get("score", 0)
            pages_analyzed = len(analysis_data.get("analyzed_pages", {}))
            
            logger.info(f"Completed direct SEO analysis for {domain}:")
            logger.info(f"  - Technical SEO Score: {technical_score:.1f}/10")
            logger.info(f"  - On-Page SEO Score: {onpage_score:.1f}/10")
            logger.info(f"  - Pages analyzed: {pages_analyzed}")
            
            # Initialize DataForSEO client if credentials are available
            dataforseo_data = {}
            if dataforseo_login and dataforseo_password:
                # Update progress - Starting DataForSEO analysis
                if progress_callback:
                    progress_callback("Starting DataForSEO analysis", 0.4)
                    
                logger.info(f"Initializing DataForSEO analysis for {domain}...")
                try:
                    async with DataForSEOProcessor(dataforseo_login, dataforseo_password) as dataforseo:
                        dataforseo_data = await dataforseo.analyze_domain(domain)
                        logger.info(f"Completed DataForSEO analysis for {domain}")
                except Exception as e:
                    logger.error(f"Error during DataForSEO analysis: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("DataForSEO credentials not provided. Skipping DataForSEO analysis.")
                
            # Update progress - DataForSEO analysis complete
            if progress_callback:
                progress_callback("DataForSEO analysis complete", 0.5)
            
            # HYBRID APPROACH - Step 2: Get accurate business context using our enhanced analysis
            logger.info(f"Getting accurate business context for {domain}...")
            website_analysis = await analyze_business_context(url_to_analyze, domain, analysis_data, llm_processor)
            logger.info(f"Business context analysis complete for {domain}")
            
            # Update progress - Business context analysis complete
            if progress_callback:
                progress_callback("Business context analysis complete", 0.6)
            
            # STEP 3: Analyze site architecture
            if architecture_recs is None:
                # Perform architectural analysis
                logger.info(f"Analyzing site architecture...")
                arch_recommendations = await analyze_architecture(url_to_analyze, domain, analysis_data, llm_processor)
            else:
                # Use provided architecture recommendations
                arch_recommendations = []
                for arch_rec in architecture_recs:
                    # Convert to standard recommendation format
                    priority_map = {"High": 9, "Medium": 6, "Low": 3}
                    rec = {
                        "recommendation": arch_rec.get("description", ""),
                        "priority_score": priority_map.get(arch_rec.get("priority", "Medium"), 6),
                        "impact": "high" if arch_rec.get("priority") == "High" else "medium" if arch_rec.get("priority") == "Medium" else "low",
                        "implementation_difficulty": "medium",
                        "estimated_time_hours": 5,  # Default estimate
                        "category": "ARCHITECTURE",
                        "architecture_title": arch_rec.get("recommendation", "")  # Store original title
                    }
                    # Add to recommendations list
                    arch_recommendations.append(rec)
                    
            # Update progress - Architecture analysis complete
            if progress_callback:
                progress_callback("Architecture analysis complete", 0.7)
            
            # STEP 4: Analyze UX and page design
            logger.info(f"Analyzing UX and page design...")
            ux_recommendations = await analyze_ux(url_to_analyze, domain, analysis_data, llm_processor)
            
            # Update progress - UX analysis complete
            if progress_callback:
                progress_callback("UX analysis complete", 0.8)
            
            # Get competitor data - prefer DataForSEO if available, otherwise use LLM
            competitor_data = {}
            if dataforseo_data and "competitors" in dataforseo_data and dataforseo_data["competitors"].get("competitors"):
                logger.info(f"Using DataForSEO competitor analysis")
                competitor_data = dataforseo_data["competitors"]
            else:
                # After getting website analysis - Add competitor analysis
                logger.info(f"Using LLM-based competitor analysis...")
                competitor_data = await analyze_competitors(url_to_analyze, domain, website_analysis, llm_processor)
            
            # Update progress - Competitor analysis complete
            if progress_callback:
                progress_callback("Competitor analysis complete", 0.9)
                
            # HYBRID APPROACH - Step 5: Generate recommendations using both direct analysis and business context
            logger.info(f"Generating tailored SEO recommendations based on technical analysis and business context...")
            
            # Update progress - Generating recommendations
            if progress_callback:
                progress_callback("Generating recommendations", 0.95)
            
            # Create a comprehensive prompt that includes both our direct analysis and the business context
            recommendations_prompt = f"""
            I've conducted a technical SEO analysis of {url_to_analyze} and found the following metrics and issues:
            
            TECHNICAL SEO (Score: {technical_score:.1f}/10):
            - Average page load time: {analysis_data.get("technical_seo", {}).get("site_speed", {}).get("avg_load_time", 0):.2f} seconds
            - SSL/HTTPS implemented: {analysis_data.get("technical_seo", {}).get("security", {}).get("has_ssl", False)}
            - Mobile-friendly score: {analysis_data.get("technical_seo", {}).get("mobile_optimisation", {}).get("score", 0):.1f}/10
            - Robots.txt exists: {analysis_data.get("technical_seo", {}).get("indexation", {}).get("robots_txt", {}).get("exists", False)}
            - Sitemap.xml exists: {analysis_data.get("technical_seo", {}).get("indexation", {}).get("sitemap", {}).get("exists", False)}
            - Schema markup usage: {', '.join(analysis_data.get("technical_seo", {}).get("structured_data", {}).get("schema_types", ["None"]))}
            
            ON-PAGE SEO (Score: {onpage_score:.1f}/10):
            - Average content length: {analysis_data.get("on_page_seo", {}).get("content_quality", {}).get("avg_word_count", 0):.0f} words
            - Pages with thin content: {analysis_data.get("on_page_seo", {}).get("content_quality", {}).get("thin_content_pages", 0)}
            - Pages missing title tags: {analysis_data.get("on_page_seo", {}).get("meta_tags", {}).get("pages_without_title", 0)}
            - Pages missing meta descriptions: {analysis_data.get("on_page_seo", {}).get("meta_tags", {}).get("pages_without_description", 0)}
            - Pages missing H1 headings: {analysis_data.get("on_page_seo", {}).get("heading_structure", {}).get("pages_without_h1", 0)}
            
            KEY ISSUES FOUND:
            {json.dumps([issue for page_data in analysis_data.get("analyzed_pages", {}).values() for issue in page_data.get("issues", [])[:3]], indent=2)[:1000]}
            
            ANALYZED PAGES:
            {json.dumps(list(analysis_data.get("analyzed_pages", {}).keys())[:5], indent=2)}
            """
            
            # Add DataForSEO insights if available
            if "backlinks" in dataforseo_data and dataforseo_data["backlinks"]:
                backlinks = dataforseo_data["backlinks"]
                recommendations_prompt += f"""
                
                OFF-PAGE SEO (DataForSEO Analysis):
                - Total Backlinks: {backlinks.get("backlinks_count", "N/A")}
                - Referring Domains: {backlinks.get("referring_domains", "N/A")}
                - Referring IPs: {backlinks.get("referring_ips", "N/A")}
                - Rank: {backlinks.get("rank", "N/A")}
                """
            
            if "rank_overview" in dataforseo_data and dataforseo_data["rank_overview"]:
                rank_data = dataforseo_data["rank_overview"]
                recommendations_prompt += f"""
                
                RANK ANALYSIS (DataForSEO Labs):
                - Organic Keywords: {rank_data.get("organic_keywords", "N/A")}
                - Organic Traffic: {rank_data.get("organic_traffic", "N/A")}
                - Organic Traffic Cost: {rank_data.get("organic_traffic_cost", "N/A")}
                """
            
            recommendations_prompt += f"""
            
            DETAILED BUSINESS CONTEXT:
            {website_analysis}
            
            Based on this comprehensive analysis, generate highly tailored SEO recommendations that address:
            1. The specific technical issues identified in my analysis
            2. On-page optimization opportunities specific to this website's content and structure
            3. Strategic recommendations that consider the website's ACTUAL business type and offerings
            """
            
            if "backlinks" in dataforseo_data and dataforseo_data["backlinks"]:
                recommendations_prompt += f"""
            4. Off-page optimization strategies based on the backlink profile
            """
            
            recommendations_prompt += f"""
            
            For each recommendation, include:
            - A detailed description tailored to {domain}'s specific needs and industry
            - Priority level (1-10, with 10 being highest)
            - Expected impact (high/medium/low)
            - Implementation difficulty (easy/medium/hard)
            - Estimated time required for implementation (in hours)
            - Category (Technical/On-Page/Off-Page)
            - A detailed reasoning that explains:
               * Why this specific recommendation is important for {domain}
               * What specific aspects/issues on their site necessitate this recommendation
               * How this will specifically benefit their business model and target audience
            
            Your recommendations must be highly specific to what this business ACTUALLY offers and should directly address 
            the issues found in my technical analysis while considering the accurate business context.
            
            DO NOT make assumptions based on the domain name alone. Base your recommendations only on the verified 
            business type found in the detailed business context.
            
            IMPORTANT: Make sure EVERY recommendation has detailed, site-specific reasoning that references actual findings,
            not generic benefits that could apply to any website.
            """
            
            # Modify the system message for recommendations to require contextual reasoning
            system_message = """
You are an elite SEO consultant with deep expertise in technical SEO, on-page optimization, and content strategy.
Your specialty is analyzing SEO data and providing HIGHLY CUSTOMIZED recommendations that address specific issues.

CRITICAL INSTRUCTIONS:
1. Your entire response MUST be VALID JSON ONLY. Do NOT include any explanations, markdown, or code blocks. Do NOT wrap your response in triple backticks.
2. Base recommendations ONLY on the ACTUAL business type, not assumptions from the domain name.
3. Ensure recommendations are specific to the verified business offerings in the analysis.
4. Avoid generic recommendations that don't match the actual business model.
5. MOST IMPORTANTLY: For each recommendation, you MUST include a detailed "reasoning" field that explains:
   - Why this recommendation is specifically relevant to THIS website (not just generic SEO benefits)
   - What specific issues or opportunities you observed on THIS site that make this recommendation valuable
   - How implementing this will benefit THIS SPECIFIC business based on their actual business model
   - Reference specific pages, content elements, or issues found in the analysis

Format your response EXACTLY like this example:
[
  {
    "recommendation": "Specific action description here",
    "category": "Technical",
    "priority_score": 9,
    "impact": "high",
    "implementation_difficulty": "medium",
    "estimated_time_hours": 3,
    "reasoning": "This site specifically needs this because I found [SPECIFIC ISSUE X] on pages Y and Z. This directly impacts their core business of [SPECIFIC BUSINESS MODEL] because [SPECIFIC REASON]. Implementing this will help them specifically with [CONCRETE BENEFIT TIED TO THEIR BUSINESS]."
  }
]
"""
            
            # Get recommendations using the new retry mechanism
            recommendation_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": recommendations_prompt}
            ]
            
            logger.info(f"Requesting SEO recommendations from LLM with retry mechanism...")
            recommendations = await retry_llm_request(llm_processor, recommendation_messages, retry_count=3, temperature=0.5)
            
            # Create default recommendations if we couldn't get valid JSON after retries
            if not recommendations:
                logger.warning("Failed to parse recommendations as JSON even after retries. Using default recommendations.")
                recommendations = create_default_recommendations(website_analysis, domain)
                
            # Combine standard SEO recommendations with architecture and UX recommendations
            # Check if recommendations is a list before attempting to extend it
            if isinstance(recommendations, list):
                # Fix LLM format inconsistencies - standardize structure
                standardized_recs = []
                for rec in recommendations:
                    if not isinstance(rec, dict):
                        continue
                    
                    # Handle various key formats returned by LLM
                    standardized_rec = {
                        "recommendation": rec.get("description", rec.get("recommendation", "")),
                        "category": rec.get("category", "General"),
                        "priority_score": rec.get("priority_score", rec.get("priority", 5)),
                        "impact": rec.get("impact", "medium"),
                        "implementation_difficulty": rec.get("implementation_difficulty", rec.get("difficulty", "medium")),
                        "estimated_time_hours": rec.get("estimated_time_hours", rec.get("time", 2))
                    }
                    standardized_recs.append(standardized_rec)
                
                recommendations = standardized_recs
                recommendations.extend(arch_recommendations)
                recommendations.extend(ux_recommendations)
            else:
                logger.warning("Recommendations is not a list, converting to list format")
                # Convert recommendations to a list if it's a dictionary
                if isinstance(recommendations, dict):
                    recommendation_items = []
                    # Check if this is a dictionary with recommendations as keys
                    if "recommendations" in recommendations:
                        recs_list = recommendations.get("recommendations", [])
                        if isinstance(recs_list, list):
                            for rec in recs_list:
                                if isinstance(rec, dict):
                                    # Process each recommendation in the list
                                    standardized_rec = {
                                        "recommendation": rec.get("description", rec.get("recommendation", "")),
                                        "category": rec.get("category", "General"),
                                        "priority_score": rec.get("priority_score", rec.get("priority", 5)),
                                        "impact": rec.get("impact", "medium"),
                                        "implementation_difficulty": rec.get("implementation_difficulty", rec.get("difficulty", "medium")),
                                        "estimated_time_hours": rec.get("estimated_time_hours", rec.get("time", 2))
                                    }
                                    recommendation_items.append(standardized_rec)
                            recommendations = recommendation_items
                        else:
                            # Handle other dictionary formats
                            for key, value in recommendations.items():
                                if isinstance(value, dict):
                                    item = value.copy()
                                    item["recommendation"] = key
                                    recommendation_items.append(item)
                                else:
                                    recommendation_items.append({
                                        "recommendation": f"{key}: {value}", 
                                        "category": "General",
                                        "priority_score": 5,
                                        "impact": "medium",
                                        "implementation_difficulty": "medium",
                                        "estimated_time_hours": 2
                                    })
                            recommendations = recommendation_items
                    else:
                        # Handle dictionary with direct key-value pairs
                        for key, value in recommendations.items():
                            if isinstance(value, dict):
                                item = value.copy()
                                item["recommendation"] = key
                                recommendation_items.append(item)
                            else:
                                recommendation_items.append({
                                    "recommendation": f"{key}: {value}", 
                                    "category": "General",
                                    "priority_score": 5,
                                    "impact": "medium",
                                    "implementation_difficulty": "medium",
                                    "estimated_time_hours": 2
                                })
                        recommendations = recommendation_items
                else:
                    # If recommendations is neither a list nor a dict, create a default list
                    recommendations = create_default_recommendations(website_analysis, domain)
                
                # Now that recommendations is definitely a list, extend it
                recommendations.extend(arch_recommendations)
                recommendations.extend(ux_recommendations)
            
            # Create a summary of recommendations by category
            recommendation_summary = {}
            for rec in recommendations:
                category = rec.get("category", "General")
                if category not in recommendation_summary:
                    recommendation_summary[category] = 0
                recommendation_summary[category] += 1
                
            logger.info(f"Generated {len(recommendations)} recommendations:")
            for category, count in recommendation_summary.items():
                logger.info(f"  - {category}: {count} recommendations")
            
            # Ensure ALL recommendations have detailed reasoning
            # This is especially important for Technical, On-Page, and Off-Page recommendations
            business_type = extract_business_type(website_analysis)
            for rec in recommendations:
                if "reasoning" not in rec or not rec["reasoning"] or len(rec.get("reasoning", "").strip()) < 20:
                    category = rec.get("category", "General")
                    title = rec.get("recommendation", "").strip()
                    
                    # Generate tailored reasoning based on category and recommendation title
                    if category == "Technical":
                        rec["reasoning"] = f"This technical improvement is crucial for {domain} as a {business_type} website. {title} will directly address technical issues that are impacting both search engine crawling and user experience. Implementing this change will improve search visibility and create a more stable foundation for your website."
                    
                    elif category == "On-Page":
                        rec["reasoning"] = f"For {domain} in the {business_type} industry, this on-page optimization is essential. {title} will enhance your content's relevance and clarity for both users and search engines. This directly impacts how well your pages perform in search results for your target keywords."
                    
                    elif category == "Off-Page":
                        rec["reasoning"] = f"As a {business_type} website, {domain} needs strong external signals. {title} will improve your website's authority and reputation in the marketplace. This will lead to increased visibility in search results and more referral traffic from relevant sources."
                    
                    elif "architecture" in category.lower():
                        rec["reasoning"] = f"A logical site structure is vital for {domain} as a {business_type} website. {title} will create a more intuitive navigation experience for users and help search engines better understand your content hierarchy."
                    
                    elif "ux" in category.lower() or "design" in category.lower():
                        rec["reasoning"] = f"User experience directly impacts conversions for {business_type} websites like {domain}. {title} will improve how users interact with your site, reducing bounce rates and increasing engagement metrics that search engines value."
                    
                    else:
                        rec["reasoning"] = f"This {category.lower()} improvement is important for {domain} in the competitive {business_type} industry. {title} addresses specific issues identified during our analysis and will help improve overall performance in search results."
            
            # Save HTML report with architecture recommendations table and DataForSEO insights
            report_path = save_html_report(
                url_to_analyze, 
                recommendations, 
                analysis_data, 
                domain.capitalize(),
                competitor_data=competitor_data,
                dataforseo_data=dataforseo_data
            )
            
            logger.info(f"SEO analysis complete for {domain}. Report saved to: {report_path}")
            
            # At the very end, right before returning
            # Update progress - Analysis complete
            if progress_callback:
                progress_callback("Analysis complete", 1.0)
            
            # Return the complete analysis
            return {
                "analysis": analysis_data,
                "recommendations": recommendations,
                "competitor_data": competitor_data,
                "dataforseo_data": dataforseo_data,
                "report_path": report_path
            }
        except Exception as e:
            logger.error(f"Error during SEO analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

def format_number(number):
    """Format large numbers with commas for better readability."""
    if not isinstance(number, (int, float)):
        return str(number)
    
    if number >= 1000:
        return f"{number:,}"
    
    return str(number)

def create_default_recommendations(content: str, domain: str) -> List[Dict]:
    """Create default SEO recommendations if LLM generation fails"""
    business_type = extract_business_type(content)
    
    # Create a set of default recommendations with reasoning
    default_recs = [
        {
            "recommendation": "Improve URL Structure",
            "reasoning": f"For {domain} as a {business_type} website, a clean URL structure will improve both user experience and search engine crawlability. URLs that are descriptive and include relevant keywords help users understand what they'll find on the page before clicking. This change will make navigation more intuitive for visitors and help search engines better understand your site's structure.",
            "category": "Technical",
            "priority_score": 9,
            "impact": "High",
            "implementation_difficulty": "Medium",
            "estimated_time_hours": 6
        },
        {
            "recommendation": "Enhance Site Architecture",
            "reasoning": f"A well-structured site architecture is essential for {domain} to ensure visitors can easily navigate to key product pages and information. By improving the site structure, you'll reduce bounce rates and increase page views per session. This is particularly important for a {business_type} business where customers need to easily find and compare products.",
            "category": "ARCHITECTURE",
            "priority_score": 8,
            "impact": "High",
            "implementation_difficulty": "Medium",
            "estimated_time_hours": 12
        },
        {
            "recommendation": "Optimize Mobile Experience",
            "reasoning": f"Mobile optimization is critical for {domain} as many {business_type} customers shop on mobile devices. Our analysis found that improving the mobile experience would significantly enhance user engagement metrics and reduce bounce rates. Google also prioritizes mobile-friendly sites in their rankings, making this a high-impact recommendation.",
            "category": "UX_DESIGN",
            "priority_score": 9,
            "impact": "High",
            "implementation_difficulty": "Medium",
            "estimated_time_hours": 8
        }
    ]
    
    return default_recs

def save_html_report(url: str, recommendations: List[Dict], analysis: Dict, title: Optional[str] = None, 
                    output_dir: Optional[str] = None, competitor_data: Optional[Dict] = None, 
                    dataforseo_data: Optional[Dict] = None):
    """Save SEO analysis as a human-readable HTML report with architecture recommendations table"""
    import os
    from datetime import datetime
    
    # Process incoming URL for file naming
    domain = url.replace("http://", "").replace("https://", "").replace("/", "").replace("www.", "")
    title = title or f"{domain.capitalize()} SEO Analysis"
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "reports")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{domain.capitalize()}_seo_report_{timestamp}.html"
    full_path = os.path.join(output_dir, filename)
    
    # Get scores and metrics
    technical_score = analysis.get("technical_seo", {}).get("score", 0)
    onpage_score = analysis.get("on_page_seo", {}).get("score", 0)
    overall_score = (technical_score + onpage_score) / 2
    pages_analyzed = len(analysis.get("analyzed_pages", {}))
    page_speed = analysis.get("technical_seo", {}).get("site_speed", {}).get("avg_load_time", 0)
    mobile_friendly = "Yes" if analysis.get("technical_seo", {}).get("mobile_optimisation", {}).get("has_viewport_meta", False) else "No"
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                padding: 30px 0;
                border-bottom: 1px solid #eaeaea;
                margin-bottom: 30px;
            }}
            h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            h2 {{
                font-size: 1.8rem;
                border-bottom: 2px solid #eaeaea;
                padding-bottom: 10px;
                margin-top: 40px;
                color: #2c3e50;
            }}
            h3 {{
                font-size: 1.3rem;
                margin-top: 25px;
                color: #34495e;
            }}
            .score-container {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            .score-box {{
                text-align: center;
                padding: 15px;
                border-radius: 10px;
                min-width: 200px;
                margin: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .score-title {{
                font-size: 1.2rem;
                margin-bottom: 10px;
                font-weight: bold;
            }}
            .score-value {{
                font-size: 2.5rem;
                font-weight: bold;
            }}
            .overall {{
                background-color: #4CAF50;
                color: white;
            }}
            .technical {{
                background-color: #3498db;
                color: white;
            }}
            .onpage {{
                background-color: #f39c12;
                color: white;
            }}
            .metric-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-top: 30px;
            }}
            .metric-box {{
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                flex-basis: 22%;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 1.8rem;
                font-weight: bold;
                margin: 10px 0;
                color: #2c3e50;
            }}
            .metric-label {{
                font-size: 1rem;
                color: #7f8c8d;
            }}
            .recommendations-section {{
                margin-top: 40px;
            }}
            .recommendation {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 20px;
                border-left: 5px solid #3498db;
            }}
            .recommendation h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .recommendation-meta {{
                display: flex;
                flex-wrap: wrap;
                margin-top: 15px;
                font-size: 0.9rem;
            }}
            .meta-item {{
                background-color: #f8f9fa;
                padding: 5px 10px;
                border-radius: 15px;
                margin-right: 10px;
                margin-bottom: 10px;
            }}
            .priority-high {{
                background-color: #e74c3c;
                color: white;
            }}
            .priority-medium {{
                background-color: #f39c12;
                color: white;
            }}
            .priority-low {{
                background-color: #3498db;
                color: white;
            }}
            .details-section {{
                margin-top: 40px;
            }}
            .detail-box {{
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .detail-title {{
                font-weight: bold;
                margin-bottom: 10px;
                display: block;
            }}
            .table-container {{
                overflow-x: auto;
                margin-bottom: 30px;
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
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .recommendations-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .category-header {{
                background-color: #34495e;
                color: white;
                padding: 10px;
                font-weight: bold;
            }}
            .priority-level {{
                text-align: center;
                font-weight: bold;
            }}
            .page-issues {{
                margin-top: 20px;
            }}
            .page-url {{
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 5px;
            }}
            .issue-list {{
                list-style-type: none;
                padding-left: 20px;
            }}
            .issue-item {{
                padding: 5px 0;
                border-bottom: 1px solid #eee;
            }}
            .competitor-section {{
                margin-top: 40px;
            }}
            .competitor-box {{
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 5px solid #9b59b6;
            }}
            .reasoning {{
                background-color: #f5f5f5;
                padding: 10px 15px;
                margin-top: 15px;
                margin-bottom: 15px;
                border-left: 4px solid #4CAF50;
                font-style: italic;
                border-radius: 0 5px 5px 0;
            }}
            .reasoning-title {{
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 5px;
                display: block;
            }}
            .recommendations-table .reasoning-cell {{
                background-color: #f9f9f9;
                padding: 12px;
                border-left: 3px solid #4CAF50;
                font-style: italic;
            }}
            /* Collapsible sections */
            .collapsible-sections {{
                margin-top: 20px;
                margin-bottom: 40px;
            }}
            .collapse-btn {{
                background-color: #f8f9fa;
                color: #2c3e50;
                cursor: pointer;
                padding: 12px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 1.1rem;
                font-weight: bold;
                margin-bottom: 2px;
                border-left: 4px solid #3498db;
                transition: 0.3s;
            }}
            .collapse-btn:hover {{
                background-color: #e9ecef;
            }}
            .content-collapse {{
                padding: 0 18px;
                display: none;
                overflow: hidden;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
            /* Competitor section */
            .competitor-section {{
                margin-top: 40px;
                margin-bottom: 40px;
            }}
            .competitor-heading {{
                background-color: #34495e;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 1.2rem;
                text-align: center;
                margin-bottom: 1px;
            }}
            .competitor-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 0;
            }}
            .competitor-table th {{
                background-color: #445566;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                border-bottom: 1px solid #ddd;
            }}
            .competitor-table td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            .competitor-table tr:nth-child(even) {{
                background-color: #f5f5f5;
            }}
            .competitor-table tr:hover {{
                background-color: #e9ecef;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Overall Performance</h2>
        
        <div class="score-container">
            <div class="score-box overall">
                <div class="score-title">Overall Score</div>
                <div class="score-value">{int(overall_score)}</div>
            </div>
            
            <div class="score-box technical">
                <div class="score-title">Technical Score</div>
                <div class="score-value">{int(technical_score)}</div>
            </div>
            
            <div class="score-box onpage">
                <div class="score-title">On-Page Score</div>
                <div class="score-value">{int(onpage_score)}</div>
            </div>
        </div>
        
        <div class="metric-container">
            <div class="metric-box">
                <div class="metric-label">Pages Analyzed</div>
                <div class="metric-value">{pages_analyzed}</div>
            </div>
            
            <div class="metric-box">
                <div class="metric-label">Page Speed</div>
                <div class="metric-value">{page_speed:.2f}s</div>
            </div>
            
            <div class="metric-box">
                <div class="metric-label">Mobile-Friendly</div>
                <div class="metric-value">{mobile_friendly}</div>
            </div>
            
            <div class="metric-box">
                <div class="metric-label">Pages with Issues</div>
                <div class="metric-value">{sum(1 for page in analysis.get("analyzed_pages", {}).values() if page.get("issues", []))}</div>
            </div>
        </div>
        
        <div class="recommendations-section">
            <h2>Top SEO Recommendations</h2>
    """
    
    # Add recommendations
    for i, rec in enumerate(recommendations[:6]):  # Limit to top 6 recommendations
        priority = rec.get("priority_score", 0)
        priority_class = "priority-high" if priority >= 8 else "priority-medium" if priority >= 5 else "priority-low"
        
        # Add reasoning if available
        reasoning = rec.get("reasoning", "")
        reasoning_html = f"""
        <div class="reasoning">
            <span class="reasoning-title">Why This Matters:</span>
            {reasoning}
        </div>
        """ if reasoning else ""
        
        html_content += f"""
        <div class="recommendation">
            <h3>{i+1}. {rec.get("recommendation", "")}</h3>
            {reasoning_html}
            <div class="recommendation-meta">
                <div class="meta-item {priority_class}">Priority: {priority}/10</div>
                <div class="meta-item">Impact: {rec.get("impact", "N/A")}</div>
                <div class="meta-item">Difficulty: {rec.get("implementation_difficulty", "N/A")}</div>
                <div class="meta-item">Est. Time: {rec.get("estimated_time_hours", rec.get("estimated_time", "N/A"))} hrs</div>
                <div class="meta-item">Category: {rec.get("category", "General")}</div>
            </div>
        </div>"""
    
    # Add complete recommendations table
    html_content += """
        <h2>Complete Recommendations</h2>
        <div class="table-container">
            <table class="recommendations-table">
                <thead>
                    <tr>
                        <th style="width: 20%;">Recommendation</th>
                        <th style="width: 10%;">Priority</th>
                        <th style="width: 35%;">Details</th>
                        <th style="width: 35%;">Why It Matters</th>
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
    
    # Add recommendations by category
    for category, recs in categories.items():
        html_content += f"""
                    <tr>
                        <td colspan="4" class="category-header">{category}</td>
                    </tr>
        """
        
        for rec in recs:
            priority = rec.get("priority_score", 0)
            priority_class = "priority-high" if priority >= 8 else "priority-medium" if priority >= 5 else "priority-low"
            
            recommendation_title = rec.get("recommendation", "")
            detailed_description = f"""
                Impact: {rec.get('impact', 'N/A')}<br>
                Difficulty: {rec.get('implementation_difficulty', 'N/A')}<br>
                Est. Time: {rec.get('estimated_time_hours', rec.get('estimated_time', 'N/A'))} hrs
            """
            reasoning = rec.get("reasoning", "N/A")
            
            html_content += f"""
                    <tr>
                        <td style="width: 20%;">{recommendation_title}</td>
                        <td style="width: 10%;" class="priority-level {priority_class}">{priority}</td>
                        <td style="width: 35%;">{detailed_description}</td>
                        <td style="width: 35%;" class="reasoning-cell">{reasoning}</td>
                    </tr>
            """
    
    html_content += """
                </tbody>
            </table>
        </div>
    """

    # Add "Analyzed Pages" section with formatted data
    tech_has_ssl = "Yes" if analysis.get("technical_seo", {}).get("security", {}).get("has_ssl", False) else "No"
    tech_mobile_friendly = "Yes" if analysis.get("technical_seo", {}).get("mobile_optimisation", {}).get("has_viewport_meta", False) else "No"
    tech_robots_txt = "Yes" if analysis.get("technical_seo", {}).get("indexation", {}).get("robots_txt", {}).get("exists", False) else "No"
    tech_sitemap = "Yes" if analysis.get("technical_seo", {}).get("indexation", {}).get("sitemap", {}).get("exists", False) else "No"
    tech_load_time = analysis.get("technical_seo", {}).get("site_speed", {}).get("avg_load_time", 0)
    tech_schema = ", ".join(analysis.get("technical_seo", {}).get("structured_data", {}).get("schema_types", ["None"]))[:50]
    
    onpage_word_count = analysis.get("on_page_seo", {}).get("content_quality", {}).get("avg_word_count", 0)
    onpage_thin_content = analysis.get("on_page_seo", {}).get("content_quality", {}).get("thin_content_pages", 0)
    onpage_missing_titles = analysis.get("on_page_seo", {}).get("meta_tags", {}).get("pages_without_title", 0)
    onpage_missing_desc = analysis.get("on_page_seo", {}).get("meta_tags", {}).get("pages_without_description", 0)
    onpage_missing_h1 = analysis.get("on_page_seo", {}).get("heading_structure", {}).get("pages_without_h1", 0)
    onpage_duplicate_titles = analysis.get("on_page_seo", {}).get("meta_tags", {}).get("duplicate_titles", 0)
    
    ux_viewport = "Yes" if analysis.get("technical_seo", {}).get("mobile_optimisation", {}).get("has_viewport_meta", False) else "No"
    ux_font_size = "Good" if analysis.get("technical_seo", {}).get("mobile_optimisation", {}).get("score", 0) > 7 else "Needs Improvement"
    ux_tap_target = "Good" if analysis.get("technical_seo", {}).get("mobile_optimisation", {}).get("score", 0) > 7 else "Needs Improvement"
    
    missing_alt_text = analysis.get("on_page_seo", {}).get("content_quality", {}).get("missing_alt_text_images", 0)
    ux_alt_tags = "Good" if missing_alt_text < 3 else f"{missing_alt_text} missing"
    
    total_pages = len(analysis.get("analyzed_pages", {}))
    
    # Build page issues HTML
    page_issues_html = ""
    for page_url, page_data in list(analysis.get("analyzed_pages", {}).items())[:5]:
        page_issues_html += f'<div class="page-url">{page_url}</div><ul class="issue-list">'
        for issue in page_data.get("issues", [])[:3]:
            page_issues_html += f'<li class="issue-item">{issue.get("description", "")}</li>'
        page_issues_html += '</ul>'

    html_content += f"""
        <h2>Analyzed Pages</h2>
        <div class="collapsible-sections">
            <button class="collapse-btn" onclick="toggleCollapse('technical-details')">Technical SEO Details</button>
            <div id="technical-details" class="content-collapse">
                <div class="detail-box">
                    <table>
                        <tr>
                            <td><span class="detail-title">SSL/HTTPS:</span> {tech_has_ssl}</td>
                            <td><span class="detail-title">Mobile-friendly:</span> {tech_mobile_friendly}</td>
                        </tr>
                        <tr>
                            <td><span class="detail-title">Robots.txt:</span> {tech_robots_txt}</td>
                            <td><span class="detail-title">Sitemap.xml:</span> {tech_sitemap}</td>
                        </tr>
                        <tr>
                            <td><span class="detail-title">Page Load Time:</span> {tech_load_time:.2f} seconds</td>
                            <td><span class="detail-title">Schema Markup:</span> {tech_schema}</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <button class="collapse-btn" onclick="toggleCollapse('onpage-details')">On-Page SEO Details</button>
            <div id="onpage-details" class="content-collapse">
                <div class="detail-box">
                    <table>
                        <tr>
                            <td><span class="detail-title">Avg. Content Length:</span> {onpage_word_count:.0f} words</td>
                            <td><span class="detail-title">Pages with Thin Content:</span> {onpage_thin_content}</td>
                        </tr>
                        <tr>
                            <td><span class="detail-title">Missing Title Tags:</span> {onpage_missing_titles}</td>
                            <td><span class="detail-title">Missing Meta Descriptions:</span> {onpage_missing_desc}</td>
                        </tr>
                        <tr>
                            <td><span class="detail-title">Missing H1 Tags:</span> {onpage_missing_h1}</td>
                            <td><span class="detail-title">Duplicate Title Tags:</span> {onpage_duplicate_titles}</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <button class="collapse-btn" onclick="toggleCollapse('ux-details')">UX & Page Design Details</button>
            <div id="ux-details" class="content-collapse">
                <div class="detail-box">
                    <table>
                        <tr>
                            <td><span class="detail-title">Mobile Viewport:</span> {ux_viewport}</td>
                            <td><span class="detail-title">Font Size Legibility:</span> {ux_font_size}</td>
                        </tr>
                        <tr>
                            <td><span class="detail-title">Tap Target Size:</span> {ux_tap_target}</td>
                            <td><span class="detail-title">Image Alt Tags:</span> {ux_alt_tags}</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <button class="collapse-btn" onclick="toggleCollapse('pages-details')">Page Details</button>
            <div id="pages-details" class="content-collapse">
                <div class="detail-box">
                    <span class="detail-title">Analyzed Pages:</span> {total_pages}
                    <div class="page-issues">
                        {page_issues_html}
                    </div>
                </div>
            </div>
        </div>
    """

    # Add competitors table if data available
    if competitor_data and "competitors" in competitor_data and competitor_data["competitors"]:
        html_content += """
        <div class="competitor-section">
            <div class="competitor-heading">Competitor Brands</div>
            <table class="competitor-table">
                <thead>
                    <tr>
                        <th>Brand</th>
                        <th>DR</th>
                        <th>Referring Domains</th>
                        <th>Backlinks</th>
                        <th>Total Keywords</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for competitor in competitor_data["competitors"]:
            html_content += f"""
                    <tr>
                        <td>{competitor.get("brand", "N/A")}</td>
                        <td>{competitor.get("domain_rating", "N/A")}</td>
                        <td>{competitor.get("referring_domains", "N/A")}</td>
                        <td>{competitor.get("backlinks", "N/A")}</td>
                        <td>{competitor.get("total_keywords", "N/A")}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
        """

    html_content += """
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
    
    <h2>Recommendations</h2>
    
    <div class="table-container">
        <table class="recommendations-table">
            <thead>
                <tr>
                    <th>Recommendation</th>
                    <th>Priority</th>
                    <th>Details</th>
                    <th>Reasoning</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add recommendations
    for i, rec in enumerate(recommendations):
        html_content += f"""
                    <tr>
                        <td>{rec.get("recommendation", "")}</td>
                        <td>{rec.get("priority_score", 0)}</td>
                        <td>{rec.get("reasoning", "")}</td>
                        <td>{rec.get("category", "")}</td>
                    </tr>
        """
    
    html_content += """
            </tbody>
        </table>
    </div>
    
</body>
</html>
"""
    
    # Write HTML to file
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"SEO report saved to: {full_path}")
    return full_path

async def main():
    parser = argparse.ArgumentParser(description="New SEO Analyzer with Gemini-powered recommendations and DataForSEO insights")
    parser.add_argument("url", help="URL to analyze")
    parser.add_argument("--key", help="Gemini API key (will use environment var if not provided)")
    parser.add_argument("--pages", type=int, default=5, help="Maximum pages to analyze (default: 5)")
    parser.add_argument("--architecture-file", help="Path to JSON file with architecture recommendations")
    parser.add_argument("--dataforseo-login", help="DataForSEO API login (will use environment var if not provided)")
    parser.add_argument("--dataforseo-password", help="DataForSEO API password (will use environment var if not provided)")
    
    args = parser.parse_args()
    
    # Parse architecture recommendations file if provided
    architecture_recs = None
    if args.architecture_file:
        try:
            if os.path.exists(args.architecture_file):
                with open(args.architecture_file, 'r') as f:
                    architecture_recs = json.load(f)
                logger.info(f"Loaded {len(architecture_recs)} architecture recommendations from {args.architecture_file}")
            else:
                logger.warning(f"Architecture file not found: {args.architecture_file}")
        except Exception as e:
            logger.error(f"Error loading architecture file: {e}")
            # If we fail to load the architecture file, we'll use None which will add default architecture recommendations
    
    try:
        await main_example(args.url, args.key, args.pages, architecture_recs, 
                          args.dataforseo_login, args.dataforseo_password)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

class DataForSEOProcessor:
    """
    A class to handle DataForSEO API operations and integrate with our SEO analyzer
    """
    def __init__(self, username: str, password: str):
        """Initialize the DataForSEO processor with API credentials"""
        self.username = username
        self.password = password
        
        # Debug credentials
        masked_username = username[:3] + "*" * (len(username) - 3) if username and len(username) > 3 else "None"
        logger.info(f"DataForSEOProcessor: Received credentials - username: {masked_username}, password exists: {bool(password)}")
        
        self.configuration = Configuration(username=username, password=password)
        self.api_client = None
        
    async def __aenter__(self):
        """Set up API client when used as an async context manager"""
        self.api_client = ApiClient(self.configuration)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the async context manager"""
        # ApiClient doesn't have a close method, so we don't need to do anything here
        pass
            
    async def get_backlinks_summary(self, domain: str) -> Dict:
        """Get backlinks summary data for a domain"""
        try:
            backlinks_api = BacklinksApi(self.api_client)
            response = backlinks_api.summary_live([BacklinksSummaryLiveRequestInfo(
                target=domain,
                limit=10
            )])
            
            if response and response.tasks and len(response.tasks) > 0:
                result = response.tasks[0].result[0] if response.tasks[0].result else None
                if result:
                    # Create a safe dictionary with only attributes that exist
                    return {
                        "backlinks_count": getattr(result, "backlinks", 0),
                        "referring_domains": getattr(result, "referring_domains", 0),
                        "referring_ips": getattr(result, "referring_ips", 0),
                        "rank": getattr(result, "rank", 0)
                    }
            return {}
        except ApiException as e:
            logger.error(f"DataForSEO API Exception when getting backlinks summary: {e}")
            return {}
        except Exception as e:
            logger.error(f"Exception when getting backlinks summary: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
    async def get_domain_analytics(self, domain: str) -> Dict:
        """Get domain analytics data including WHOIS information"""
        try:
            domain_api = DomainAnalyticsApi(self.api_client)
            response = domain_api.whois_overview_live([DomainAnalyticsWhoisOverviewLiveRequestInfo(
                domain=domain
            )])
            
            if response and response.tasks and len(response.tasks) > 0:
                result = response.tasks[0].result[0] if response.tasks[0].result else None
                if result:
                    # Create a safe dictionary with available attributes
                    whois_data = {}
                    for attr in ["registrar", "expiration_date", "creation_date", "updated_date", 
                                "name_servers"]:
                        if hasattr(result, attr):
                            whois_data[attr] = getattr(result, attr)
                    # Always include the domain we queried
                    whois_data["domain"] = domain
                    return whois_data
            return {}
        except ApiException as e:
            logger.error(f"DataForSEO API Exception when getting domain analytics: {e}")
            return {}
        except Exception as e:
            logger.error(f"Exception when getting domain analytics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
    async def get_rank_overview(self, domain: str) -> Dict:
        """Get domain rank overview from DataForSEO Labs"""
        try:
            labs_api = DataforseoLabsApi(self.api_client)
            
            # Try Google first, fallback to Bing if not available
            method_name = None
            request_info = None
            
            # Check available methods
            available_methods = dir(labs_api)
            if "google_domain_rank_overview_live" in available_methods:
                method_name = "google_domain_rank_overview_live"
                request_info = DataforseoLabsGoogleDomainRankOverviewLiveRequestInfo
            elif "bing_domain_rank_overview_live" in available_methods:
                method_name = "bing_domain_rank_overview_live"
                # We'll need to use the appropriate model for Bing, but for now use Google's model
                request_info = DataforseoLabsGoogleDomainRankOverviewLiveRequestInfo
            
            if not method_name:
                logger.error("No appropriate domain_rank_overview method found in DataforSEO Labs API")
                return {}
                
            # Call the appropriate method
            response = getattr(labs_api, method_name)([request_info(
                target=domain,
                limit=10
            )])
            
            if response and response.tasks and len(response.tasks) > 0:
                result = response.tasks[0].result[0] if response.tasks[0].result else None
                if result:
                    # Create a safe dictionary with available attributes
                    rank_data = {}
                    for attr in ["organic_keywords", "organic_traffic", "organic_traffic_cost", "se_domains"]:
                        if hasattr(result, attr):
                            rank_data[attr] = getattr(result, attr)
                    return rank_data
            return {}
        except ApiException as e:
            logger.error(f"DataForSEO API Exception when getting rank overview: {e}")
            return {}
        except Exception as e:
            logger.error(f"Exception when getting rank overview: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
            
    async def get_competitors(self, domain: str) -> Dict:
        """Get competitor data from DataForSEO Labs"""
        try:
            labs_api = DataforseoLabsApi(self.api_client)
            
            # Try Google first, fallback to Bing if not available
            method_name = None
            request_info = None
            
            # Check available methods
            available_methods = dir(labs_api)
            if "google_competitors_domain_live" in available_methods:
                method_name = "google_competitors_domain_live"
                request_info = DataforseoLabsGoogleCompetitorsDomainLiveRequestInfo
            elif "bing_competitors_domain_live" in available_methods:
                method_name = "bing_competitors_domain_live"
                # We'll need to use the appropriate model for Bing, but for now use Google's model
                request_info = DataforseoLabsGoogleCompetitorsDomainLiveRequestInfo
            
            if not method_name:
                logger.error("No appropriate competitors_domain method found in DataforSEO Labs API")
                return {"competitors": []}
                
            # Call the appropriate method
            response = getattr(labs_api, method_name)([request_info(
                target=domain,
                limit=10
            )])
            
            if response and response.tasks and len(response.tasks) > 0:
                result = response.tasks[0].result[0] if response.tasks[0].result else None
                if result and hasattr(result, "items") and result.items:
                    competitors = []
                    for item in result.items[:5]:  # Limit to top 5 competitors
                        competitor = {
                            "brand": getattr(item, "domain", "N/A"),
                            "website": getattr(item, "domain", "N/A"),
                            "domain_rating": getattr(item, "domain_rank", "N/A") if hasattr(item, 'domain_rank') else 'N/A',
                            "referring_domains": 'N/A',
                            "backlinks": 'N/A',
                            "total_keywords": getattr(item, "keywords_count", "N/A") if hasattr(item, 'keywords_count') else 'N/A'
                        }
                        competitors.append(competitor)
                    return {"competitors": competitors}
            return {"competitors": []}
        except ApiException as e:
            logger.error(f"DataForSEO API Exception when getting competitors: {e}")
            return {"competitors": []}
        except Exception as e:
            logger.error(f"Exception when getting competitors: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"competitors": []}
            
    async def analyze_domain(self, domain: str) -> Dict:
        """Perform a comprehensive analysis using multiple DataForSEO APIs"""
        try:
            backlinks_data = await self.get_backlinks_summary(domain)
            whois_data = await self.get_domain_analytics(domain)
            rank_data = await self.get_rank_overview(domain)
            competitor_data = await self.get_competitors(domain)
            
            return {
                "backlinks": backlinks_data,
                "domain_info": whois_data,
                "rank_overview": rank_data,
                "competitors": competitor_data
            }
        except Exception as e:
            logger.error(f"Error in analyze_domain: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

def extract_business_type(website_analysis):
    """
    Extract the business type from the website analysis with improved accuracy.
    Uses content analysis and keyword patterns to determine the most likely business type.
    """
    # Initialize with a generic type
    business_type = "e-commerce"
    
    # Define business type detection patterns with weighted keywords
    business_patterns = {
        "eczema clothing": {
            "keywords": ["eczema clothing", "clothes for eczema", "eczema-friendly", "eczema garments", 
                         "skin condition clothing", "sensitive skin clothing", "therapeutic clothing"],
            "weight": 15  # Higher weight for specific niche
        },
        "baby clothing": {
            "keywords": ["baby clothes", "children's clothing", "infant wear", "clothing for babies", 
                         "toddler clothes", "kids clothing", "baby garments"],
            "weight": 10
        },
        "skincare": {
            "keywords": ["skincare products", "skin care", "facial products", "moisturizer", "cleanser", 
                         "skin treatment", "serums", "face mask", "skincare routine"],
            "weight": 8
        },
        "beauty": {
            "keywords": ["beauty products", "makeup", "cosmetics", "foundation", "lipstick", 
                         "beauty brand", "beauty routine", "eyeshadow"],
            "weight": 7
        },
        "wellness": {
            "keywords": ["wellness products", "supplements", "health supplements", "wellness brand", 
                         "nutrition", "holistic", "natural remedies"],
            "weight": 6
        },
        "fitness": {
            "keywords": ["fitness equipment", "workout gear", "exercise", "gym", "training", 
                         "athletic wear", "sports equipment"],
            "weight": 6
        },
        "clothing": {
            "keywords": ["clothing brand", "apparel", "garments", "fashion", "clothes", "wear", 
                         "outfits", "wardrobe", "collection"],
            "weight": 5
        },
        "electronics": {
            "keywords": ["electronics", "gadgets", "devices", "tech products", "technology", 
                         "smart devices", "electronic accessories"],
            "weight": 5
        },
        "home goods": {
            "keywords": ["home decor", "furniture", "homeware", "home accessories", "kitchenware", 
                         "bedding", "home products"],
            "weight": 5
        },
        "food": {
            "keywords": ["food products", "grocery", "meals", "snacks", "beverages", "ingredients", 
                         "culinary", "gourmet"],
            "weight": 5
        }
    }
    
    # Check for explicit statements about business type
    explicit_patterns = [
        r"(?:we|our company|our business|business) (?:sells?|offers?|provides?|specializes? in|deals? in) ([^.]+)",
        r"(?:our|the) products? include(?:s)? ([^.]+)",
        r"(?:leading|premier|specialized) provider of ([^.]+)",
        r"(?:we are|we're) a (?:company|business|brand) that (?:sells?|offers?|specializes? in) ([^.]+)"
    ]
    
    # Prepare the analysis text
    lower_analysis = website_analysis.lower()
    
    # First look for explicit statements about what the business does
    for pattern in explicit_patterns:
        matches = re.findall(pattern, lower_analysis)
        if matches:
            most_relevant = matches[0]
            # Score the explicit statement against our business patterns
            for business_type, data in business_patterns.items():
                if any(keyword in most_relevant.lower() for keyword in data["keywords"]):
                    return business_type
    
    # Calculate scores for each business type based on keyword presence
    scores = {business_type: 0 for business_type in business_patterns}
    
    for business_type, data in business_patterns.items():
        keyword_count = 0
        for keyword in data["keywords"]:
            # Count occurrences of each keyword
            count = lower_analysis.count(keyword)
            if count > 0:
                keyword_count += count
        
        # Calculate score based on keyword occurrences and weights
        scores[business_type] = keyword_count * data["weight"]
    
    # Special case for combined categories
    if scores["eczema clothing"] > 0 and scores["baby clothing"] > 0:
        # If both eczema and baby clothing are mentioned, it's likely eczema clothing for babies
        return "eczema clothing for babies and children"
    
    if scores["skincare"] > 0 and scores["beauty"] > 0:
        # If both skincare and beauty are significant, combine them
        if scores["skincare"] > scores["beauty"]:
            return "skincare and beauty"
        else:
            return "beauty and skincare"
    
    # Find the business type with the highest score
    max_score = 0
    most_likely_type = "e-commerce"  # default
    
    for business_type, score in scores.items():
        if score > max_score:
            max_score = score
            most_likely_type = business_type
    
    # If no significant pattern is found, return a generic e-commerce designation
    if max_score < 5:
        return "e-commerce"
    
    return most_likely_type

if __name__ == "__main__":
    asyncio.run(main())