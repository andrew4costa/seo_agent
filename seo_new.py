import asyncio
import aiohttp
from typing import List, Dict, Set, Optional, Any
from bs4 import BeautifulSoup # Not used in the snippet, but kept if used elsewhere
from urllib.parse import urljoin, urlparse # Not used in the snippet, but kept
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import os
import re
import configparser # Not actively used in provided snippets, but often for config
import pathlib
from aiohttp import ClientError # Not explicitly used but good for type hinting aiohttp errors
import tldextract # For domain parsing
from dotenv import load_dotenv

import seo_analyzer # Import our new SEO analyzer module
# The LLMProcessor and save_html_report are imported from a file in test_scripts,
# which is unusual for production code but we'll keep it as is for now.
from test_scripts.seo_fixed import LLMProcessor, save_report, save_html_report

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seo_agent")

# Constants
CONFIG_FILE = pathlib.Path.home() / ".seo_agent_config.ini" # Example, not actively used in snippets
REPORTS_DIR = pathlib.Path("/Users/andrewcosta/Desktop/seo_agent/reports") # Ensure this path is correct or configurable

def _attempt_json_load_and_fix(text_to_parse: str, logger_prefix: str = "JSON extraction") -> Optional[Any]:
    """Helper to try json.loads and log failures."""
    try:
        return json.loads(text_to_parse)
    except json.JSONDecodeError as e:
        logger.warning(f"{logger_prefix}: Parse attempt failed: {e}. Text (first 300): {text_to_parse[:300]}")
        return None

def extract_json_from_text(text: str) -> Optional[Any]: # Can return list or dict
    logger.debug(f"Attempting to extract JSON. Raw LLM response (first 500 chars):\n{text[:500]}")

    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("JSON extraction: Empty or invalid input text.")
        return None

    # 1. Remove markdown code blocks and strip whitespace
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
    else:
        cleaned_text = text.strip()
    logger.debug(f"JSON extraction: Text after markdown stripping (first 300): {cleaned_text[:300]}")

    if not cleaned_text:
        logger.warning("JSON extraction: Text is empty after markdown stripping.")
        return None

    # Attempt 1: Direct parsing of cleaned_text
    if (cleaned_text.startswith('{') and cleaned_text.endswith('}')) or \
       (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
        loaded_json = _attempt_json_load_and_fix(cleaned_text, "Direct parse of cleaned text")
        if loaded_json is not None:
            return loaded_json

    # Attempt 2: Try to fix common LLM errors from the cleaned_text and then parse
    # Remove trailing commas before closing brackets/braces
    # Handles cases like: [1, 2,] or { "a":1, }
    fixed_text = re.sub(r",\s*(\}|\])", r"\1", cleaned_text)
    
    # Replace Python-style None/True/False with JSON null/true/false
    # Use word boundaries to avoid replacing "None" in "NoneSuch"
    fixed_text = re.sub(r'\bNone\b', 'null', fixed_text)
    fixed_text = re.sub(r'\bTrue\b', 'true', fixed_text)
    fixed_text = re.sub(r'\bFalse\b', 'false', fixed_text)

    # Note: Fixing unescaped quotes within strings is complex and error-prone with regex.
    # It's often better to improve prompts or handle specific errors if they persist.
    # If the above fixes are applied, try parsing this `fixed_text` if it looks like JSON
    if (fixed_text.startswith('{') and fixed_text.endswith('}')) or \
       (fixed_text.startswith('[') and fixed_text.endswith(']')):
        logger.debug(f"JSON extraction: Attempting parse after basic fixes. Text (first 300): {fixed_text[:300]}")
        loaded_json = _attempt_json_load_and_fix(fixed_text, "Parse after basic fixes")
        if loaded_json is not None:
            return loaded_json

    # Attempt 3: Fallback to finding first '{' or '[' to last '}' or ']' on the *original cleaned_text*
    # This helps if there was still some minimal surrounding text not caught by markdown stripping.
    # This attempt is made on `cleaned_text` which had markdown removed but not other fixes,
    # in case those fixes were too aggressive for some structures.
    if not ((cleaned_text.startswith('{') and cleaned_text.endswith('}')) or \
            (cleaned_text.startswith('[') and cleaned_text.endswith(']'))):
        
        patterns_to_try = [('{', '}'), ('[', ']')] # Object first, then array
        for open_char, close_char in patterns_to_try:
            first_idx = cleaned_text.find(open_char)
            last_idx = cleaned_text.rfind(close_char)
            if first_idx != -1 and last_idx > first_idx:
                potential_json_substring = cleaned_text[first_idx : last_idx+1]
                logger.debug(f"JSON extraction: Trying substring parse ('{open_char}'...'{close_char}'). Substring (first 300): {potential_json_substring[:300]}")
                loaded_json = _attempt_json_load_and_fix(potential_json_substring, f"Substring parse ('{open_char}'...'{close_char}')")
                if loaded_json is not None:
                    return loaded_json

    logger.error(f"JSON extraction: FAILED. Could not extract valid JSON. Original (first 300): {text[:300]} Processed (first 300): {cleaned_text[:300]}")
    return None

async def retry_llm_request(llm_processor: LLMProcessor, messages: List[Dict], retry_count: int = 3, temperature: float = 0.5) -> Optional[Any]:
    logger.info(f"Requesting from LLM with retry mechanism (temp: {temperature}, retries: {retry_count})...")
    # ... (rest of retry_llm_request, ensure it can return dict or list) ...
    for attempt in range(retry_count):
        try:
            adjusted_temp = max(0.1, temperature - (attempt * 0.1))
            response = await llm_processor._make_api_request(messages, temperature=adjusted_temp) # Assuming this is the correct method name
            content = response['choices'][0]['message']['content']
            
            json_data = extract_json_from_text(content) # Use the improved extractor
            if json_data is not None:
                logger.info(f"Successfully extracted JSON from LLM (attempt {attempt+1})")
                return json_data 
            else:
                logger.warning(f"Failed to extract JSON on attempt {attempt+1}. Content: {content[:200]}...")
                if attempt < retry_count -1 and messages and messages[0]["role"] == "system":
                         messages[0]["content"] += "\nCRITICAL: Your entire response MUST be a single, valid JSON object or array. No other text is allowed."
            
        except Exception as e:
            logger.error(f"Error in retry_llm_request attempt {attempt+1}: {str(e)}", exc_info=True)
            if attempt >= retry_count - 1:
                logger.error("All LLM retry attempts failed.")
                # Depending on context, might raise or return a specific error object/None
                return None # Or raise e
    return None

async def analyze_architecture(url: str, domain: str, analysis_data: Dict, llm_processor: LLMProcessor) -> List[Dict]:
    logger.info(f"Analyzing site architecture for {domain}...")
    pages = analysis_data.get("analyzed_pages", {})
    page_urls = list(pages.keys())
    sitemap_exists = analysis_data.get("technical_seo", {}).get("indexation", {}).get("sitemap", {}).get("exists", False)
    sitemap_data = analysis_data.get("technical_seo", {}).get("indexation", {}).get("sitemap", {})
    sitemap_url = sitemap_data.get("url", "")
    sitemap_entry_count = sitemap_data.get("entry_count", 0)
    sitemap_issues = []
    if sitemap_exists:
        if sitemap_entry_count > 0 and len(pages) > 0 and sitemap_entry_count < len(pages) * 0.8: 
            sitemap_issues.append(f"Sitemap contains {sitemap_entry_count} URLs, but {len(pages)} pages were found during analysis.")
        if sitemap_data.get("errors", []): # Ensure this key exists or default
            sitemap_issues.append(f"Sitemap has errors: {', '.join(sitemap_data.get('errors', []))}")
        if sitemap_data.get("warnings", []): # Ensure this key exists or default
            sitemap_issues.append(f"Sitemap has warnings: {', '.join(sitemap_data.get('warnings', []))}")
            
    hierarchy = {} # Simplified placeholder
    categories = [] # Simplified placeholder
    url_patterns = page_urls[:10] # Simplified placeholder
    internal_links = {url: data.get("links", {}).get("internal", []) for url, data in pages.items()} # Simplified

    architecture_prompt = f"""
    Analyze the website architecture and URL structure for {url} based on the following data:
    URL PATTERNS (sample of {min(10, len(url_patterns))} patterns): {json.dumps(url_patterns, indent=2)}
    PAGES ANALYZED: {len(pages)}
    SITEMAP DATA:
    - Sitemap.xml exists: {sitemap_exists}
    {(f"- Sitemap URL: {sitemap_url}\\n- Sitemap entry count: {sitemap_entry_count}\\n- Potential sitemap issues: {sitemap_issues if sitemap_issues else 'None detected'}" if sitemap_exists else "")}
    
    Based on this architectural data, provide recommendations for: URL Structure, Information Architecture, Navigation, Content Categorization, Mobile Architecture.
    For each recommendation: title, detailed explanation, priority (1-10), difficulty, estimated hours, category (must be ARCHITECTURE).
    IMPORTANT SITEMAP INSTRUCTIONS:
    - If NO sitemap exists, recommend creating one.
    - If a sitemap EXISTS but has issues, recommend IMPROVING the existing sitemap.
    - If a sitemap EXISTS and seems complete, focus on other architectural improvements.
    """
    system_message = """You are an expert in website architecture and SEO. Return recommendations in JSON array format: [{"recommendation": "Title", "reasoning": "Explanation", "priority_score": 8, "impact": "High", "implementation_difficulty": "Medium", "estimated_time_hours": 4, "category": "ARCHITECTURE"}]. Follow SITEMAP INSTRUCTIONS."""
    architecture_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": architecture_prompt}]
    logger.info(f"Requesting architecture analysis from LLM...")
    try:
        response_data = await retry_llm_request(llm_processor, architecture_messages, retry_count=2, temperature=0.3)
        architecture_recommendations = []
        if isinstance(response_data, list): architecture_recommendations = response_data
        elif isinstance(response_data, dict) and "recommendations" in response_data and isinstance(response_data["recommendations"], list):
            architecture_recommendations = response_data["recommendations"]
        else: logger.warning("Architecture LLM response not a list. Using default.")

        if not architecture_recommendations: return create_default_architecture_recommendations(sitemap_exists, sitemap_issues)
        
        filtered_recommendations = []
        for rec in architecture_recommendations:
            if isinstance(rec, dict):
                rec_text = rec.get("recommendation", "").lower()
                if sitemap_exists and "create" in rec_text and "sitemap" in rec_text and not any(w in rec_text for w in ["improve","enhance","update","fix"]):
                    if sitemap_issues:
                        rec["recommendation"] = f"Improve existing sitemap (Issues: {', '.join(sitemap_issues[:1])})"
                        rec["reasoning"] = f"Existing sitemap issues: {', '.join(sitemap_issues)}. Improve for better indexing."
                    else: continue # Skip create sitemap if exists and no issues
                rec["category"] = "ARCHITECTURE"
                filtered_recommendations.append(rec)
        logger.info(f"Generated {len(filtered_recommendations)} architecture recommendations")
        return filtered_recommendations
    except Exception as e:
        logger.error(f"Error in architecture analysis: {str(e)}", exc_info=True)
        return create_default_architecture_recommendations(sitemap_exists, sitemap_issues)

def create_default_architecture_recommendations(sitemap_exists=False, sitemap_issues=None) -> List[Dict]:
    if sitemap_issues is None: sitemap_issues = []
    recs = [{"recommendation": "Improve URL structure", "priority_score": 8, "category": "ARCHITECTURE", "reasoning": "Logical URLs help SEO."}]
    if not sitemap_exists: recs.append({"recommendation": "Create XML sitemap", "priority_score": 8, "category": "ARCHITECTURE", "reasoning": "Sitemaps aid crawling."})
    elif sitemap_issues: recs.append({"recommendation": f"Improve sitemap (Issues: {sitemap_issues})", "priority_score": 7, "category": "ARCHITECTURE", "reasoning": "Fix sitemap issues."})
    recs.append({"recommendation": "Enhance internal linking", "priority_score": 7, "category": "ARCHITECTURE", "reasoning": "Internal links distribute authority."})
    return recs

async def analyze_ux(url: str, domain: str, analysis_data: Dict, llm_processor: LLMProcessor) -> List[Dict]:
    logger.info(f"Analyzing UX and page design for {domain}...")
    pages = analysis_data.get("analyzed_pages", {})
    page_urls = list(pages.keys())
    ux_metrics = {
        "mobile_friendly": analysis_data.get("technical_seo", {}).get("mobile_optimisation", {}).get("has_viewport_meta", False),
        "mobile_score": analysis_data.get("technical_seo", {}).get("mobile_optimisation", {}).get("score", 0),
        "avg_load_time": analysis_data.get("technical_seo", {}).get("site_speed", {}).get("avg_load_time", 0),
    }
    ux_prompt = f"""Analyze UX for {url} based on: Mobile Friendly: {ux_metrics['mobile_friendly']}, Mobile Score: {ux_metrics['mobile_score']}/10, Avg Load: {ux_metrics['avg_load_time']:.2f}s. Sample URLs: {json.dumps(page_urls[:3])}. Provide UX/design recommendations (title, explanation, priority 1-10, difficulty, hours, category UX_DESIGN)."""
    system_message = """You are a UX expert. Analyze for UX recommendations. JSON array format: [{"recommendation": ..., "reasoning": ..., "category": "UX_DESIGN", ...}]."""
    ux_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": ux_prompt}]
    logger.info(f"Requesting UX analysis from LLM...")
    try:
        response_data = await retry_llm_request(llm_processor, ux_messages, retry_count=2, temperature=0.3)
        ux_recommendations = []
        if isinstance(response_data, list): ux_recommendations = response_data
        elif isinstance(response_data, dict) and "recommendations" in response_data and isinstance(response_data["recommendations"], list):
            ux_recommendations = response_data["recommendations"]
        else: logger.warning("UX LLM response not a list. Using default.")
        
        if not ux_recommendations: return create_default_ux_recommendations()
        for rec in ux_recommendations:
            if isinstance(rec, dict): rec["category"] = "UX_DESIGN"
        logger.info(f"Generated {len(ux_recommendations)} UX recommendations")
        return ux_recommendations
    except Exception as e:
        logger.error(f"Error during UX analysis: {str(e)}", exc_info=True)
        return create_default_ux_recommendations()

def create_default_ux_recommendations() -> List[Dict]:
    return [{"recommendation": "Perform professional UX assessment", "priority_score": 7, "category": "UX_DESIGN", "reasoning": "A UX review can identify key improvements."}]

async def analyze_competitors(url: str, domain: str, website_analysis: str, llm_processor: LLMProcessor) -> Dict:
    logger.info(f"Analyzing REAL competitors for {domain} using LLM...")
    # ... (Existing LLM-based competitor analysis logic remains, as it doesn't use DataForSEO)
    # Ensure this function returns a Dict: {"competitors": [...]}
    def _filter_competitors(competitors_list): # Corrected variable name
        actual_competitors = []
        for comp in competitors_list: # Iterate over the list
            brand_name = comp.get("brand", "").lower()
            if "unknown" not in brand_name and "competitor" not in brand_name and "placeholder" not in brand_name and brand_name:
                actual_competitors.append(comp)
        return actual_competitors

    competitors_prompt = f"""Based on business analysis for {url}: {website_analysis}. Identify 3-5 REAL competitor brands. For each: Brand Name, Competitor Website, ESTIMATE DR (0-100 or N/A), Referring Domains (N/A if unknown), Backlinks (N/A if unknown), Total Keywords (N/A if unknown). IMPORTANT: Only REAL brands. JSON: {{"competitors": [{{"brand": ..., "website": ..., etc.}}]}} or {{"competitors": []}}."""
    system_message = """You are an SEO/competitive analysis expert. Identify REAL competitors. JSON format ONLY. Use N/A for unknown metrics. Empty list if no real competitors."""
    
    competitors_data = {"competitors": []} # Default
    for attempt in range(2): # Try twice (e.g. strict then lenient, or just retry)
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": competitors_prompt}]
        logger.info(f"Requesting competitor analysis from LLM (attempt {attempt+1})...")
        try:
            response_json = await retry_llm_request(llm_processor, messages, retry_count=1, temperature=0.2) # Only 1 retry inside retry_llm_request
            if response_json and "competitors" in response_json and isinstance(response_json["competitors"], list):
                filtered = _filter_competitors(response_json["competitors"])
                if filtered:
                    competitors_data["competitors"] = filtered
                    logger.info(f"Found {len(filtered)} real competitors.")
                    return competitors_data
                else:
                    logger.info(f"LLM returned competitors but all were filtered on attempt {attempt +1}.")
            else:
                logger.warning(f"Invalid competitor data from LLM on attempt {attempt+1}: {str(response_json)[:100]}")
        except Exception as e:
            logger.error(f"Error in analyze_competitors LLM call (attempt {attempt+1}): {e}")
        # Modify prompt for a more lenient second attempt if needed (not explicitly shown here for brevity)
    logger.warning("Competitor analysis failed to find real competitors after all attempts.")
    return create_default_competitor_data(domain)


def create_default_competitor_data(domain: str, is_fallback: bool = False, count: int = 0) -> Dict:
    return {"competitors": []}

async def analyze_business_context(url: str, domain: str, analysis_data: Dict, llm_processor: LLMProcessor) -> str:
    logger.info(f"Performing detailed business context analysis for {domain}...")
    pages = analysis_data.get("analyzed_pages", {})
    titles = [p.get("title", "") for p in pages.values()]
    descriptions = [p.get("meta_description", "") for p in pages.values()]
    page_content_samples = [p.get("content", "")[:500] for p in pages.values()] # Sample content

    context_prompt = f"""Analyze ACTUAL business type of {url}. AVOID domain name assumptions (e.g. happy-skin.com could be eczema clothing, not skincare). Base ONLY on evidence from:
    PAGE TITLES: {json.dumps(titles[:5])}
    META DESCRIPTIONS: {json.dumps(descriptions[:5])}
    CONTENT SAMPLES: {json.dumps(page_content_samples[:2])}
    Determine: 1. SPECIFIC business type/industry & ACTUAL products/services. 2. Target audience & problem solved. 3. Unique selling proposition/brand values.
    CRITICAL: If evidence for eczema clothing for babies/children, state explicitly.
    """
    system_message = """You are an expert business analyst. Determine company's actual business model from website content. IGNORE domain name assumptions. Be SPECIFIC (e.g., "organic skincare for sensitive skin", NOT "beauty products"). If evidence is unclear, state that. For Happy-Skin.com - if it's baby eczema clothing, state it!"""
    
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": context_prompt}]
    try:
        # Business context doesn't strictly need to be JSON, a descriptive string is fine.
        response = await llm_processor._make_api_request(messages, temperature=0.2)
        business_context = response['choices'][0]['message']['content']
        
        # Simplified verification for happy-skin.com
        if "happy-skin.com" in url.lower() and "skincare" in business_context.lower() and "clothing" not in business_context.lower():
            business_context += "\nVERIFICATION NOTE: For happy-skin.com, ensure this is not primarily eczema clothing for babies/children."
        
        logger.info(f"Completed detailed business context analysis for {domain}")
        return business_context
    except Exception as e:
        logger.error(f"Error during business context analysis: {str(e)}", exc_info=True)
        return f"Failed to analyze business context for {domain}. Error: {e}"

async def main_example(url_to_analyze: str, gemini_api_key_arg: Optional[str] = None, max_pages: int = 5, architecture_recs: Optional[List[Dict]] = None, progress_callback=None):
    """Run the SEO analysis with Gemini-powered recommendations (DataForSEO integration removed)"""
    if not url_to_analyze.startswith(('http://', 'https://')):
        url_to_analyze = 'https://' + url_to_analyze
    domain = urlparse(url_to_analyze).netloc or url_to_analyze
        
    effective_gemini_key = gemini_api_key_arg or os.getenv("GEMINI_API_KEY")
    
    if not effective_gemini_key and not os.getenv("TOGETHER_API_KEY") and not (os.getenv("USE_LOCAL_LLM", "false").lower() == "true"):
        logger.error("No suitable LLM API key found or local LLM not configured.")
        return {"error": "LLM configuration missing", "analysis": {}, "recommendations": [], "competitor_data": {}, "dataforseo_data": {}, "report_path": None}

    async with LLMProcessor(api_key=effective_gemini_key) as llm_processor:
        if effective_gemini_key: # Ensure provider is set if using Gemini
            llm_processor.llm_provider = "gemini"
            llm_processor.gemini_api_key = effective_gemini_key
        
        try:
            if progress_callback: progress_callback("Starting technical analysis", 0.1)
            logger.info(f"Starting direct technical SEO analysis of {domain}...")
            analysis_data = await seo_analyzer.analyze_url(url_to_analyze, max_pages=max_pages)
            if progress_callback: progress_callback("Technical analysis complete", 0.3)
            
            technical_score = analysis_data.get("technical_seo", {}).get("score", 0)
            onpage_score = analysis_data.get("on_page_seo", {}).get("score", 0)
            
            dataforseo_data = {} # Initialize as empty dict as DataForSEO is removed
            logger.info("DataForSEO analysis skipped (integration removed).")
            if progress_callback: progress_callback("DataForSEO check skipped", 0.5)
            
            logger.info(f"Getting accurate business context for {domain}...")
            website_analysis = await analyze_business_context(url_to_analyze, domain, analysis_data, llm_processor)
            if progress_callback: progress_callback("Business context analysis complete", 0.6)
            
            arch_recommendations = []
            if architecture_recs is None:
                logger.info(f"Analyzing site architecture...")
                arch_recommendations = await analyze_architecture(url_to_analyze, domain, analysis_data, llm_processor)
            # else: arch_recommendations = architecture_recs # If pre-defined used
            if progress_callback: progress_callback("Architecture analysis complete", 0.7)
            
            logger.info(f"Analyzing UX and page design...")
            ux_recommendations = await analyze_ux(url_to_analyze, domain, analysis_data, llm_processor)
            if progress_callback: progress_callback("UX analysis complete", 0.8)
            
            logger.info(f"Using LLM-based competitor analysis...")
            competitor_data = await analyze_competitors(url_to_analyze, domain, website_analysis, llm_processor)
            if progress_callback: progress_callback("Competitor analysis complete", 0.9)
                
            logger.info(f"Generating tailored SEO recommendations...")
            if progress_callback: progress_callback("Generating recommendations", 0.95)
            
            recommendations_prompt = f"""
            Technical SEO for {url_to_analyze} (Tech Score: {technical_score:.1f}, On-Page Score: {onpage_score:.1f}):
            Load Time: {analysis_data.get("technical_seo", {}).get("site_speed", {}).get("avg_load_time", 0):.2f}s, SSL: {analysis_data.get("technical_seo", {}).get("security", {}).get("has_ssl", False)}, Mobile Score: {analysis_data.get("technical_seo", {}).get("mobile_optimisation", {}).get("score", 0):.1f}/10.
            Content: Avg Length {analysis_data.get("on_page_seo", {}).get("content_quality", {}).get("avg_word_count", 0):.0f}, Missing Titles: {analysis_data.get("on_page_seo", {}).get("meta_tags", {}).get("pages_without_title", 0)}.
            BUSINESS CONTEXT: {website_analysis}
            Generate tailored SEO recommendations for {domain} covering Technical, On-Page, Content, and strategic insights based on business type.
            For each recommendation: "recommendation" (title), "reasoning" (DETAILED, SITE-SPECIFIC), "priority_score" (1-10), "impact" (high/medium/low), "implementation_difficulty" (easy/medium/hard), "estimated_time_hours", "category".
            NO GENERIC ADVICE. Base on ACTUAL findings and business.
            """
            system_message = """You are an elite SEO consultant. Response MUST be VALID JSON ONLY: [{"recommendation": ..., "category": ..., etc.}]. Provide HIGHLY CUSTOMIZED recommendations with specific reasoning based on analysis and business type."""
            
            recommendation_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": recommendations_prompt}]
            recommendations_list = await retry_llm_request(llm_processor, recommendation_messages, retry_count=3, temperature=0.5)
            
            if not recommendations_list or not isinstance(recommendations_list, list):
                logger.warning("Failed to get valid list of recommendations. Using default.")
                recommendations_list = create_default_recommendations(website_analysis, domain)
            
            # Standardize and extend recommendations
            final_recommendations = []
            for rec in recommendations_list:
                if isinstance(rec, dict):
                    final_recommendations.append({
                        "recommendation": rec.get("description", rec.get("recommendation", "N/A")),
                        "category": rec.get("category", "General"),
                        "priority_score": int(rec.get("priority_score", rec.get("priority", 5))),
                        "impact": rec.get("impact", "medium"),
                        "implementation_difficulty": rec.get("implementation_difficulty", rec.get("difficulty", "medium")),
                        "estimated_time_hours": int(rec.get("estimated_time_hours", rec.get("time", 2))),
                        "reasoning": rec.get("reasoning", "Default: Address for improved SEO.")
                    })
            if isinstance(arch_recommendations, list): final_recommendations.extend(arch_recommendations)
            if isinstance(ux_recommendations, list): final_recommendations.extend(ux_recommendations)
            
            # Fallback reasoning if missing
            business_type_extracted = extract_business_type(website_analysis)
            for rec in final_recommendations:
                if not rec.get("reasoning") or len(rec.get("reasoning", "").strip()) < 20:
                     rec["reasoning"] = f"This {rec.get('category','General').lower()} task for {domain} ({business_type_extracted}) addresses findings from the analysis to boost performance."
            
            # Ensure analysis_data has a basic off_page_seo structure if not present
            if "off_page_seo" not in analysis_data:
                analysis_data["off_page_seo"] = {
                    "social_signals": {"status": "Not analyzed in this version"},
                    "other_external_factors": {"status": "Not analyzed in this version"}
                }

            # Add competitor data to analysis_data if it needs to be in the HTML report via this dict
            analysis_data['competitors'] = competitor_data # Assuming save_html_report might look for it here

            report_path = save_html_report(
                url_to_analyze, 
                final_recommendations, 
                analysis_data, # This now contains all data including competitor info if structured above
                domain.capitalize()
                # Removed competitor_data and dataforseo_data as separate arguments
            )
            logger.info(f"SEO analysis complete for {domain}. Report saved to: {report_path}")
            if progress_callback: progress_callback("Analysis complete", 1.0)
            
            return {
                "analysis": analysis_data,
                "recommendations": final_recommendations,
                "competitor_data": competitor_data, # Still return for structured data, even if not passed to save_html_report separately
                "dataforseo_data": {}, # Return empty dict for consistent structure
                "report_path": report_path
            }
        except Exception as e:
            logger.error(f"Error during SEO analysis for {url_to_analyze}: {str(e)}", exc_info=True)
            return {"error": str(e), "analysis": {}, "recommendations": [], "competitor_data": {}, "dataforseo_data": {}, "report_path": None}

def format_number(number):
    if not isinstance(number, (int, float)): return str(number)
    if number >= 1000: return f"{number:,}"
    return str(number)

def create_default_recommendations(content: str, domain: str) -> List[Dict]:
    business_type = extract_business_type(content)
    return [
        {"recommendation": "Improve URL Structure", "reasoning": f"For {domain} ({business_type}), clean URLs aid UX & SEO.", "category": "Technical", "priority_score": 9, "impact": "High", "implementation_difficulty": "Medium", "estimated_time_hours": 6},
        {"recommendation": "Enhance Site Architecture", "reasoning": f"Good architecture for {domain} ({business_type}) improves navigation & reduces bounce.", "category": "ARCHITECTURE", "priority_score": 8, "impact": "High", "implementation_difficulty": "Medium", "estimated_time_hours": 12},
        {"recommendation": "Optimize Mobile Experience", "reasoning": f"Mobile optimization for {domain} ({business_type}) is critical as many customers use mobile.", "category": "UX_DESIGN", "priority_score": 9, "impact": "High", "implementation_difficulty": "Medium", "estimated_time_hours": 8}
    ]

def extract_business_type(website_analysis_text: str): # Renamed for clarity
    # Simplified version for brevity
    if not website_analysis_text or not isinstance(website_analysis_text, str): return "e-commerce"
    lower_analysis = website_analysis_text.lower()
    if "eczema clothing" in lower_analysis and ("baby" in lower_analysis or "children" in lower_analysis):
        return "eczema clothing for babies and children"
    if "clothing" in lower_analysis: return "clothing"
    if "skincare" in lower_analysis: return "skincare"
    return "e-commerce" # Default

async def main(): # For command-line execution
    parser = argparse.ArgumentParser(description="SEO Analyzer (DataForSEO removed)")
    parser.add_argument("url", help="URL to analyze")
    parser.add_argument("--key", help="Gemini API key (uses env var GEMINI_API_KEY if not provided)")
    parser.add_argument("--pages", type=int, default=5, help="Max pages to analyze (default: 5)")
    parser.add_argument("--architecture-file", help="Path to JSON with architecture recommendations")
    
    args = parser.parse_args()
    architecture_recs = None
    if args.architecture_file:
        try:
            if os.path.exists(args.architecture_file):
                with open(args.architecture_file, 'r') as f:
                    architecture_recs = json.load(f)
            else: logger.warning(f"Architecture file not found: {args.architecture_file}")
        except Exception as e: logger.error(f"Error loading architecture file: {e}")
    
    try:
        # Call main_example without DataForSEO args
        results = await main_example(args.url, gemini_api_key_arg=args.key, max_pages=args.pages, architecture_recs=architecture_recs)
        if results.get("error"):
            logger.error(f"Analysis failed: {results['error']}")
        elif results.get("report_path"):
            logger.info(f"CLI Analysis successful. Report at: {results['report_path']}")
        else:
            logger.warning("CLI Analysis completed but no report path or error returned.")
    except Exception as e:
        logger.error(f"Error in CLI main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())