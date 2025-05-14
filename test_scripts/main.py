import asyncio
import aiohttp
from typing import List, Dict, Set
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime


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

class LLMProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key,
        self.api_url = "https://api.openai.com/v1/chat/completions",
        self.logger = logging.getLogger(__name__)
        self.headers = {
            "Authorisation": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

@dataclass
class LLMProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.logger = logging.getLogger(__name__)
        self.headers = {
            "Authorisation": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.default_model = "gpt-4-turbo-preview"
    
    async def _make_api_request(self, messages: List[Dict[str, str]], temperature: float=0.7) -> Dict:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.default_model,
                    "messages": messages,
                    "temperature": temperature
                }
            
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    self.logger.error(f"API request failed with status {response.status}")
                    raise Exception(f"API request failed: {await response.text()}")
                
                return await response.json()
        
        except Exception as e:
            self.logger.error(f"Error making API request: {str(e)}")
            raise

    async def analyse_content(self, prompt: str) -> Dict[str, any]:
        messages = [
            {"role": "system", "content": "You are an expert SEO analyst. Provide detailed, structured analysis in JSON format."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._make_api_request(messages, temperature=0.3)
            content = response['choices'][0]['message']['content']

            try:
                analysis = json.loads(content)
                return analysis
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM response as JSON")
                return {
                    "error": "Invalid JSON response",
                    "raw_content": content
                }
        
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def generate_recommendations(self, prompt: str) -> List[Dict]:
        messages = [
            {
                "role": "system",
                "content": """You are an expert SEO consultant. Generate specific, actionable recommendations 
                in JSOn format. Each recommendation should include:
                - recommendation: detailed description
                - priority_score: 1-10
                - impact: high/medium/low
                - implementation_difficulty: easy/medium/hard
                -estimated_time: in hours"""
            },
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self._make_api_request(messages, temperature=0.4)
            content = response['choices'][0]['message']['content']
            try:
                recommendations = json.loads(content)
                if isinstance(recommendations, list):
                    return recommendations
                else:
                    return [recommendations]
            
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM recommendations as JSON")
                return [{
                    "error": "Invalid JSON response",
                    "raw_content": content
                }]
    
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return [{"error": str(e)}]
    
    async def validate_json_response(self, content: str) -> Dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
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


class ComprehensiveSEOAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = LLMProcessor(api_key)
        self.logger = logging.getLogger(__name__)

    async def analyse_website(self, url: str) -> Dict:
        results = {
            "technical_seo": await self.analyse_technical_seo(url),
            "on_page_seo": await self.analyse_on_page_seo(url),
            "off_page_seo": await self.analyse_off_page_seo(url),
            "summary": {},
            "recommendations": []        
        }

        results["summary"] = await self._generate_summary(results)
        results["recommendations"] = await self._generate_recommendations(results)
        return results
    
    async def analyse_technical_seo(self, url: str) -> Dict:
        technical_metrics = {
            "site_speed": await self._analyse_site_speed(url),
            "mobile_optimisation": await self._check_mobile_optimisation(url),
            "security": await self._analyse_security(url),
            "indexation": await self.analyse_indexation(url),
            "site_architecture": await self._analyse_site_architecture(url),
            "crawl_efficiency": await self._analyse_crawl_efficiency(url),
            "international_targeting": await self._check_international_targeting(url),
            "structured_data": await self._analyse_structured_data(url)
        }

        return {
            "metrics": technical_metrics,
            "issues": await self._identify_technical_issues(technical_metrics),
            "score": await self._calculate_techincal_score(technical_metrics)
        }
    
    async def analyse_on_page_seo(self, url: str) -> Dict:
        on_page_metrics = {
            "content_quality": await self._analyse_content_quality(url),
            "keyword_optimisation": await self._analyse_keyword_usage(url),
            "meta_tags": await self._analyse_meta_tags(url),
            "heading_structure": await self._analyse_heading_structure(url),
            "internal_linking": await self._analyse_internal_linking(url),
            "content_structure": await self._analyse_content_structure(url),
            "user_experience": await self._analyse_user_experience(url),
            "multimedia_optimisation": await self._analyse_multimedia(url)
        }

        return {
            "metrics": on_page_metrics,
            "issues": await self._identify_on_page_issues(on_page_metrics),
            "score": await self._calculate_on_page_score(on_page_metrics)
        }
    
    async def analyse_off_page_seo(self, url: str) -> Dict:
        off_page_metrics = {
            "backlink_profile": await self._analyse_backlinks(url),
            "brand_signals": await self._analyse_brand_signals(url),
            "social_presence": await self._analyse_social_presence(url),
            "local_seo": await self._analyse_local_seo(url),
            "competitor_comparison": await self._analyse_competitors(url),
            "authority_metrics": await self._analyse_authority(url),
            "mentions": await self._analyse_brand_mentions(url),
            "trust_signals": await self._analyse_trust_signals(url)
        }

        return {
            "metrics": off_page_metrics,
            "issues": await self._identify_off_page_issues(off_page_metrics),
            "score": await self._calculate_off_page_score(off_page_metrics)
        }
    
    async def _analyse_site_speed(self, url: str) -> Dict:
        try:
            async with self.session.get(url) as response:
                start_time = datetime.now()
                await response.read()
                end_time = datetime.now()
                load_time = (end_time - start_time).total_seconds()
        
            metrics = {
                "page_load_time": 0,
                "time_to_first_byte": 0,
                "largest_contentful_paint": 0,
                "first_input_delay": 0,
                "cumulative_layout_shift": 0
            }

            return metrics
        
        except Exception as e:
            self.logger.error(f"Error analysing site speed: {str(e)}")
            return {
                "error": str(e),
                "performance_score": 0
            }


        async with aiohttp.ClientSession() as session:
            # implement speed test logic here
            pass

        return metrics
    
    async def _analyse_content_quality(self, url: str) -> Dict:
        """analyse content quality using llm"""
        prompt:f"""
        Analyse the content quality for {url} considering:
        1. Comprehensiveness
        2. Expertise demonstation
        3. Originality
        4. User Value
        5. Content freshness

        Format response as JSON with scores and recommendations for each point.
        """

        return await self.llm.analyse_content(prompt)
    
    async def _analyse_backlinks(self, url: str) -> Dict:
        # implement backlink analysis logic here
        # integrate with tools like ahrefs, moz or majestic
        return {
            "total_backlinks": 0,
            "referring_domains": 0,
            "domain_authority_distribution": {},
            "anchor_text_analysis": {},
            "link_quality_metrics": {}
        }
    
    async def generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate proritised recommendations based on analysis"""
        technical_recommendations = await self._generate_technical_recommendations(analysis["technical_seo"])

        on_page_recommendations = await self._generate_on_page_recommendations(analysis["on_page_seo"])

        off_page_recommendations = await self._generate_off_page_recommendations(analysis["off_page_seo"])
    
        all_recommendations = (
            technical_recommendations +
            on_page_recommendations +
            off_page_recommendations
        )

        return sorted(
            all_recommendations,
            ley = lambda x: (x["priority_score"], x["impact"]),
            reverse = True
        )
    
    async def _generate_technical_recommendations(self, technical_analysis: Dict) -> List[Dict]:
        """Generate technical SEO recommendations"""
        prompt = f"""
        Based on this technical SEO analysis:
        {json.dumps(technical_analysis)}

        Generate prioritised recommendations for:
        1. Site speed optimisation
        2. Mobile optimisation
        3. Indexation improvements
        4. Security enhancements
        5. Structured data implementation

        Format as a JSON list with priority_score and impact for each recommendation.
        """

        return await self.llm.generate_recommendations(prompt)
    

    async def _generate_on_page_recommendations(self, on_page_analysis: Dict) -> List[Dict]:
        """generate on-page SEO recommendations"""
        
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
        """

        return await self.llm.generate_recommendations(prompt)
    
    async def _generate_off_page_recommendation(self, off_page_analysis: Dict) -> List[Dict]:
        """Generate off-page SEO recommendations"""
        prompt = f"""
        Based on this off-page SEO analysis:
        {json.dumps(off_page_analysis)}

        Generate prioritised recommendations for:
        1. Link building opportunities
        2. Brand signal improvements
        3. Social media presence
        4. Local SEO optimisation
        5. Autority building

        Format as JSON list with priority_score and impact for each recommendation.
        """

        return await self.llm.generate_recommendations(prompt)
    

async def main():
    agent = ComprehensiveSEOAgent(api_key="key")

    url = "https://www.happy-skin.com"
    analysis = await agent.analyse_website(url)
    recommendations = await agent.generate_recommendations(analysis)

    print("Analysis complete!")
    print("\nTechnical SEO Score:", analysis["technical_seo"]["score"])
    print("On-page SEO Score:", analysis["on_page_seo"]["score"])
    print("Off-page SEO Score:", analysis["off_page_seo"]["score"])
    print("\nTop 3 Recommendations:")
    for rec in recommendations[:3]:
        print(f"- {rec['recommendation']} (Priority: {rec['priority_score']})")


if __name__ == "__main__":
    asyncio.run(main())