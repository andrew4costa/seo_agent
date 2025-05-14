#!/usr/bin/env python
"""
SEO Report Generator
This script runs the SEO analysis tool with real website crawling and analysis
"""

import os
import json
import time
import logging
from datetime import datetime
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
import concurrent.futures
import ssl
import socket
import random
from collections import Counter
import traceback
import tldextract

# Set up logging
logger = logging.getLogger('seo_agent')

# Define HEADERS globally so helper functions can access it
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_page_content(url):
    """Get the HTML content of a page"""
    try:
        # Use the globally defined HEADERS
        # Add debug logging for the URL we're fetching
        logger.info(f"Fetching content for URL: {url}")

        # Allow redirects, which is especially important for www vs non-www domains
        response = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)

        # Log the final URL after any redirects
        if response.url != url:
            logger.info(f"URL redirected to: {response.url}")

        response.raise_for_status()
        return {
            'content': response.text,
            'headers': response.headers,
            'final_url': response.url  # Store the final URL after redirects
        }
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error fetching {url}: {str(e)}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error fetching {url}: {str(e)}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout Error fetching {url}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error fetching {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {str(e)}")
        return None

def extract_links(soup, base_url, domain):
    """Extract links from a page that belong to the same domain"""
    links = []
    found_urls = set()  # Use a set for efficient duplicate checking

    try:
        logger.info(f"Extracting links from {base_url} for domain {domain}")

        # Find all <a> tags with href attributes
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Skip empty and javascript pseudo-links
            if not href or href.startswith('javascript:'):
                continue

            # Handle relative URLs
            if not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)

            # Remove fragments
            href = href.split('#')[0]

            # Skip if empty after removing fragment
            if not href:
                continue

            parsed_href = urlparse(href)
            parsed_domain = parsed_href.netloc

            # Log all links for debugging 
            logger.debug(f"Found link: {href}, netloc: {parsed_domain}, domain: {domain}")

            # Only include links from the same domain
            if parsed_domain == domain and href not in found_urls:
                found_urls.add(href)
                links.append(href)
                logger.debug(f"Added internal link: {href}")

        # Also look for links in onclick attributes (often used for navigation)
        for element in soup.find_all(attrs={'onclick': True}):
            onclick = element.get('onclick', '')
            url_match = re.search(r'(\'|")((http|https)://[^\'"]*)(\1)', onclick)
            if url_match:
                href = url_match.group(2)
                parsed_href = urlparse(href)
                if parsed_href.netloc == domain and href not in found_urls:
                    found_urls.add(href)
                    links.append(href)
                    logger.debug(f"Added onclick internal link: {href}")

        # Also check data-url attributes that are sometimes used for dynamic loading
        for element in soup.find_all(attrs={'data-url': True}):
            href = element.get('data-url', '')
            if href:
                # Handle relative URLs
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)

                parsed_href = urlparse(href)
                if parsed_href.netloc == domain and href not in found_urls:
                    found_urls.add(href)
                    links.append(href)
                    logger.debug(f"Added data-url internal link: {href}")

        logger.info(f"Found {len(links)} links on {base_url}")
    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {str(e)}")

    return links

def count_words(soup):
    """Count the number of words in the page content"""
    try:
        # Get text from the body
        text = soup.body.get_text(separator=' ', strip=True)
        # Count words
        words = text.split()
        return len(words)
    except Exception as e:
        logger.error(f"Error counting words: {str(e)}")
        return 0

def check_meta_tags(soup):
    """Check if the page has proper meta tags"""
    issues = []

    # Check for title
    title = soup.find('title')
    if not title or not title.text.strip():
        issues.append("Missing page title")
    elif len(title.text.strip()) < 10:
        issues.append("Title too short")
    elif len(title.text.strip()) > 70:
        issues.append("Title too long")

    # Check for meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if not meta_desc or not meta_desc.get('content', '').strip():
        issues.append("Missing meta description")
    elif meta_desc and len(meta_desc.get('content', '')) < 50:
        issues.append("Meta description too short")
    elif meta_desc and len(meta_desc.get('content', '')) > 160:
        issues.append("Meta description too long")

    return issues

def check_heading_structure(soup):
    """Check if the page has proper heading structure"""
    issues = []

    h1_tags = soup.find_all('h1')
    if not h1_tags:
        issues.append("Missing H1 tag")
    elif len(h1_tags) > 1:
        issues.append("Multiple H1 tags")

    return issues

def check_image_alt_tags(soup):
    """Check if images have alt tags"""
    missing_alt = 0
    images = soup.find_all('img')

    for img in images:
        if not img.get('alt') or img.get('alt').strip() == '':
            missing_alt += 1

    if missing_alt > 0:
        return [f"Missing alt text on {missing_alt} images"]
    return []

def check_internal_links(soup, base_url, domain):
    """Check if the page has enough internal links and return all found internal links"""
    issues = []

    internal_links = []
    external_links = []

    # Get all links from the page
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Skip empty, anchor, or javascript links
        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue

        # Handle relative URLs
        if not href.startswith(('http://', 'https://')):
            href = urljoin(base_url, href)

        try:
            parsed_href = urlparse(href)
            # Check if the link is internal (same domain)
            if parsed_href.netloc == domain:
                # Normalize URL to avoid duplicates - remove trailing slash and fragments
                normalized_url = href.split('#')[0].rstrip('/')
                # Skip self-references to the exact same page
                if normalized_url != base_url.rstrip('/'):
                    if normalized_url not in internal_links:
                        internal_links.append(normalized_url)
            elif parsed_href.netloc:  # Only add external links with a valid domain
                external_links.append(href)
        except:
            # Skip invalid URLs
            continue

    if len(internal_links) < 3:
        issues.append(f"Low internal linking (only {len(internal_links)} internal links)")

    return issues, internal_links, external_links

def check_content_html_ratio(html_content, soup):
    """Check the content-to-HTML ratio"""
    issues = []

    if not html_content or not soup.body:
        return issues

    html_size = len(html_content)
    text = soup.body.get_text(separator=' ', strip=True)
    text_size = len(text)

    if html_size == 0:
        ratio = 0
    else:
        ratio = (text_size / html_size) * 100

    if ratio < 10:
        issues.append(f"Low content-to-HTML ratio ({ratio:.1f}%)")

    return issues, ratio

def check_keyword_density(soup, keyword=None):
    """Check keyword density in the content"""
    issues = []
    keyword_info = {}

    if not soup.body or not keyword:
        return issues, keyword_info

    text = soup.body.get_text(separator=' ', strip=True).lower()
    words = re.findall(r'\b\w+\b', text)

    if not words:
        return issues, keyword_info

    # Count word frequencies
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only consider words with more than 3 characters
            word_counts[word] = word_counts.get(word, 0) + 1

    # Calculate densities
    total_words = len(words)
    keyword_densities = {word: (count / total_words) * 100 for word, count in word_counts.items()}

    # Sort by frequency
    top_keywords = sorted(keyword_densities.items(), key=lambda x: x[1], reverse=True)[:5]

    # Check main keyword density if provided
    if keyword and keyword.lower() in word_counts:
        keyword_density = (word_counts[keyword.lower()] / total_words) * 100
        if keyword_density > 5:
            issues.append(f"Keyword '{keyword}' appears too frequently ({keyword_density:.1f}%)")
        elif keyword_density < 0.5:
            issues.append(f"Keyword '{keyword}' density is low ({keyword_density:.1f}%)")

    keyword_info = {
        'top_keywords': top_keywords,
        'total_words': total_words
    }

    return issues, keyword_info

def check_mobile_friendliness(response_headers):
    """Check if the page has mobile-friendly indicators"""
    issues = []

    # Check for viewport meta tag (in the response body, not headers)
    # This is done in the analyze_page function with the soup

    # Check for server headers that might indicate mobile optimization
    has_vary_user_agent = False

    if response_headers:
        for header, value in response_headers.items():
            if header.lower() == 'vary' and 'user-agent' in value.lower():
                has_vary_user_agent = True

    if not has_vary_user_agent:
        issues.append("No 'Vary: User-Agent' header found")

    return issues

def check_ssl(url):
    """Check if the page uses HTTPS"""
    issues = []

    if not url.startswith('https://'):
        issues.append("Site not using HTTPS")

    return issues

def analyze_page(url, progress_callback=None, page_num=0, total_pages=1, domain=None, keyword=None):
    """Analyze a single page"""
    try:
        # Update progress
        if progress_callback:
            progress_callback(10 + (page_num / total_pages * 40),
                             f"Analyzing page {page_num+1}/{total_pages}: {url}",
                             1)

        # Get page content
        page_data = get_page_content(url)
        if not page_data:
            logger.warning(f"Could not fetch content for {url}, returning minimal data")
            return {
                'url': url,
                'title': "Error - Could not access page",
                'meta_description': "Error - Could not access page",
                'h1': "Error - Could not access page",
                'word_count': 0,
                'load_time': 0,
                'has_viewport': False,
                'internal_links': [],
                'external_links': [],
                'issues': 1,
                'issue_details': [f"Error accessing page: Could not fetch content"]
            }

        html_content = page_data['content']
        response_headers = page_data['headers']

        # Use the final URL after redirects for analysis
        final_url = page_data.get('final_url', url)
        if final_url != url:
            logger.info(f"Using final URL for analysis: {final_url} (redirected from {url})")

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract basic info
        title = soup.find('title')
        title_text = title.text.strip() if title else "No title found"

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc_text = meta_desc.get('content', '') if meta_desc else "No meta description found"

        h1 = soup.find('h1')
        h1_text = h1.text.strip() if h1 else "No H1 found"

        # Check for viewport meta tag
        viewport = soup.find('meta', attrs={'name': 'viewport'})
        has_viewport = bool(viewport)

        # Count words
        word_count = count_words(soup)

        # Calculate load time (estimate)
        load_time = len(html_content) / 100000

        # Set domain if not provided or use domain from final URL after redirects
        if not domain:
            domain = urlparse(final_url).netloc

        # Check for issues
        issues = []
        issues.extend(check_meta_tags(soup))
        issues.extend(check_heading_structure(soup))
        issues.extend(check_image_alt_tags(soup))

        # New checks - use final_url for internal link checking
        internal_link_issues, internal_links, external_links = check_internal_links(soup, final_url, domain)
        issues.extend(internal_link_issues)

        content_ratio_issues, content_ratio = check_content_html_ratio(html_content, soup)
        issues.extend(content_ratio_issues)

        keyword_issues, keyword_info = check_keyword_density(soup, keyword)
        issues.extend(keyword_issues)

        if not has_viewport:
            issues.append("Missing viewport meta tag")

        # Get response headers for mobile-friendliness check
        mobile_issues = check_mobile_friendliness(response_headers)
        issues.extend(mobile_issues)

        ssl_issues = check_ssl(final_url)
        issues.extend(ssl_issues)

        # Placeholder for schema check within analyze_page if it were to be implemented
        # For now, schema is checked site-wide in generate_seo_report
        # has_schema_on_page = "application/ld+json" in html_content # Basic check

        # If we have no content, add a specific issue
        if word_count == 0:
            issues.append("Page has no content")

        return {
            'url': final_url,  # Use the final URL after redirects
            'original_url': url if final_url != url else None,  # Store original URL if redirected
            'title': title_text,
            'meta_description': meta_desc_text,
            'h1': h1_text,
            'word_count': word_count,
            'load_time': round(load_time, 2),
            'has_viewport': has_viewport,
            'internal_links': internal_links,
            'external_links': external_links,
            'content_ratio': round(content_ratio, 2) if 'content_ratio' in locals() else None,
            'keyword_info': keyword_info if 'keyword_info' in locals() else None,
            # 'has_schema_on_page': has_schema_on_page, # Add if schema check is per page
            'issues': len(issues),
            'issue_details': issues
        }

    except Exception as e:
        logger.error(f"Error analyzing page {url}: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging
        return {
            'url': url,
            'title': "Error analyzing page",
            'meta_description': "Error",
            'h1': "Error",
            'word_count': 0,
            'load_time': 0,
            'internal_links': [],
            'external_links': [],
            'issues': 1,
            'issue_details': [f"Error analyzing page: {str(e)}"]
        }

def generate_recommendations(page_insights):
    """Generate recommendations based on page insights, now including actionable steps"""
    recommendations = []

    # Helper to format actionable steps
    def format_steps(steps_list):
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps_list)])

    # Check for title issues
    title_issues_pages = [page['url'] for page in page_insights if any("title" in issue.lower() for issue in page.get('issue_details', []))]
    title_issues_count = len(title_issues_pages)
    if title_issues_count > 0:
        title_issue_details = []
        for page in page_insights:
            title_issues_found = [issue for issue in page.get('issue_details', []) if "title" in issue.lower()]
            if title_issues_found:
                title_issue_details.append(f"- {page['url']}: {', '.join(title_issues_found)}")
        steps = [
            f"Identify the {title_issues_count} page(s) with title issues from the list provided in 'Why This Matters'.",
            "For each page, navigate to your website's content management system (CMS) or HTML editor.",
            "Edit the <title> tag to be unique, descriptive, and between 50-60 characters.",
            "Ensure the primary keyword for the page is included naturally within the title.",
            "Save changes and request re-indexing via Google Search Console if necessary."
        ]
        recommendations.append({
            'title': 'Optimize page titles',
            'description': f'Fix title issues on {title_issues_count} pages. Ensure titles are 50-60 characters long and include relevant keywords.',
            'impact': 'high',
            'effort': 'low',
            'detailed_reason': f"Page titles are crucial for SEO... Issues found:\n{chr(10).join(title_issue_details[:3])}{chr(10)}{'... and more' if len(title_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for meta description issues
    meta_issues_pages = [page['url'] for page in page_insights if any("meta description" in issue.lower() for issue in page.get('issue_details', []))]
    meta_issues_count = len(meta_issues_pages)
    if meta_issues_count > 0:
        meta_issue_details = []
        for page in page_insights:
            meta_issues_found = [issue for issue in page.get('issue_details', []) if issue and "meta description" in issue.lower()]
            if meta_issues_found:
                meta_issue_details.append(f"- {page['url']}: {', '.join(meta_issues_found)}")
        steps = [
            f"Locate the {meta_issues_count} page(s) with meta description issues.",
            "In your CMS or HTML, edit the <meta name=\"description\" content=\"...\"> tag.",
            "Write a compelling summary of the page content, 120-160 characters long.",
            "Include relevant keywords naturally and a call-to-action if appropriate.",
            "Ensure each meta description is unique across your site."
        ]
        recommendations.append({
            'title': 'Improve meta descriptions',
            'description': f'Fix meta description issues on {meta_issues_count} pages. Ensure descriptions are 120-160 characters and compelling.',
            'impact': 'medium',
            'effort': 'medium',
            'detailed_reason': f"Meta descriptions improve click-through rates... Issues found:\n{chr(10).join(meta_issue_details[:3])}{chr(10)}{'... and more' if len(meta_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for heading structure issues (H1)
    heading_issues_pages = [page['url'] for page in page_insights if any("h1" in issue.lower() for issue in page.get('issue_details', []))]
    heading_issues_count = len(heading_issues_pages)
    if heading_issues_count > 0:
        heading_issue_details = []
        for page in page_insights:
            heading_issues_found = [issue for issue in page.get('issue_details', []) if "h1" in issue.lower()]
            if heading_issues_found:
                heading_issue_details.append(f"- {page['url']}: {', '.join(heading_issues_found)}")
        steps = [
            f"Identify the {heading_issues_count} page(s) with H1 tag issues.",
            "Ensure each page has exactly one <h1> tag.",
            "The <h1> tag should accurately represent the main topic of the page content.",
            "Use subsequent headings (<h2>, <h3>, etc.) to structure the rest of the content logically."
        ]
        recommendations.append({
            'title': 'Fix heading structure',
            'description': f'Fix heading structure issues on {heading_issues_count} pages. Each page should have exactly one H1 tag.',
            'impact': 'medium',
            'effort': 'low',
            'detailed_reason': f"Proper heading structure helps SEO... Issues found:\n{chr(10).join(heading_issue_details[:3])}{chr(10)}{'... and more' if len(heading_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for alt text issues
    alt_issues_pages = []
    if page_insights:
        for page in page_insights:
            if page and page.get('issue_details'): # Check if page and issue_details exist
                for issue_detail_str in page.get('issue_details', []):
                    if issue_detail_str and "alt text" in issue_detail_str.lower(): # Check issue_detail_str
                        page_url = page.get('url')
                        if page_url and page_url not in alt_issues_pages: 
                            alt_issues_pages.append(page_url)
                        break # Found an alt text issue for this page
    
    alt_issues_count = len(alt_issues_pages)
    if alt_issues_count > 0:
        alt_issue_details = []
        img_count_total = 0 # Recalculate img_count_total based on actual identified issues
        # This loop now also helps define which pages have alt text issues for alt_issue_details
        if page_insights:
            for page in page_insights:
                if page and page.get('url') in alt_issues_pages: # Only process pages known to have alt text issues
                    page_specific_alt_issues_found = []
                    page_issue_details = page.get('issue_details', [])
                    if page_issue_details:
                        for issue_str in page_issue_details:
                            if issue_str and "alt text" in issue_str.lower():
                                page_specific_alt_issues_found.append(issue_str)
                                if "Missing alt text on" in issue_str:
                                    match = re.search(r'(\d+)', issue_str)
                                    if match:
                                        try:
                                            img_count_total += int(match.group(1))
                                        except ValueError:
                                            logger.warning(f"Could not parse number of images from alt text issue: {issue_str}")
                    if page_specific_alt_issues_found: # Add details if any specific alt text issues were logged for this page
                        alt_issue_details.append(f"- {page.get('url')}: {', '.join(page_specific_alt_issues_found)}")
        steps = [
            f"Identify images lacking alt text across {alt_issues_count} page(s) (approx. {img_count_total} images if reported per page).",
            "For each image, add a descriptive 'alt' attribute in the <img> tag.",
            "The alt text should accurately describe the image content and context.",
            "If an image is purely decorative, use an empty alt attribute (alt=\"\")."
        ]
        recommendations.append({
            'title': 'Add missing alt text',
            'description': f'Add alt text to images on {alt_issues_count} pages for better accessibility and SEO.',
            'impact': 'medium',
            'effort': 'medium',
            'detailed_reason': f"Alt text is crucial for accessibility and image SEO... Issues found:\n{chr(10).join(alt_issue_details[:3])}{chr(10)}{'... and more' if len(alt_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for internal linking issues
    internal_link_issues_pages = [page['url'] for page in page_insights if any("internal linking" in issue.lower() for issue in page.get('issue_details', []))]
    internal_link_issues_count = len(internal_link_issues_pages)
    if internal_link_issues_count > 0:
        link_issue_details = []
        for page in page_insights:
            link_issues_found = [issue for issue in page.get('issue_details', []) if "internal linking" in issue.lower()]
            if link_issues_found:
                link_count = len(page.get('internal_links', [])) if isinstance(page.get('internal_links'), list) else 0
                link_issue_details.append(f"- {page['url']}: Only {link_count} internal links found by crawler on this page.")
        steps = [
            f"Review the {internal_link_issues_count} page(s) identified with low internal linking.",
            "Identify relevant content on other pages of your site that could link to these pages.",
            "Add contextual internal links from relevant anchor text on those other pages.",
            "Aim for at least 3-5 relevant internal links to and from important pages."
        ]
        recommendations.append({
            'title': 'Improve internal linking',
            'description': f'Increase internal links on {internal_link_issues_count} pages to improve site structure and navigation.',
            'impact': 'medium',
            'effort': 'medium',
            'detailed_reason': f"Strong internal linking helps SEO... Issues found:\n{chr(10).join(link_issue_details[:3])}{chr(10)}{'... and more' if len(link_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for content-to-HTML ratio issues
    content_ratio_issues_pages = [page['url'] for page in page_insights if any("content-to-html ratio" in issue.lower() for issue in page.get('issue_details', []))]
    content_ratio_issues_count = len(content_ratio_issues_pages)
    if content_ratio_issues_count > 0:
        ratio_issue_details = []
        for page in page_insights:
            ratio_issues_found = [issue for issue in page.get('issue_details', []) if "content-to-html ratio" in issue.lower()]
            if ratio_issues_found:
                ratio_value = page.get('content_ratio', 'unknown')
                ratio_issue_details.append(f"- {page['url']}: Content-to-HTML ratio of {ratio_value}%")
        steps = [
            f"Identify the {content_ratio_issues_count} page(s) with low content-to-HTML ratio.",
            "Review the HTML source of these pages to identify unnecessary code, comments, or inline styles.",
            "Minify HTML, CSS, and JavaScript files where possible.",
            "Ensure the main content of the page is substantial and not outweighed by code."
        ]
        recommendations.append({
            'title': 'Improve content-to-HTML ratio',
            'description': f'Clean up HTML code on {content_ratio_issues_count} pages to improve the text-to-code ratio.',
            'impact': 'low',
            'effort': 'medium',
            'detailed_reason': f"A low content-to-HTML ratio may suggest code bloat... Issues found:\n{chr(10).join(ratio_issue_details[:3])}{chr(10)}{'... and more' if len(ratio_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for mobile-friendliness issues (viewport & Vary header)
    mobile_issues_pages = [page['url'] for page in page_insights if 
                      any("viewport" in issue.lower() for issue in page.get('issue_details', [])) or
                      any("user-agent" in issue.lower() for issue in page.get('issue_details', []))]
    mobile_issues_count = len(mobile_issues_pages)
    if mobile_issues_count > 0:
        mobile_issue_details = []
        for page in page_insights:
            viewport_issues = [issue for issue in page.get('issue_details', []) if "viewport" in issue.lower()]
            user_agent_issues = [issue for issue in page.get('issue_details', []) if "user-agent" in issue.lower()]
            all_mobile_issues = viewport_issues + user_agent_issues
            if all_mobile_issues:
                mobile_issue_details.append(f"- {page['url']}: {', '.join(all_mobile_issues)}")
        steps = [
            f"Review the {mobile_issues_count} page(s) with mobile-friendliness issues.",
            "Ensure all pages have a <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"> tag.",
            "Test pages using Google's Mobile-Friendly Test tool.",
            "Ensure your web server sends the 'Vary: User-Agent' HTTP header if you serve different HTML to mobile users (less common now with responsive design).",
            "Implement responsive web design principles so content adapts to all screen sizes."
        ]
        recommendations.append({
            'title': 'Improve mobile-friendliness',
            'description': f'Optimize {mobile_issues_count} pages for mobile devices by adding proper viewport tags and responsive design.',
            'impact': 'high',
            'effort': 'high',
            'detailed_reason': f"Mobile-friendliness is crucial for ranking... Issues found:\n{chr(10).join(mobile_issue_details[:3])}{chr(10)}{'... and more' if len(mobile_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Check for SSL issues
    ssl_issues_pages = [page['url'] for page in page_insights if any("https" in issue.lower() for issue in page.get('issue_details', []))]
    ssl_issues_count = len(ssl_issues_pages)
    if ssl_issues_count > 0:
        ssl_issue_details = []
        for page in page_insights:
            ssl_issues_found = [issue for issue in page.get('issue_details', []) if "https" in issue.lower()]
            if ssl_issues_found:
                ssl_issue_details.append(f"- {page['url']}: {', '.join(ssl_issues_found)}")
        steps = [
            "Ensure your website has a valid SSL/TLS certificate installed.",
            "Configure your server to redirect all HTTP traffic to HTTPS.",
            "Update all internal links, canonical tags, and sitemap URLs to use HTTPS.",
            "Check for mixed content (HTTP resources on HTTPS pages) and update them to HTTPS."
        ]
        recommendations.append({
            'title': 'Implement HTTPS',
            'description': 'Switch to HTTPS to improve security and SEO ranking.',
            'impact': 'high',
            'effort': 'medium',
            'detailed_reason': f"HTTPS is a confirmed Google ranking factor... Issues found:\n{chr(10).join(ssl_issue_details[:3])}{chr(10)}{'... and more' if len(ssl_issue_details) > 3 else ''}",
            'actionable_steps': format_steps(steps)
        })

    # Add more recommendation types here if needed...

    return recommendations

def calculate_score(page_insights):
    """Calculate overall SEO score based on page insights"""
    total_issues = sum(page.get('issues', 0) for page in page_insights)
    pages_count = len(page_insights)

    if pages_count == 0:
        return 50  # Default score if no pages analyzed

    # Calculate average issues per page
    avg_issues = total_issues / pages_count

    # Calculate score (fewer issues = higher score)
    # Base score of 100, subtract points for issues
    score = 100 - (avg_issues * 10)

    # Ensure score is between 0 and 100
    score = max(0, min(100, score))

    return int(score)

def check_robots_txt(base_url):
    """Check if robots.txt exists and is accessible"""
    robots_url = urljoin(base_url, '/robots.txt')
    try:
        response = requests.get(robots_url, timeout=5, headers=HEADERS)
        return response.status_code == 200 and response.text.strip() != ""
    except requests.RequestException:
        return False

def check_sitemap_xml(base_url):
    """Check if sitemap.xml (or common variations) exists and is accessible"""
    sitemap_paths = ['/sitemap.xml', '/sitemap_index.xml', '/sitemap.xml.gz', '/sitemap_index.xml.gz']
    for path in sitemap_paths:
        sitemap_url = urljoin(base_url, path)
        try:
            response = requests.head(sitemap_url, timeout=5, headers=HEADERS, allow_redirects=True)
            if response.status_code == 200:
                return True
            # Check for sitemap in robots.txt as well (more advanced, not implemented here for brevity)
        except requests.RequestException:
            continue
    return False

# --- Helper Functions for Dynamic Scores & Visualization ---
# MOVED TO MODULE LEVEL
def evaluate_click_depth_score(click_depth_dict, total_pages_analyzed):
    if not total_pages_analyzed or not click_depth_dict:
        return "N/A"
    pages_within_3_clicks = 0
    for depth_str, count in click_depth_dict.items():
        try:
            depth = int(depth_str)
            if depth <= 3:
                pages_within_3_clicks += count
        except ValueError:
            continue 
    
    percentage_within_3_clicks = (pages_within_3_clicks / total_pages_analyzed) * 100 if total_pages_analyzed > 0 else 0

    if percentage_within_3_clicks > 70:
        return "Good"
    elif percentage_within_3_clicks >= 50:
        return "Fair"
    else:
        return "Needs Improvement"

def evaluate_internal_linking_health(links_by_page_list, total_pages_analyzed):
    if not total_pages_analyzed or not links_by_page_list:
        return "N/A"
    
    total_internal_links_sum = sum(page.get('links_count', 0) for page in links_by_page_list)
    pages_with_few_links = sum(1 for page in links_by_page_list if page.get('links_count', 0) < 2)
    
    avg_links_per_page = total_internal_links_sum / total_pages_analyzed if total_pages_analyzed > 0 else 0
    percentage_few_links = (pages_with_few_links / total_pages_analyzed) * 100 if total_pages_analyzed > 0 else 0

    if avg_links_per_page > 5 and percentage_few_links < 10:
        return "Good"
    elif avg_links_per_page >= 3 and percentage_few_links < 25:
        return "Fair"
    else:
        return "Needs Improvement"

def evaluate_url_structure_health(analyzed_urls_list):
    if not analyzed_urls_list:
        return "N/A"
    
    num_urls = len(analyzed_urls_list)
    total_length = 0
    uppercase_count = 0
    underscore_count = 0
    param_heavy_count = 0

    for u_str in analyzed_urls_list:
        total_length += len(u_str)
        parsed_u = urlparse(u_str) 
        if any(c.isupper() for c in parsed_u.path):
            uppercase_count += 1
        if '_' in parsed_u.path:
            underscore_count += 1
        if parsed_u.query.count('=') > 2:
            param_heavy_count +=1
    
    avg_len = total_length / num_urls if num_urls > 0 else 0

    score = 0
    if avg_len < 75: score +=1
    if uppercase_count < (num_urls * 0.1): score +=1
    if underscore_count < (num_urls * 0.1): score +=1
    if param_heavy_count < (num_urls * 0.2): score +=1

    if score >= 3:
        return "Generally Optimized"
    elif score >= 2:
        return "Some Inconsistencies"
    else:
        return "Needs Improvement"

def generate_text_tree(analyzed_pages_map, base_url, max_display_depth=2):
    """Generates a text-based site structure tree up to max_display_depth.
    analyzed_pages_map: A dictionary mapping page URLs to their data (like page_insights).
    base_url: The root URL of the site.
    max_display_depth: How many levels deep to render the tree (0 = homepage only).
    """
    if not analyzed_pages_map:
        return "(No analyzed pages to build tree)"

    def normalize_url_for_tree(page_url):
        parsed = urlparse(page_url)
        # Normalize netloc by removing www. and path by rstrip('/')
        norm_netloc = parsed.netloc.replace("www.", "")
        norm_path = parsed.path.rstrip('/')
        if not norm_path: norm_path = '/' # Ensure root path is at least '/'
        return norm_netloc, norm_path

    # Create a lookup for page data by (normalized_netloc, normalized_path) for easier access
    path_to_page_data = {}
    for url_key, data_val in analyzed_pages_map.items():
        if url_key: # Ensure URL key is not None
            norm_netloc, norm_path = normalize_url_for_tree(url_key)
            path_to_page_data[(norm_netloc, norm_path)] = data_val
    
    base_norm_netloc, root_path_norm = normalize_url_for_tree(base_url)
    root_key = (base_norm_netloc, root_path_norm)

    page_depths = {}
    page_children = {key: [] for key in path_to_page_data.keys()}
    
    queue = [(root_key, 0)]
    visited_keys = {root_key}
    if root_key in path_to_page_data: # Check if normalized root_key exists
        page_depths[root_key] = 0
    else:
        # Attempt to find a key that matches just the root_path_norm if netloc differs slightly
        # This is a fallback, ideally base_url should perfectly match one of the analyzed page URLs after normalization
        found_fallback_root = False
        for k_norm_netloc, k_norm_path in path_to_page_data.keys():
            if k_norm_path == root_path_norm:
                # If we find a page with the same path, assume it's the intended root for tree building
                # This handles cases where http/https or www might be inconsistent in input base_url vs page_insights urls
                root_key = (k_norm_netloc, k_norm_path)
                page_depths[root_key] = 0
                visited_keys = {root_key} # Reset visited with the found key
                queue = [(root_key, 0)] # Reset queue
                found_fallback_root = True
                logger.info(f"Tree Warning: Homepage base_url normalized to {base_norm_netloc, root_path_norm}, but found matching root path via {root_key} in analyzed pages.")
                break
        if not found_fallback_root:
            return f"(Homepage with normalized path '{root_path_norm}' from base_url '{base_url}' not found in analyzed pages map. Keys: {list(path_to_page_data.keys())[:5]}...)"

    head = 0
    while head < len(queue):
        current_key_from_queue, current_d = queue[head]
        head += 1

        if current_d >= max_display_depth + 1: 
            continue

        current_page_node_data = path_to_page_data.get(current_key_from_queue)
        if not current_page_node_data or not current_page_node_data.get('internal_links'):
            continue

        for child_link_url in current_page_node_data['internal_links']:
            if not child_link_url: continue
            child_norm_netloc, child_norm_path = normalize_url_for_tree(child_link_url)
            child_key = (child_norm_netloc, child_norm_path)

            if child_key in path_to_page_data: 
                if child_key not in visited_keys:
                    visited_keys.add(child_key)
                    page_depths[child_key] = current_d + 1
                    page_children[current_key_from_queue].append(child_key) # Use key for parent
                    if current_d + 1 <= max_display_depth + 1:
                         queue.append((child_key, current_d + 1))
                elif page_depths.get(child_key, float('inf')) > current_d + 1:
                    page_depths[child_key] = current_d + 1 
    
    tree_lines = []
    def build_lines_recursive(node_key, current_depth_val, prefix_str=""):
        _nd_norm_netloc, nd_norm_path = node_key
        display_path_str = nd_norm_path if len(nd_norm_path) < 50 else nd_norm_path[:47] + "..."
        if node_key == root_key:
            tree_lines.append(f"{display_path_str} (Homepage)")
        else:
            tree_lines.append(f"{prefix_str}{display_path_str}")

        if current_depth_val < max_display_depth:
            children_of_node_keys = sorted([ckey for ckey in page_children.get(node_key, []) if page_depths.get(ckey) == current_depth_val + 1],
                                           key=lambda k: k[1]) # Sort by path
            for i, child_k in enumerate(children_of_node_keys):
                is_last_child = (i == len(children_of_node_keys) - 1)
                new_prefix_str = prefix_str + ("    " if "└" in prefix_str else "│   ")
                connector_str = "└── " if is_last_child else "├── "
                build_lines_recursive(child_k, current_depth_val + 1, new_prefix_str + connector_str)
    
    build_lines_recursive(root_key, 0)
        
    return "\n".join(tree_lines)
# END OF HELPER FUNCTION DEFINITIONS AT MODULE LEVEL

def generate_seo_report(url, pages=5, progress_callback=None, keyword=None):
    """Generate SEO report for a website"""
    # Initialize the report structure
    report = {
        'url': url,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'score': 0,
        'page_insights': [],
        'recommendations': [],
        'top_issues': {},
        'performance': {},
        'technical_issues': [], # Ensure technical_issues is initialized in report
        'ux_analysis': {},
        'architecture_analysis': {},
        'click_depth': {}
    }

    try:
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Step 1: Initial setup
        if progress_callback:
            progress_callback(5, 'Setting up analysis', 1)

        # Determine SSL status from the final URL
        final_site_url_for_checks = url # Will be updated if homepage redirects
        site_has_ssl = final_site_url_for_checks.startswith('https://')

        # Get domain from URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Step 2: Crawl website to find pages
        if progress_callback:
            progress_callback(10, 'Crawling website to discover pages', 1)

        # Get homepage content
        homepage_data = get_page_content(url)
        if not homepage_data:
            if progress_callback:
                progress_callback(100, 'Failed to access website', 5)
            return report

        # Use the final URL after any redirects
        if 'final_url' in homepage_data:
            url = homepage_data['final_url']
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            logger.info(f"Using final homepage URL: {url}")
            final_site_url_for_checks = url # Update for subsequent checks
            site_has_ssl = final_site_url_for_checks.startswith('https://') # Re-check SSL for final URL

        # Parse homepage
        soup = BeautifulSoup(homepage_data['content'], 'html.parser')

        # Find links on the homepage
        all_links = extract_links(soup, url, domain)

        # Keep track of all discovered links and visited links
        discovered_links = set(all_links)
        visited_links = set([url])  # We've visited the homepage
        queue = []

        # Add remaining links from homepage to queue
        for link in all_links:
            if link not in discovered_links and link not in visited_links:
                queue.append(link)

        # Track depth and ensure we don't go too deep
        max_depth = 3  # How deep to crawl (increased from 2)
        max_pages_per_level = 5  # How many pages to visit at each level (increased from 3)
        current_depth = 1

        logger.info(f"Starting breadth-first crawl with max depth {max_depth}")

        # Breadth-first crawl
        while queue and current_depth <= max_depth:
            logger.info(f"Crawling depth {current_depth}, queue size: {len(queue)}")

            # Process this level
            level_links = queue.copy()
            queue = []  # Reset queue for next level

            # Only process a limited number of pages per level to avoid overloading
            for i, link in enumerate(level_links[:max_pages_per_level]):
                if link in visited_links:
                    continue

                logger.info(f"Visiting {link} (page {i+1}/{min(max_pages_per_level, len(level_links))} at depth {current_depth})")
                visited_links.add(link)

                try:
                    link_data = get_page_content(link)
                    if link_data:
                        # Use the final URL after any redirects for link extraction
                        final_link = link_data.get('final_url', link)
                        if final_link != link:
                            logger.info(f"Link {link} redirected to {final_link}")

                        link_soup = BeautifulSoup(link_data['content'], 'html.parser')
                        # Pass the final link URL to extract_links
                        new_links = extract_links(link_soup, final_link, domain)

                        # Add new links to the queue for next level
                        for new_link in new_links:
                            if new_link not in discovered_links and new_link not in visited_links:
                                discovered_links.add(new_link)
                                queue.append(new_link)
                except Exception as e:
                    logger.error(f"Error crawling {link}: {str(e)}")

            # Go to next depth level
            current_depth += 1

        logger.info(f"Crawling complete. Found {len(discovered_links)} unique links on {domain}")

        # Now categorize all the discovered links
        all_links = list(discovered_links)

        # Sort links for deterministic ordering
        all_links.sort()

        # Categorize links by type - adjust the patterns for happy-skin.com's site structure
        # THIS BLOCK NOW BECOMES THE DEFAULT FOR ALL SITES
        product_pages = [link for link in all_links if any(pattern in link.lower() for pattern in ['/product/', '/shop/', '/item/', '/buy/'])]
        blog_pages = [link for link in all_links if any(pattern in link.lower() for pattern in ['/blog/', '/news/', '/article/', '/post/'])]
        important_pages = [link for link in all_links if any(pattern in link.lower() for pattern in ['/about/', '/contact/', '/faq/', '/how-', '/features/', '/why-'])]
        legal_pages = [link for link in all_links if any(pattern in link.lower() for pattern in ['/privacy-policy', '/terms', '/cookie', '/legal/', '/gdpr'])]

        other_pages = [link for link in all_links if (
            link not in product_pages and 
            link not in blog_pages and 
            link not in important_pages and 
            link not in legal_pages
        )]

        logger.info(f"Categorized links: {len(product_pages)} product pages, {len(blog_pages)} blog pages, " 
                    f"{len(important_pages)} important pages, {len(legal_pages)} legal pages, {len(other_pages)} other pages")

        # Balance the page types for analysis
        links_to_analyze = [url]  # Always include homepage

        # Define how many of each type to include
        page_count = pages - 1  # Account for homepage

        # REMOVE HAPPY-SKIN.COM SPECIFIC LOGIC BLOCK
        # # For happy-skin.com, prioritize product pages heavily
        # if 'happy-skin.com' in domain.lower():
        #     # ... (removed happy-skin.com specific balancing logic) ...
        # else:
        # THIS BLOCK NOW BECOMES THE DEFAULT FOR ALL SITES
        available_types = sum(1 for x in [product_pages, blog_pages, important_pages, other_pages, legal_pages] if len(x) > 0)
        if available_types == 0:
            logger.warning("No additional pages found beyond homepage")
        else:
            remaining = page_count
            if product_pages:
                max_product = min(max(int(page_count * 0.3), 1), len(product_pages), remaining)
                links_to_analyze.extend(product_pages[:max_product])
                remaining -= max_product
            if blog_pages and remaining > 0:
                max_blog = min(max(int(page_count * 0.2), 1), len(blog_pages), remaining)
                links_to_analyze.extend(blog_pages[:max_blog])
                remaining -= max_blog
            if important_pages and remaining > 0:
                max_important = min(max(int(page_count * 0.2), 1), len(important_pages), remaining)
                links_to_analyze.extend(important_pages[:max_important])
                remaining -= max_important
            if other_pages and remaining > 0:
                max_other = min(max(int(page_count * 0.2), 1), len(other_pages), remaining)
                links_to_analyze.extend(other_pages[:max_other])
                remaining -= max_other
            if legal_pages and remaining > 0:
                max_legal = min(max(int(page_count * 0.1), 1), len(legal_pages), remaining)
                links_to_analyze.extend(legal_pages[:max_legal])

        # Normalize URLs before de-duplication and final selection
        normalized_links_to_analyze = []
        for link_url in links_to_analyze:
            parsed_link = urlparse(link_url)
            # Consistent scheme, netloc, and path with trailing slash, no fragment/query for this purpose
            # Path should be '/' if empty
            path = parsed_link.path if parsed_link.path else '/'
            if not path.endswith('/'):
                path += '/'
            # Reconstruct, ensuring no double slashes if path was already '/'
            if path == '//': path = '/'
            # Using http as a common base for comparison, actual fetch uses original scheme
            norm_url = f"http://{parsed_link.netloc.replace('www.','')}{path}" 
            # For the actual list, keep original URLs but use normalized for de-duplication map
            normalized_links_to_analyze.append(link_url)

        # De-duplicate based on the original URLs after selection strategy, then slice
        # The previous dict.fromkeys would keep original forms if they were different strings
        # A more robust de-duplication would use a normalized form as key if we want to be stricter
        # For now, let's ensure the list passed to analyze_page has gone through the intended selection.
        
        # Ensure we don't exceed the requested page count and remove duplicates (based on original string form)
        final_selected_links = list(dict.fromkeys(links_to_analyze))[:pages]
        report['analyzed_urls'] = final_selected_links # Store the actual URLs that will be analyzed
        
        # The variable `links_to_analyze` should now hold the final list for the ThreadPoolExecutor
        links_to_analyze = final_selected_links 

        logger.info(f"Final pages to analyze ({len(links_to_analyze)}): {links_to_analyze}")

        # Step 3: Analyze each page
        page_insights = []

        # Use ThreadPoolExecutor for parallel processing but process results in a deterministic order
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(links_to_analyze))) as executor:
            future_to_url = {
                executor.submit(
                    analyze_page, 
                    link, 
                    progress_callback, 
                    i, 
                    len(links_to_analyze),
                    domain,
                    keyword
                ): link for i, link in enumerate(links_to_analyze)
            }

            # Process results in the original URL order for consistency
            for link in links_to_analyze:
                for future, url_analyzed in future_to_url.items():
                    if url_analyzed == link:
                        try:
                            page_data = future.result()
                            if page_data:
                                page_insights.append(page_data)
                        except Exception as e:
                            logger.error(f"Error analyzing {url_analyzed}: {str(e)}")
                        break

        # Step 4: Performance analysis
        if progress_callback:
            progress_callback(60, 'Analyzing performance', 3)

        # Site-wide technical checks (before 'performance' dict is fully built)
        site_has_robots = check_robots_txt(final_site_url_for_checks)
        site_has_sitemap = check_sitemap_xml(final_site_url_for_checks)
        
        # Schema check (aggregate from pages)
        site_has_schema = False
        if page_insights: # Ensure page_insights is populated before iterating
            for page in page_insights:
                # Assuming analyze_page might add 'schema_types' or a boolean 'has_schema_on_page'
                if page.get('has_schema_on_page', False) or page.get('schema_types'):
                    site_has_schema = True
                    break

        # Calculate average load time
        avg_load_time = sum(page.get('load_time', 0) for page in page_insights) / len(page_insights) if page_insights else 0

        # Use a simple algorithm to estimate page speed score (1-100)
        page_speed = max(0, min(100, int(100 - (avg_load_time * 20))))

        # Analyze mobile-friendliness based on viewport meta tag presence
        mobile_friendly = all(page.get('has_viewport', False) for page in page_insights) if page_insights else False

        performance = {
            'page_speed': page_speed,
            'mobile_friendly': mobile_friendly,
            'load_time': round(avg_load_time, 2),
            'render_time': round(avg_load_time * 0.7, 2),  # Estimate
            'server_response_time': round(avg_load_time * 0.3, 2),  # Estimate
            'has_ssl': site_has_ssl,
            'has_robots': site_has_robots,
            'has_sitemap': site_has_sitemap,
            'has_schema': site_has_schema
        }
        report['performance'] = performance # Assign to report early
        current_performance_data = performance 

        # --- START OF FULL LINK ANALYSIS LOGIC ---
        # Collect all internal links - with proper deduplication
        
        unique_internal_links = set() # Collects all unique internal links found across the site
        internal_links_by_page = []

        if page_insights: # Ensure page_insights is not None and has data
            for page_data_from_insights in page_insights:
                page_url = page_data_from_insights.get('url')
                if not page_url: # Skip if a page in insights somehow has no URL
                    logger.warning("Page found in page_insights with no URL. Skipping for internal_links_by_page.")
                    continue 

                # Get internal links found on this page
                current_page_internal_links = page_data_from_insights.get('internal_links', [])
                if not isinstance(current_page_internal_links, list):
                    logger.warning(f"internal_links for {page_url} is not a list, treating as empty.")
                    current_page_internal_links = [] 

                # Filter out self-references and de-duplicate for this page
                normalized_page_url = page_url.rstrip('/')
                processed_links_for_this_page = []
                seen_on_this_page = set()

                for link_url in current_page_internal_links:
                    if link_url: # Ensure link_url is not None or empty
                        normalized_link_url = link_url.rstrip('/')
                        # Ensure it's not a self-reference and not already added for this page
                        if normalized_link_url != normalized_page_url and normalized_link_url not in seen_on_this_page:
                            processed_links_for_this_page.append(link_url) # Store original link_url for display
                            seen_on_this_page.add(normalized_link_url)
                            unique_internal_links.add(link_url) # Add to site-wide unique links collection
                
                internal_links_by_page.append({
                    'page_url': page_url,
                    'links': processed_links_for_this_page,
                    'links_count': len(processed_links_for_this_page)
                })
        
        internal_links_by_page.sort(key=lambda x: x['links_count'], reverse=True)
        # 'internal_links_by_page' is now built directly from page_insights.
        # 'unique_internal_links' holds all unique internal URLs found anywhere.

        # Calculate Click Depth (using page_insights and the main site url 'url')
        click_depth_data = {0: [urlparse(url).path]} 
        page_to_depth = {urlparse(url).path: 0}
        queue_for_depth = [(urlparse(url).path, 0)]
        visited_for_depth = {urlparse(url).path}
        head = 0
        while head < len(queue_for_depth):
            current_path, depth = queue_for_depth[head]; head += 1
            if depth + 1 > 5: continue 
            links_from_current_page_for_depth = []
            for pi_page in page_insights: 
                if urlparse(pi_page['url']).path == current_path:
                    links_from_current_page_for_depth = pi_page.get('internal_links', [])
                    break
            for link_url_str_for_depth in links_from_current_page_for_depth:
                link_path = urlparse(link_url_str_for_depth).path
                if link_path not in visited_for_depth:
                    visited_for_depth.add(link_path)
                    page_to_depth[link_path] = depth + 1
                    if depth + 1 not in click_depth_data: click_depth_data[depth + 1] = []
                    click_depth_data[depth + 1].append(link_path)
                    queue_for_depth.append((link_path, depth + 1))
        report_click_depth = {str(k): len(v) for k, v in click_depth_data.items()}
        # This variable 'report_click_depth' is used by evaluate_click_depth_score & assigned to report['click_depth']

        # Generate Site Structure Visualization (using the new multi-level generate_text_tree)
        # Create the analyzed_pages_map needed for generate_text_tree
        analyzed_pages_map_for_tree = {p.get('url'): p for p in page_insights if p and p.get('url')}
        site_visualization = generate_text_tree(analyzed_pages_map_for_tree, url, max_display_depth=2)

        # Collect external links and referring domains
        external_domains = set()
        outbound_links = [] 
        domain_to_outbound_links = {}
        
        # Get the registered domain of the site being analyzed
        main_domain_info = tldextract.extract(final_site_url_for_checks) # final_site_url_for_checks is the base URL of the site
        main_registered_domain = main_domain_info.registered_domain

        for page in page_insights:
            external_links_list = page.get('external_links', [])
            if isinstance(external_links_list, list):
                for ext_link in external_links_list:
                    if not ext_link: continue # Skip None or empty links
                    try:
                        parsed_ext_link = urlparse(ext_link)
                        if not parsed_ext_link.netloc: # Skip links without a domain (e.g., mailto:, tel:)
                            continue

                        link_domain_info = tldextract.extract(ext_link)
                        link_registered_domain = link_domain_info.registered_domain

                        # Consider external if the registered domain is different
                        if link_registered_domain and link_registered_domain != main_registered_domain:
                            outbound_links.append(ext_link)
                            # Store the full netloc for display, but use registered_domain for grouping if needed
                            external_domains.add(parsed_ext_link.netloc) 
                            if parsed_ext_link.netloc not in domain_to_outbound_links:
                                domain_to_outbound_links[parsed_ext_link.netloc] = []
                            domain_to_outbound_links[parsed_ext_link.netloc].append(ext_link)
                    except Exception as e_parse_link:
                        logger.debug(f"Could not parse or process external link {ext_link}: {e_parse_link}")
                        pass 

        external_domains_list_data = []
        for dmn in sorted(list(external_domains)):
            external_domains_list_data.append({
                'domain': dmn,
                'backlinks': domain_to_outbound_links.get(dmn, []),
                'backlinks_count': len(domain_to_outbound_links.get(dmn, []))
            })
        external_domains_list_data.sort(key=lambda x: x['backlinks_count'], reverse=True)

        backlink_data = {
            'backlinks_count': len(outbound_links),
            'referring_domains': len(external_domains),
            'internal_links_count': len(internal_links_by_page),
            'all_backlinks': outbound_links,
            'referring_domains_list': external_domains_list_data,
            'all_internal_links': list(unique_internal_links),
            'internal_links_by_page': internal_links_by_page
        }
        # --- END OF FULL LINK ANALYSIS LOGIC ---

        # Step 5: Generate recommendations
        if progress_callback:
            progress_callback(70, 'Generating recommendations', 4)

        # Make sure we always generate recommendations even if we have error pages
        # First filter out completely failed pages where we don't have proper data
        filtered_insights = [page for page in page_insights if 'url' in page and 'issue_details' in page]

        if filtered_insights:
            recommendations = generate_recommendations(filtered_insights)
        else:
            recommendations = [
                {
                    'title': 'Fix website accessibility',
                    'description': 'Your website has accessibility issues. Make sure all pages can be accessed by search engines.',
                    'impact': 'high',
                    'effort': 'medium',
                    'detailed_reason': 'Search engines need to be able to access and crawl your pages to index them properly. If your pages return errors, they won\'t be included in search results.'
                }
            ]

        # Step 6: Calculate overall score using a more stable weighted method
        if progress_callback:
            progress_callback(90, 'Calculating overall score', 5)

        basic_score = calculate_score(page_insights)
        # Ensure 'performance' is defined before being used by calculate_weighted_score
        # It should have been defined in Step 4. If not, default to empty dict.
        weighted_score = calculate_weighted_score(page_insights, current_performance_data)
        overall_score = weighted_score

        # Compile technical_issues (MUST be fully compiled before UX analysis that uses it)
        technical_issues = []
        issue_types = {}
        if page_insights:
            for page in page_insights:
                for issue_text in page.get('issue_details', []):
                    issue_type_normalized = re.sub(r'\d+', 'X', issue_text).lower().replace(' ', '_')
                    # Refined regex to remove any parenthesized content and specific counts for cleaner type keys
                    issue_type_normalized = re.sub(r'_\(.*?\)', '', issue_type_normalized) 
                    issue_type_normalized = re.sub(r'_on_x_images', '_on_images', issue_type_normalized)
                    issue_type_normalized = re.sub(r'_x%', '_ratio', issue_type_normalized) # e.g. low_content-to-html_ratio
                    # Further specific normalizations can be added if needed
                    
                    if issue_type_normalized in issue_types:
                        issue_types[issue_type_normalized] += 1
                    else:
                        issue_types[issue_type_normalized] = 1
        
        for issue_type_key, count_val in issue_types.items():
            impact_level = 'medium' 
            if any(kw in issue_type_key for kw in ['title', 'h1', 'https', 'viewport', 'no_content']):
                impact_level = 'high'
            elif any(kw in issue_type_key for kw in ['alt_text', 'content-to-html_ratio']):
                impact_level = 'low'
            elif any(kw in issue_type_key for kw in ['meta description', 'internal linking', 'vary:_user-agent']):
                impact_level = 'medium'
            technical_issues.append({
                'type': issue_type_key,
                'count': count_val,
                'impact': impact_level
            })
        report['technical_issues'] = technical_issues # Assign to report object

        # --- Evaluate scores for architecture (Calls now refer to module-level functions) ---
        arch_click_depth_score = evaluate_click_depth_score(report_click_depth, len(page_insights))
        arch_internal_linking_health = evaluate_internal_linking_health(internal_links_by_page, len(page_insights))
        arch_url_structure_health = evaluate_url_structure_health(report.get('analyzed_urls', []))

        # --- Dynamically build UX Analysis factors ---
        ux_positive_factors = []
        ux_improvement_areas = []
        current_performance_data = report.get('performance', {})

        if current_performance_data.get('mobile_friendly', False):
            ux_positive_factors.append("Site appears responsive (mobile-friendly flag is true).")
        else:
            ux_improvement_areas.append("Improve site-wide mobile responsiveness (e.g., check viewport, CSS).")

        if current_performance_data.get('load_time', 5) < 2.0:
            ux_positive_factors.append(f"Good average page load time observed ({current_performance_data.get('load_time')}s).")
        elif current_performance_data.get('load_time', 0) > 3.5:
            ux_improvement_areas.append(f"Address overall average page load speed (currently {current_performance_data.get('load_time')}s).")
        
        if current_performance_data.get('has_ssl', False):
            ux_positive_factors.append("Secure browsing (HTTPS enabled).")
        else:
            ux_improvement_areas.append("Implement HTTPS across the site for security and user trust.")

        # UX factors based on architecture scores
        if arch_internal_linking_health == "Good":
            ux_positive_factors.append("Generally good internal linking structure observed.")
        elif arch_internal_linking_health == "Needs Improvement":
            ux_improvement_areas.append("Strengthen internal linking to improve navigation and SEO.")

        # UX factors based on compiled technical_issues
        missing_h1_count = 0
        missing_alt_text_count = 0
        for issue_item in technical_issues: # This should now be safe
            issue_type_str = issue_item.get('type', '')
            if issue_type_str == 'missing_h1_tag':
                missing_h1_count = issue_item.get('count', 0)
            if 'missing_alt_text' in issue_type_str:
                missing_alt_text_count += issue_item.get('count', 0)
        
        if missing_h1_count > 0:
            ux_improvement_areas.append(f"Ensure all key pages ({missing_h1_count} found with missing H1) have a clear H1 heading.")
        if missing_alt_text_count > 0:
            ux_improvement_areas.append(f"Improve image accessibility by adding descriptive alt text to images ({missing_alt_text_count} instances found).")

        if arch_click_depth_score == "Needs Improvement":
            ux_improvement_areas.append("Simplify site navigation to make content more accessible (reduce click depth).")
        if arch_url_structure_health == "Needs Improvement":
            ux_improvement_areas.append("Review and simplify URL structures for better readability and SEO.")
        
        if not ux_positive_factors: ux_positive_factors.append("Basic site elements appear functional.")
        if not ux_improvement_areas: ux_improvement_areas.append("Consider a full UX audit for detailed insights.")

        report['ux_analysis'] = {
            'positive_factors': ux_positive_factors,
            'improvement_areas': ux_improvement_areas
        }
        report['architecture_analysis'] = {
            'click_depth_score': arch_click_depth_score,
            'internal_linking': arch_internal_linking_health,
            'url_structure': arch_url_structure_health,
            'visualization': site_visualization # Now uses the new multi-level tree
        }

        # Populate report with data
        report['overall_score'] = overall_score
        report['recommendations'] = recommendations
        report['page_insights'] = page_insights
        report['performance'] = current_performance_data # Use the potentially defaulted one
        report['pages_analyzed'] = len(page_insights)
        report['backlink_data'] = backlink_data
        report['click_depth'] = report_click_depth

        # Final step
        if progress_callback:
            progress_callback(100, 'Report completed', 5)

        return report

    except Exception as e:
        logger.error(f"Error generating SEO report: {str(e)}")
        traceback.print_exc() # Print full traceback
        if progress_callback:
            progress_callback(100, f'Error: {str(e)}', 5)
        # Return the partially filled or default report object so frontend doesn't break on missing keys
        # Ensure all top-level keys expected by the template are present, even if empty
        final_report_keys = ['url', 'date', 'score', 'page_insights', 'recommendations', 'top_issues', 'performance', 'technical_issues', 'ux_analysis', 'architecture_analysis', 'click_depth', 'analyzed_urls', 'backlink_data']
        for key in final_report_keys:
            if key not in report:
                if key.endswith('_list') or key in ['page_insights', 'recommendations', 'technical_issues', 'analyzed_urls']:
                    report[key] = []
                elif key.endswith('_analysis') or key in ['performance', 'top_issues', 'click_depth', 'backlink_data']:
                    report[key] = {}
                else:
                    report[key] = None # Or appropriate default like 0 for score
        return report

def calculate_weighted_score(page_insights, performance):
    """Calculate a more stable weighted SEO score"""
    # Base calculation
    total_issues = sum(page.get('issues', 0) for page in page_insights)
    pages_count = len(page_insights)

    if pages_count == 0:
        return 50  # Default score if no pages analyzed

    # Get issue types for weighting
    high_impact_issues = 0
    medium_impact_issues = 0
    low_impact_issues = 0

    # Count issues by impact
    for page in page_insights:
        for issue in page.get('issue_details', []):
            issue_lower = issue.lower()
            if 'title' in issue_lower or 'h1' in issue_lower or 'https' in issue_lower or 'viewport' in issue_lower:
                high_impact_issues += 1
            elif 'meta description' in issue_lower or 'internal linking' in issue_lower:
                medium_impact_issues += 1
            else:
                low_impact_issues += 1

    # Calculate weighted issue score (high issues count more)
    weighted_issues = (high_impact_issues * 1.5) + (medium_impact_issues * 1.0) + (low_impact_issues * 0.5)

    # Calculate base score but with less penalty per issue
    base_score = max(0, min(100, 100 - (weighted_issues / pages_count * 8)))

    # Apply performance factors with reduced impact
    # Page speed is important but should not cause large fluctuations
    page_speed_factor = performance.get('page_speed', 70) / 100
    page_speed_weight = 0.15  # Reduce from default to minimize fluctuations

    # Mobile friendliness is a binary score
    mobile_factor = 1.0 if performance.get('mobile_friendly', False) else 0.85
    mobile_weight = 0.15

    # Calculate weighted score
    weighted_score = (base_score * 0.7) + (base_score * page_speed_factor * page_speed_weight) + (base_score * mobile_factor * mobile_weight)

    # Round to nearest integer
    return int(round(weighted_score))

if __name__ == "__main__":
    # For testing purposes
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "example.com"

    # Define a simple progress callback for testing
    def print_progress(progress, step, steps_completed):
        print(f"Progress: {progress}%, Step: {step}, Steps completed: {steps_completed}")

    # Generate the report
    report = generate_seo_report(url, progress_callback=print_progress)

    # Print the report summary
    print(f"\nSEO Report for {url}")
    print(f"Overall score: {report['overall_score']}/100")
    print(f"Pages analyzed: {report['pages_analyzed']}")
    print("\nTop recommendations:")
    for i, rec in enumerate(report['recommendations'][:3]):
        print(f"{i+1}. {rec['title']} ({rec['impact']} impact, {rec['effort']} effort)")
    print("\nDone!") 