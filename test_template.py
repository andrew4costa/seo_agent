import os
from jinja2 import Environment, FileSystemLoader

# Set up the Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'))

try:
    # Try to load the template
    template = env.get_template('seo_report/report_template.html')
    print(f"Template loaded successfully: {template.name}")
    
    # Try rendering with minimal data
    result = template.render(
        title="Test Report",
        report_date="2023-05-13",
        url="https://example.com",
        overall_score=80,
        technical_score=75,
        onpage_score=85,
        pages_analyzed=5,
        page_speed="1.25",
        mobile_friendly="Yes",
        pages_with_issues=2,
        recommendations_by_category={},
        analyzed_pages={},
        tech_has_ssl="Yes",
        tech_mobile_friendly="Yes",
        tech_robots_txt="Yes",
        tech_sitemap="No",
        tech_load_time="1.5",
        tech_schema="None",
        onpage_word_count=500,
        onpage_thin_content=0,
        onpage_missing_titles=0,
        onpage_missing_desc=1,
        onpage_missing_h1=0,
        onpage_duplicate_titles=0,
        backlink_data={
            "referring_domains": 0,
            "backlinks_count": 0,
            "internal_links_count": 0,
            "referring_domains_list": [],
            "all_backlinks": [],
            "internal_links_by_page": {}
        },
        technical_issues=[],
        ux_analysis={
            "positive_factors": ["Test factor"],
            "improvement_areas": ["Test area"]
        },
        architecture_analysis={
            "click_depth_score": "Good",
            "internal_linking": "Good",
            "url_structure": "Good",
            "visualization": "Test visualization"
        },
        click_depth={"1": 1, "2": 2, "3": 3}
    )
    
    print("Template rendered successfully!")
    
    # Save the result to a file
    with open("test_report.html", "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"Test report saved to: {os.path.abspath('test_report.html')}")
    
except Exception as e:
    print(f"Error: {e}") 