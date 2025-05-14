import json
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

def display_analysis(json_file):
    """Display the SEO analysis results in a readable format."""
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found.")
        return
    
    # Load the JSON data
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not a valid JSON file.")
        return
    
    console = Console()
    
    # Print header
    console.print(f"\n[bold blue]SEO Analysis Results for {json_file}[/bold blue]\n")
    
    # Print scores
    scores_table = Table(title="SEO Scores")
    scores_table.add_column("Category", style="cyan")
    scores_table.add_column("Score", style="green")
    
    tech_score = data.get("technical_seo", {}).get("score", 0)
    onpage_score = data.get("on_page_seo", {}).get("score", 0)
    offpage_score = data.get("off_page_seo", {}).get("score", 0)
    
    overall_score = (tech_score + onpage_score + offpage_score) / 3
    
    scores_table.add_row("Technical SEO", f"{tech_score:.1f}/10")
    scores_table.add_row("On-Page SEO", f"{onpage_score:.1f}/10")
    scores_table.add_row("Off-Page SEO", f"{offpage_score:.1f}/10")
    scores_table.add_row("Overall Score", f"{overall_score:.1f}/10")
    
    console.print(scores_table)
    console.print()
    
    # Technical SEO details
    tech_seo = data.get("technical_seo", {})
    tech_panel = Panel(
        f"[bold]Site Speed:[/bold] {tech_seo.get('site_speed', {}).get('avg_load_time', 0):.2f} seconds\n"
        f"[bold]Mobile-Friendly Score:[/bold] {tech_seo.get('mobile_optimisation', {}).get('score', 0):.1f}/10\n"
        f"[bold]SSL/HTTPS:[/bold] {'Yes' if tech_seo.get('security', {}).get('has_ssl', False) else 'No'}\n"
        f"[bold]Robots.txt:[/bold] {'Found' if tech_seo.get('indexation', {}).get('robots_txt', {}).get('exists', False) else 'Not Found'}\n"
        f"[bold]Sitemap.xml:[/bold] {'Found' if tech_seo.get('indexation', {}).get('sitemap', {}).get('exists', False) else 'Not Found'}\n"
        f"[bold]Schema Markup:[/bold] {', '.join(tech_seo.get('structured_data', {}).get('schema_types', [])) or 'None'}\n",
        title="[bold cyan]Technical SEO Details[/bold cyan]"
    )
    console.print(tech_panel)
    
    # On-Page SEO details
    onpage_seo = data.get("on_page_seo", {})
    onpage_panel = Panel(
        f"[bold]Average Word Count:[/bold] {onpage_seo.get('content_quality', {}).get('avg_word_count', 0):.0f} words\n"
        f"[bold]Pages with Thin Content:[/bold] {onpage_seo.get('content_quality', {}).get('thin_content_pages', 0)}\n"
        f"[bold]Pages Missing Title:[/bold] {onpage_seo.get('meta_tags', {}).get('pages_without_title', 0)}\n"
        f"[bold]Pages Missing Meta Description:[/bold] {onpage_seo.get('meta_tags', {}).get('pages_without_description', 0)}\n"
        f"[bold]Pages Missing H1:[/bold] {onpage_seo.get('heading_structure', {}).get('pages_without_h1', 0)}\n"
        f"[bold]Images Missing Alt Text:[/bold] {onpage_seo.get('image_optimization', {}).get('images_without_alt', 0)}\n",
        title="[bold cyan]On-Page SEO Details[/bold cyan]"
    )
    console.print(onpage_panel)
    
    # Page-specific issues
    console.print("[bold cyan]Page-Specific Issues[/bold cyan]")
    
    for url, page_data in data.get("analyzed_pages", {}).items():
        console.print(f"\n[bold]Page:[/bold] {url}")
        console.print(f"[bold]Title:[/bold] {page_data.get('title', 'None')}")
        console.print(f"[bold]Description:[/bold] {page_data.get('description', 'None')}")
        console.print(f"[bold]Word Count:[/bold] {page_data.get('word_count', 0)}")
        console.print(f"[bold]Load Time:[/bold] {page_data.get('load_time', 0):.2f} seconds")
        
        if page_data.get("issues"):
            console.print("[bold]Issues:[/bold]")
            issues_table = Table(show_header=True, header_style="bold magenta")
            issues_table.add_column("Severity")
            issues_table.add_column("Issue")
            
            for issue in page_data.get("issues", []):
                severity = issue.get("severity", "").lower()
                severity_color = {
                    "high": "red",
                    "medium": "yellow",
                    "low": "green"
                }.get(severity, "white")
                
                issues_table.add_row(
                    f"[{severity_color}]{issue.get('severity', '').title()}[/{severity_color}]",
                    issue.get("description", "")
                )
            
            console.print(issues_table)
        else:
            console.print("[green]No issues detected[/green]")
    
    console.print("\n[bold blue]Analysis Complete[/bold blue]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_results.py <json_file>")
        sys.exit(1)
    
    display_analysis(sys.argv[1]) 