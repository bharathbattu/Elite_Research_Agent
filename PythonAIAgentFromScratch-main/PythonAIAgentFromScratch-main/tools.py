import os
from typing import List
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from googleapiclient.discovery import build  # type: ignore
from dotenv import load_dotenv
import wikipedia  # type: ignore

load_dotenv()

def google_search(query: str, num_results: int = 5) -> str:
    """
    Perform a Google search using Google Custom Search API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 10)
        
    Returns:
        Formatted string with search results
    """
    try:
        # Get API credentials from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_CSE_ID")
        
        if not api_key or not search_engine_id:
            return "Error: Google API key or Search Engine ID not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file."
        
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)  # type: ignore
        
        # Perform the search
        result = service.cse().list(  # type: ignore
            q=query,
            cx=search_engine_id,
            num=min(num_results, 10)  # API limit is 10
        ).execute()
        
        # Format results
        if 'items' not in result:
            return f"No search results found for: {query}"
        
        formatted_results: List[str] = []
        for i, item in enumerate(result['items'], 1):  # type: ignore
            title = item.get('title', 'No title')  # type: ignore
            link = item.get('link', 'No link')  # type: ignore
            snippet = item.get('snippet', 'No description available')  # type: ignore
            
            formatted_results.append(f"{i}. {title}\n   URL: {link}\n   Description: {snippet}\n")
        
        return f"Google search results for '{query}':\n\n" + "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error performing Google search: {str(e)}"

def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Try to parse as structured research response for better formatting
    try:
        import json
        research_data = json.loads(data)
        
        # Format as a comprehensive research report
        formatted_text = f"""
===============================================================================
                        COMPREHENSIVE RESEARCH REPORT
===============================================================================
Generated: {timestamp}

RESEARCH TOPIC:
{research_data.get('topic', 'N/A')}

ABSTRACT / EXECUTIVE SUMMARY:
{research_data.get('abstract', 'N/A')}

DETAILED FINDINGS:
{research_data.get('detailed_findings', 'N/A')}

SOURCES / REFERENCES:
"""
        
        for i, source in enumerate(research_data.get('sources', []), 1):
            formatted_text += f"{i}. {source}\n"
        
        formatted_text += f"""
RESEARCH METHODOLOGY & TOOLS USED:
"""
        for tool in research_data.get('tools_used', []):
            formatted_text += f"• {tool}\n"
        
        formatted_text += f"""
KEY INSIGHTS:
"""
        for insight in research_data.get('key_insights', []):
            formatted_text += f"• {insight}\n"
        
        formatted_text += f"""
===============================================================================
End of Research Report
===============================================================================

"""
    except:
        # Fallback to simple format if JSON parsing fails
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Research report successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search_tool = Tool(
    name="google_search",
    func=google_search,
    description="Search the web using Google Custom Search API for current and comprehensive information",
)

api_wrapper = WikipediaAPIWrapper(
    wiki_client=wikipedia,
    top_k_results=1, 
    doc_content_chars_max=100
)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

