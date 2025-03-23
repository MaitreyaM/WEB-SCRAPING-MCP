from mcp.server.fastmcp import FastMCP
import asyncio
import json
from typing import Optional, Dict, List, Any, Union

# Import Crawl4AI components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy
)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("[CrawlServer] Environment variables loaded.")

# Create an MCP server for web crawling
mcp = FastMCP("WebCrawl")
mcp.settings.port = 8001
print("[CrawlServer] MCP server 'WebCrawl' created on port 8001.")

# Create a shared crawler instance
crawler = None
browser_config = BrowserConfig(
    headless=True,
    java_script_enabled=True,
    ignore_https_errors=True,
    viewport_width=1280,
    viewport_height=800,
)
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Helper function to get or create crawler
async def get_crawler():
    global crawler
    if crawler is None:
        print("[CrawlServer] Initializing new crawler instance.")
        crawler = await AsyncWebCrawler(config=browser_config).__aenter__()
    else:
        print("[CrawlServer] Using existing crawler instance.")
    return crawler

# Basic crawl tool - returns markdown content
@mcp.tool()
async def crawl_page(url: str, use_cache: bool = True) -> Dict[str, Any]:
    print(f"[CrawlServer] Starting crawl_page for URL: {url}")
    crawler = await get_crawler()
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED if use_cache else CacheMode.BYPASS,
        word_count_threshold=10,
        remove_overlay_elements=True,
    )
    result = await crawler.arun(url=url, config=run_config)
    if not result.success:
        print(f"[CrawlServer] crawl_page failed for URL: {url} with error: {result.error_message}")
        return {
            "success": False,
            "error": result.error_message,
            "status_code": result.status_code
        }
    print(f"[CrawlServer] crawl_page succeeded for URL: {url}")
    
    # Safely get attributes with fallback values
    title = getattr(result, 'title', 'No title available')
    word_count = getattr(result, 'word_count', 0)
    
    # Safely handle links and images that might be missing or have different structure
    links_internal = getattr(result.links, 'internal', []) if hasattr(result, 'links') else []
    links_external = getattr(result.links, 'external', []) if hasattr(result, 'links') else []
    images = getattr(result.media, 'images', []) if hasattr(result, 'media') and isinstance(result.media, dict) else []
    
    return {
        "success": True,
        "title": title,
        "url": result.url,
        "markdown": result.markdown.raw_markdown,
        "word_count": word_count,
        "links_count": len(links_internal) + len(links_external),
        "images_count": len(images),
    }

# Get filtered markdown with content filtering
@mcp.tool()
async def get_filtered_content(url: str, query: Optional[str] = None, threshold: float = 0.4) -> Dict[str, Any]:
    print(f"[CrawlServer] Starting get_filtered_content for URL: {url} with query: {query}")
    crawler = await get_crawler()
    if query:
        from crawl4ai.content_filter_strategy import BM25ContentFilter
        content_filter = BM25ContentFilter(query=query, threshold=threshold)
    else:
        content_filter = PruningContentFilter(threshold=threshold, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
    )
    result = await crawler.arun(url=url, config=run_config)
    if not result.success:
        print(f"[CrawlServer] get_filtered_content failed for URL: {url} with error: {result.error_message}")
        return {
            "success": False,
            "error": result.error_message,
            "status_code": result.status_code
        }
    print(f"[CrawlServer] get_filtered_content succeeded for URL: {url}")
    return {
        "success": True,
        "title": result.title,
        "url": result.url,
        "raw_markdown": result.markdown.raw_markdown,
        "filtered_markdown": result.markdown.fit_markdown,
        "word_count": result.word_count,
    }

# Extract structured data using CSS selectors
@mcp.tool()
async def extract_structured_data(url: str, schema: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    print(f"[CrawlServer] Starting extract_structured_data for URL: {url} with schema: {schema}")
    crawler = await get_crawler()
    extraction_strategy = JsonCssExtractionStrategy(schema)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED if use_cache else CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
    )
    result = await crawler.arun(url=url, config=run_config)
    if not result.success:
        print(f"[CrawlServer] extract_structured_data failed for URL: {url} with error: {result.error_message}")
        return {
            "success": False,
            "error": result.error_message,
            "status_code": result.status_code
        }
    extracted_data = json.loads(result.extracted_content) if result.extracted_content else {}
    print(f"[CrawlServer] extract_structured_data succeeded for URL: {url}")
    return {
        "success": True,
        "url": result.url,
        "data": extracted_data,
    }

# Generate CSS extraction schema using LLM
@mcp.tool()
async def generate_extraction_schema(url: str, description: str, llm_provider: str = "openai/gpt-3.5-turbo") -> Dict[str, Any]:
    print(f"[CrawlServer] Starting generate_extraction_schema for URL: {url} with description: {description}")
    if not openai_api_key and "openai" in llm_provider:
        return {"success": False, "error": "OpenAI API key is required for this operation"}
    
    try:
        crawler = await get_crawler()
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            return {"success": False, "error": result.error_message, "status_code": result.status_code}
        
        # Safely get HTML content
        html_content = getattr(result, 'html', '')
        if not html_content:
            return {"success": False, "error": "No HTML content found in the crawl result"}
            
        html_sample = html_content[:10000]
        
        api_token = openai_api_key if "openai" in llm_provider else None
        llm_config = LLMConfig(provider=llm_provider, api_token=api_token)
        schema = await JsonCssExtractionStrategy.generate_schema(
            html=html_sample,
            llm_config=llm_config,
            instruction=description
        )
        
        print(f"[CrawlServer] generate_extraction_schema succeeded for URL: {url}")
        return {"success": True, "schema": schema}
    except Exception as e:
        print(f"[CrawlServer] generate_extraction_schema failed with exception: {str(e)}")
        return {"success": False, "error": f"Exception occurred: {str(e)}"}

# Extract data using LLM
@mcp.tool()
async def extract_with_llm(url: str, instruction: str, llm_provider: str = "openai/gpt-3.5-turbo") -> Dict[str, Any]:
    print(f"[CrawlServer] Starting extract_with_llm for URL: {url} with instruction: {instruction}")
    if not openai_api_key and "openai" in llm_provider:
        return {"success": False, "error": "OpenAI API key is required for this operation"}
    
    try:
        crawler = await get_crawler()
        api_token = openai_api_key if "openai" in llm_provider else None
        llm_config = LLMConfig(provider=llm_provider, api_token=api_token)
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            instruction=instruction,
            extraction_type="free_text",
            extra_args={"temperature": 0}
        )
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
        )
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            return {"success": False, "error": result.error_message, "status_code": result.status_code}
        
        # Safely get extracted content
        extracted_content = getattr(result, 'extracted_content', '')
        
        print(f"[CrawlServer] extract_with_llm succeeded for URL: {url}")
        return {"success": True, "url": result.url, "extracted_content": extracted_content}
    except Exception as e:
        print(f"[CrawlServer] extract_with_llm failed with exception: {str(e)}")
        return {"success": False, "error": f"Exception occurred: {str(e)}"}

# Run multiple URL crawling in parallel
@mcp.tool()
async def crawl_multiple_urls(urls: List[str], max_concurrent: int = 5, get_markdown: bool = True) -> Dict[str, Any]:
    print(f"[CrawlServer] Starting crawl_multiple_urls for URLs: {urls}")
    crawler = await get_crawler()
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=True)
    results = []
    try:
        async for result in await crawler.arun_many(urls, config=run_config):
            if result.success:
                # Safely get attributes with fallback values
                title = getattr(result, 'title', 'No title available')
                word_count = getattr(result, 'word_count', 0)
                
                url_result = {
                    "success": True,
                    "url": result.url,
                    "title": title,
                    "status_code": result.status_code,
                    "word_count": word_count,
                }
                if get_markdown:
                    # Safely handle markdown content
                    markdown_content = ""
                    if hasattr(result, 'markdown') and hasattr(result.markdown, 'raw_markdown'):
                        markdown_content = result.markdown.raw_markdown
                    url_result["markdown"] = markdown_content
                results.append(url_result)
            else:
                results.append({
                    "success": False,
                    "url": result.url,
                    "error": result.error_message,
                    "status_code": result.status_code
                })
        print(f"[CrawlServer] crawl_multiple_urls completed for {len(results)} URLs.")
    except Exception as e:
        print(f"[CrawlServer] crawl_multiple_urls encountered an error: {str(e)}")
        return {"success": False, "error": str(e), "partial_results": results}
    return {"success": True, "total": len(results), "results": results}

# Cleanup function
async def cleanup():
    global crawler
    if crawler is not None:
        print("[CrawlServer] Cleaning up crawler instance.")
        await crawler.__aexit__(None, None, None)
        crawler = None

# Start the server
if __name__ == "__main__":
    try:
        print("[CrawlServer] Starting Crawl4AI MCP Server...")
        print("[CrawlServer] Available endpoints:")
        print(" - crawl_page: Basic web page crawling")
        print(" - get_filtered_content: Get filtered markdown content")
        print(" - extract_structured_data: Extract structured data using CSS")
        print(" - generate_extraction_schema: Generate CSS schema using LLM")
        print(" - extract_with_llm: Extract information using LLM")
        print(" - crawl_multiple_urls: Crawl multiple URLs in parallel")
        
        mcp.run(transport="sse")
    finally:
        asyncio.run(cleanup())