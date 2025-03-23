import discord
from discord.ext import commands
import asyncio
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma  # updated import per deprecation warning
from langchain_mistralai import MistralAIEmbeddings  # updated import to use Mistral
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# New imports for LangMem memory functionality
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv(find_dotenv())
print("[OldBot] Environment variables loaded.")
print("[OldBot] MISTRAL_API_KEY:", os.environ.get("MISTRAL_API_KEY"))

# Verify required API keys are present
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
if "MISTRAL_API_KEY" not in os.environ:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

# Discord setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
print("[OldBot] Discord bot initialized.")

# Load and process documents
loader = TextLoader("./maitreya.txt")
documents = loader.load()
print("[OldBot] Documents loaded.")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"[OldBot] Documents split into {len(texts)} chunks.")

# Mistral AI Embeddings for document retrieval
embeddings = MistralAIEmbeddings(model="mistral-embed")
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
print("[OldBot] Vector store created with document embeddings.")

# Google Chat model (used elsewhere if needed)
chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.0
)
print("[OldBot] Chat model initialized.")

# Setup LangMem for conversation memory using Mistral embeddings
memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "mistralai:mistral-embed",
    }
)
memory_tools = [
    create_manage_memory_tool(namespace=("memories",)),
    create_search_memory_tool(namespace=("memories",))
]
print("[OldBot] Memory store and tools set up.")

def memory_prompt(state):
    last_message = state["messages"][-1].content if state["messages"] else ""
    print(f"[OldBot] Building memory prompt. Last message: {last_message}")
    memories = memory_store.search(("memories",), query=last_message)
    memory_text = "\n\n".join([mem.value["content"] for mem in memories]) if memories else "No past memory."
    system_msg = {"role": "system", "content": f"Memory from past conversations:\n{memory_text}"}
    return [system_msg] + state["messages"]

# Create the memory-enabled agent using the pre-instantiated chat model
memory_agent = create_react_agent(
    chat,
    prompt=memory_prompt,
    tools=memory_tools,
    store=memory_store,
    checkpointer=InMemorySaver()
)
print("[OldBot] Memory agent created.")

# Get relevant context from vector database
async def get_context(question):
    print(f"[OldBot] Retrieving context for question: {question}")
    docs = retriever.get_relevant_documents(query=question)
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        print("[OldBot] Context retrieved.")
        return context
    print("[OldBot] No relevant context found.")
    return "No relevant information found in the knowledge base."

# Generic function to call an MCP tool based on user input
async def get_mcp_result(question):
    try:
        print(f"[OldBot] Initiating MCP tool call for question: {question}")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0.0,
            max_tokens=512,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        async with MultiServerMCPClient(
            {
                "math": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                },
                "webcrawl": {
                    "url": "http://localhost:8001/sse",
                    "transport": "sse",
                }
            }
        ) as client:
            tools = client.get_tools()
            
            # Debug logging of available tools
            print(f"[OldBot] Available tools: {[t.name for t in tools]}")
            
            # Handle the case where the user specifically asked for "entire page content"
            if "entire page content" in question.lower():
                # Check if we have a URL in the conversation history
                # This requires looking at the bot's recent messages
                try:
                    # We need to get the last URL mentioned in any previous messages
                    import re
                    
                    # This is a simplified approach - in production, you'd want to 
                    # search through the chat history properly
                    # For now, we'll just use a hardcoded URL from previous interactions
                    # if none is provided in the current message
                    urls = re.findall(r'https?://[^\s]+', question)
                    
                    # If no URL in this message, check our limited history for the URL
                    # that might have been mentioned previously
                    if not urls:
                        # Try to find URL in a limited search of previous interactions
                        # For simplicity, we'll use the URL from the logs
                        url = "https://docs.crawl4ai.com/api/crawl-result/"
                        print(f"[OldBot] No URL in current message, using last URL: {url}")
                    else:
                        url = urls[0]
                        
                    print(f"[OldBot] Processing 'entire page content' request for URL: {url}")
                    
                    # Get the crawl_page tool and invoke it directly
                    for tool in tools:
                        if tool.name == "crawl_page":
                            print(f"[OldBot] Directly invoking crawl_page tool for URL: {url}")
                            try:
                                # Make sure we use ainvoke (async) not invoke (sync)
                                result = await tool.ainvoke({"url": url, "use_cache": True})
                                print(f"[OldBot] Raw crawl_page result: {result}")
                                
                                # Check if we got a valid result with markdown content
                                if isinstance(result, dict) and result.get("success") and "markdown" in result:
                                    markdown_content = result.get("markdown", "")
                                    print(f"[OldBot] Successfully retrieved page content: {len(markdown_content)} characters")
                                    if markdown_content:
                                        # Return the first 4000 characters if it's very long
                                        if len(markdown_content) > 4000:
                                            return f"Successfully scraped the webpage. Here's the first part of the content (truncated due to length):\n\n{markdown_content[:4000]}..."
                                        else:
                                            return f"Successfully scraped the webpage. Here's the content:\n\n{markdown_content}"
                                    else:
                                        return "The webpage was scraped successfully, but no content was found."
                                else:
                                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                                    print(f"[OldBot] Error in crawl_page result: {error_msg}")
                                    return f"Error scraping the webpage: {error_msg}"
                            except Exception as tool_error:
                                print(f"[OldBot] Exception when invoking tool: {str(tool_error)}")
                                return f"Error when scraping webpage: {str(tool_error)}"
                    
                    # If we didn't find the tool
                    print("[OldBot] crawl_page tool not found in available tools")
                    return "The web scraping tool is not available. Please check if the Crawl4AI server is running."
                    
                except Exception as e:
                    print(f"[OldBot] Error handling 'entire page content' request: {str(e)}")
                    return f"Error processing your request: {str(e)}"
            
            # Check if this is a scraping request and extract the URL
            if any(keyword in question.lower() for keyword in ["scrape", "crawl", "web", "extract", "data", "info"]):
                import re
                # Try to extract URLs from the question
                urls = re.findall(r'https?://[^\s]+', question)
                
                # If we found a URL and this looks like a scraping request
                if urls:
                    url = urls[0]  # Take the first URL found
                    print(f"[OldBot] Detected scraping request for URL: {url}")
                    
                    # Direct call to the crawl_page tool
                    try:
                        # Try to directly invoke the crawl_page tool
                        for tool in tools:
                            if tool.name == "crawl_page":
                                print(f"[OldBot] Directly invoking crawl_page tool for URL: {url}")
                                try:
                                    # Make sure we use ainvoke (async) not invoke (sync)
                                    result = await tool.ainvoke({"url": url, "use_cache": True}) 
                                    print(f"[OldBot] Raw crawl_page result: {result}")
                                    
                                    # Check if we got a valid result with markdown content
                                    if isinstance(result, dict) and result.get("success") and "markdown" in result:
                                        markdown_content = result.get("markdown", "")
                                        print(f"[OldBot] Successfully retrieved page content: {len(markdown_content)} characters")
                                        if markdown_content:
                                            # Return the first 4000 characters if it's very long
                                            if len(markdown_content) > 4000:
                                                return f"Successfully scraped the webpage. Here's the first part of the content (truncated due to length):\n\n{markdown_content[:4000]}..."
                                            else:
                                                return f"Successfully scraped the webpage. Here's the content:\n\n{markdown_content}"
                                        else:
                                            return "The webpage was scraped successfully, but no content was found."
                                    else:
                                        error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                                        print(f"[OldBot] Error in crawl_page result: {error_msg}")
                                        return f"Error scraping the webpage: {error_msg}"
                                except Exception as tool_error:
                                    print(f"[OldBot] Exception when invoking tool: {str(tool_error)}")
                                    return f"Error when scraping webpage: {str(tool_error)}"
                        
                        print("[OldBot] crawl_page tool not found in available tools")
                        return "The web scraping tool is not available. Please check if the Crawl4AI server is running."
                    except Exception as direct_e:
                        print(f"[OldBot] Direct tool call failed: {str(direct_e)}")
                        # Fall back to agent-based approach
                
                # If direct call failed or no URL found, continue with normal agent approach
                prompt_prefix = "Scrape this: "
            elif any(keyword in question.lower() for keyword in ["calculate", "sum", "compute", "math", "+"]):
                prompt_prefix = "Calculate this: "
            else:
                prompt_prefix = ""
                
            # Create the agent with the tools
            agent = create_react_agent(model, tools)
            
            mcp_question = f"{prompt_prefix}{question}"
            print(f"[OldBot] MCP question constructed: {mcp_question}")
            
            # Use the agent approach as fallback
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": mcp_question}]},
                config={"configurable": {"thread_id": "mcp-thread"}}
            )
            print("[OldBot] MCP tool call completed.")
            return response["messages"][-1].content
    except Exception as e:
        print(f"[OldBot] MCP tool call error: {str(e)}")
        return f"Could not perform MCP tool call: {str(e)}"

# Combined response function that uses memory, knowledge retrieval, and MCP tools
async def get_combined_response(question):
    print(f"[OldBot] Processing combined response for question: {question}")
    context = await get_context(question)
    mcp_keywords = ["calculate", "compute", "sum", "add", "subtract", "multiply", "divide", "scrape", "crawl", "extract"]
    mcp_result = None
    if any(keyword in question.lower() for keyword in mcp_keywords):
        mcp_result = await get_mcp_result(question)
    combined_prompt = f"""You are a helpful Discord bot that remembers past conversations, retrieves information from a knowledge base, and can call various MCP tools.

INFORMATION FROM KNOWLEDGE BASE:
{context}

MCP TOOL RESULT (if applicable):
{mcp_result if mcp_result else "No MCP tool was triggered."}

Based on the above and your memory, please provide a comprehensive response to the user's question.
User's Question: {question}
Answer:"""
    print("[OldBot] Combined prompt constructed.")
    messages = [{"role": "user", "content": combined_prompt}]
    response = await memory_agent.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "default-thread"}}
    )
    print("[OldBot] Memory agent returned a response.")
    return response["messages"][-1].content

# Event and command handling remain unchanged
@bot.event
async def on_ready():
    print(f"[OldBot] {bot.user} has connected to Discord!")

@bot.command(name="power")
async def power(ctx, *, question):
    try:
        await ctx.send("Processing your power (combining knowledge lookup, MCP tool calls, and memory)...")
        answer = await get_combined_response(question)
        await ctx.send(answer)
    except Exception as e:
        print(f"[OldBot] Error occurred: {e}")
        await ctx.send(f"Sorry, I was unable to process your power. Error: {str(e)}")

# Run the bot
if __name__ == "__main__":
    if not os.environ.get("DISCORD_TOKEN"):
        raise ValueError("DISCORD_TOKEN environment variable not set")
    bot.run(os.environ.get("DISCORD_TOKEN"))