# Crawl4AI Web Scraper MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add license -->

This project provides an MCP (Model Context Protocol) server that uses the **[crawl4ai](https://github.com/unclecode/crawl4ai)** library to perform web scraping and intelligent content extraction tasks. It allows AI agents (like Claude, or agents built with LangChain/LangGraph) to interact with web pages, retrieve content, search for specific text, and perform LLM-based extraction based on natural language instructions.

This server uses:

*   **[FastMCP](https://github.com/model-context-protocol/mcp-py/blob/main/docs/fastmcp.md):** For creating the MCP server endpoint.
*   **[crawl4ai](https://github.com/unclecode/crawl4ai):** For the core web crawling and extraction logic.
*   **[dotenv](https://github.com/theskumar/python-dotenv):** For managing API keys via a `.env` file.
*   **(Optional) Docker:** For containerized deployment, bundling Python and dependencies.

## Features

*   Exposes MCP tools for web interaction:
    *   `scrape_url`: Get the full content of a webpage in Markdown format.
    *   `extract_text_by_query`: Find specific text snippets on a page based on a query.
    *   `smart_extract`: Use an LLM (currently Google Gemini) to extract structured information based on instructions.
*   Configurable via environment variables (API keys).
*   Includes Docker configuration (`Dockerfile`) for easy, self-contained deployment.
*   Communicates over Server-Sent Events (SSE) on port 8002 by default.

## Exposed MCP Tools

### `scrape_url`

Scrape a webpage and return its content in Markdown format.

**Arguments:**

*   `url` (str, **required**): The URL of the webpage to scrape.

**Returns:**

*   (str): The webpage content in Markdown format, or an error message.

### `extract_text_by_query`

Extract relevant text snippets from a webpage that contain a specific search query. Returns up to the first 5 matches found.

**Arguments:**

*   `url` (str, **required**): The URL of the webpage to search within.
*   `query` (str, **required**): The text query to search for (case-insensitive).
*   `context_size` (int, *optional*): The number of characters to include before and after the matched query text in each snippet. Defaults to `300`.

**Returns:**

*   (str): A formatted string containing the found text snippets or a message indicating no matches were found, or an error message.

### `smart_extract`

Intelligently extract specific information from a webpage using the configured LLM (currently requires Google Gemini API key) based on a natural language instruction.

**Arguments:**

*   `url` (str, **required**): The URL of the webpage to analyze and extract from.
*   `instruction` (str, **required**): Natural language instruction specifying what information to extract (e.g., "List all the speakers mentioned on this page", "Extract the main contact email address", "Summarize the key findings").

**Returns:**

*   (str): The extracted information (often formatted as JSON or structured text based on the instruction) or a message indicating no relevant information was found, or an error message (including if the required API key is missing).

## Setup and Running

You can run this server either locally or using the provided Docker configuration.

### Option 1: Running with Docker (Recommended for Deployment)

This method bundles Python and all necessary libraries. You only need Docker installed on the host machine.

1.  **Install Docker:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your OS. Start Docker Desktop.
2.  **Clone Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your repo URL
    cd your-repo-name
    ```
3.  **Create `.env` File:** Create a file named `.env` in the project root directory and add your API keys:
    ```.env
    # Required for the smart_extract tool
    GOOGLE_API_KEY=your_google_ai_api_key_here

    # Optional, checked by server but not currently used by tools
    # OPENAI_API_KEY=your_openai_key_here
    # MISTRAL_API_KEY=your_mistral_key_here
    ```
4.  **Build the Docker Image:**
    ```bash
    docker build -t crawl4ai-mcp-server .
    ```
5.  **Run the Container:** This starts the server, making port 8002 available on your host machine. It uses `--env-file` to securely pass the API keys from your local `.env` file into the container's environment.
    ```bash
    docker run -it --rm -p 8002:8002 --env-file .env crawl4ai-mcp-server
    ```
    *   `-it`: Runs interactively.
    *   `--rm`: Removes container on exit.
    *   `-p 8002:8002`: Maps host port 8002 to container port 8002.
    *   `--env-file .env`: Loads environment variables from your local `.env` file into the container. **Crucial for API keys.**
    *   `crawl4ai-mcp-server`: The name of the image you built.
6.  **Server is Running:** Logs will appear, indicating the server is listening on SSE (`http://0.0.0.0:8002`).
7.  **Connecting Client:** Configure your MCP client (e.g., LangChain agent) to connect to `http://127.0.0.1:8002/sse` with `transport: "sse"`.

### Option 2: Running Locally

This requires Python and manual installation of dependencies on your host machine.

1.  **Install Python:** Ensure Python >= 3.9 (check `crawl4ai` requirements if needed, 3.10+ recommended).
2.  **Clone Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your repo URL
    cd your-repo-name
    ```
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
    *(Or use Conda: `conda create --name crawl4ai-env python=3.11 -y && conda activate crawl4ai-env`)*
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Create `.env` File:** Create a file named `.env` in the project root directory and add your API keys (same content as in Docker setup step 3).
6.  **Run the Server:**
    ```bash
    python your_server_script_name.py # e.g., python webcrawl_mcp_server.py
    ```
7.  **Server is Running:** It will listen on `http://127.0.0.1:8002/sse`.
8.  **Connecting Client:** Configure your MCP client to connect to `http://127.0.0.1:8002/sse`.

## Environment Variables

The server uses the following environment variables, typically loaded from an `.env` file:

*   `GOOGLE_API_KEY`: **Required** for the `smart_extract` tool to function (uses Google Gemini). Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
*   `OPENAI_API_KEY`: Checked for existence but **not currently used** by any tool in this version.
*   `MISTRAL_API_KEY`: Checked for existence but **not currently used** by any tool in this version.

## Example Agent Interaction

```
# Example using the agent CLI from the previous setup

You: scrape_url https://example.com
Agent: Thinking...
[Agent calls scrape_url tool]
Agent: [Markdown content of example.com]
------------------------------
You: extract text from https://en.wikipedia.org/wiki/Web_scraping using the query "ethical considerations"
Agent: Thinking...
[Agent calls extract_text_by_query tool]
Agent: Found X matches for 'ethical considerations' on the page. Here are the relevant sections:
Match 1:
... text snippet ...
---
Match 2:
... text snippet ...
------------------------------
You: Use smart_extract on https://blog.google/technology/ai/google-gemini-ai/ to get the main points about Gemini models
Agent: Thinking...
[Agent calls smart_extract tool with Google API Key]
Agent: Successfully extracted information based on your instruction:
{
  "main_points": [
    "Gemini is Google's most capable AI model family (Ultra, Pro, Nano).",
    "Designed to be multimodal, understanding text, code, audio, image, video.",
    "Outperforms previous models on various benchmarks.",
    "Being integrated into Google products like Bard and Pixel."
  ]
}

```

## Files

*   `your_server_script_name.py`: The main Python script for the MCP server (e.g., `webcrawl_mcp_server.py`).
*   `Dockerfile`: Instructions for building the Docker container image.
*   `requirements.txt`: Python dependencies.
*   `.env.example`: (Recommended) An example environment file showing needed keys. **Do not commit your actual `.env` file.**
*   `.gitignore`: Specifies intentionally untracked files for Git (should include `.env`).
*   `README.md`: This file.

## Contributing

(Add contribution guidelines if desired)

## License

(Specify your license, e.g., MIT License)