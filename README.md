# FinEcho

## Overview
FinEcho is a sophisticated AI-driven system designed to manage and interact with multiple agents for tasks such as financial data retrieval, summarization, and transcription. It leverages advanced language models and tools to provide insights and streamline workflows.

## Features
- **Supervisor Agent**: Manages multiple agents and coordinates tasks.
- **Scraper Agent**: Downloads SEC filings and processes them.
- **Retriever Agent**: Retrieves relevant documents from a vector database.
- **API Agent**: Interacts with external APIs for financial data.
- **Audio Transcription**: Captures audio prompts and converts them to text using Groq Whisper.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FinEcho
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the `supervisor_agent.py` script:
   ```bash
   python supervisor_agent.py
   ```
2. Speak into the microphone to provide a prompt. The system will transcribe your speech and pass it to the agents for processing.

## Project Structure
- **supervisor_agent.py**: Main script to run the supervisor and agents.
- **portfolio/**: Contains downloaded SEC filings.
- **extracts/**: Contains extracted data from SEC filings.
- **yahoo-finance-mcp/**: Yahoo Finance MCP server.
- **sec-edgar-mcp/**: SEC Edgar MCP server.

## Requirements
- Python 3.9+
- SpeechRecognition library for audio transcription.
- Groq Whisper API for transcription.
- LangChain for agent management.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Contact
For questions or support, please contact [prajithr004@gmail.com].
