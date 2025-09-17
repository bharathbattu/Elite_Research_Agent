# Elite Research Agent

A sophisticated AI-powered research assistant built with LangChain, Mistral AI, and Streamlit. This application provides comprehensive academic-quality research reports by intelligently combining web search, Wikipedia knowledge, and AI analysis.

## 🌟 Features

- **Advanced AI Research**: Powered by Mistral AI language models
- **Multi-Source Intelligence**: Integrates Google Search API and Wikipedia
- **Academic-Quality Output**: Generates publication-ready research reports
- **Professional Web Interface**: Clean, responsive Streamlit UI
- **Export Capabilities**: Download reports as TXT or HTML/PDF formats
- **Research History**: Session-based history management
- **Structured Analysis**: Comprehensive reports with background, findings, challenges, and insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Search API credentials
- Mistral AI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PythonAIAgentFromScratch-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Copy `sample.env` to `.env` and add your API keys:
   ```bash
   MISTRAL_API_KEY=your_mistral_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CSE_ID=your_google_custom_search_engine_id_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the interface**
   
   Open your browser and navigate to: `http://localhost:8501`

## 🏗️ Architecture

### Core Components

- **`app.py`** - Main Streamlit web application with UI and user interactions
- **`main.py`** - Core research agent logic and AI prompt engineering
- **`tools.py`** - Research tools (Google Search, Wikipedia, File I/O)
- **`requirements.txt`** - Python dependencies
- **`sample.env`** - Environment configuration template

### Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **AI Framework**: LangChain with tool-calling agents
- **Language Model**: Mistral AI (Small/Medium/Large models)
- **Search Integration**: Google Custom Search API
- **Knowledge Base**: Wikipedia API
- **Export**: ReportLab for PDF generation
- **Configuration**: python-dotenv for environment management

## 🔧 API Setup

### Google Custom Search API

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Custom Search API"
4. Create credentials (API Key)
5. Set up a Custom Search Engine at [Google CSE](https://cse.google.com/)
6. Get your Search Engine ID

### Mistral AI API

1. Visit [Mistral AI](https://mistral.ai/)
2. Sign up for an account
3. Generate an API key from your dashboard
4. Choose from available models: mistral-small, mistral-medium, mistral-large

## 📖 Usage

1. **Enter Research Query**: Input your research topic or question
2. **Select AI Model**: Choose between Mistral Small, Medium, or Large
3. **Run Research**: Click the research button to start analysis
4. **Review Results**: Get comprehensive structured reports with:
   - Refined academic title
   - Executive summary (150-250 words)
   - Detailed multi-section findings
   - Source citations
   - Key insights and takeaways
5. **Export Results**: Download as TXT or PDF format
6. **Access History**: View and reload previous research queries

## 📊 Report Structure

Each research report includes:

- **Topic**: Academically refined research title
- **Abstract**: Professional executive summary
- **Detailed Findings**: Multi-section analysis covering:
  - Background and context
  - Current developments
  - Challenges and limitations
  - Future outlook and implications
- **Sources**: APA/MLA formatted citations
- **Methodology**: Research tools and approaches used
- **Key Insights**: 3-5 critical takeaways

## 🛠️ Customization

### Adding New Research Tools

Extend the `tools.py` file to add new research capabilities:

```python
def custom_tool(query: str) -> str:
    # Your custom research logic here
    return "Research results"

# Register as LangChain tool
custom_search_tool = Tool(
    name="custom_search",
    description="Description of your tool",
    func=custom_tool
)
```

### Modifying AI Prompts

Update the research prompts in `main.py` to customize:
- Report structure and format
- Research depth and focus areas
- Output style and academic standards

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Google Custom Search API](https://developers.google.com/custom-search)

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Review the `STREAMLIT_GUIDE.md` for detailed setup instructions
3. Ensure all API keys are correctly configured
4. Verify your Python environment and dependencies

---


Built with ❤️ using Python, LangChain, and Mistral AI
