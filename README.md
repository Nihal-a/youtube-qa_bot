# 🤖 LLM Powered Assistants

A Streamlit web app with three AI-powered features built on **Google Gemini 2.5 Flash** and **LangChain**.

## Features

### 🎬 YouTube Video Q&A
Paste any YouTube URL and ask questions about the video. The app:
1. Downloads the video transcript via `YoutubeLoader`
2. Splits it into chunks and embeds them with Google Generative AI Embeddings
3. Stores chunks in a FAISS vector database
4. Retrieves the most relevant chunks and answers your question using Gemini

### 🐾 Pet Name Generator
Select a pet type and color, and get 5 creative name suggestions powered by Gemini.

### 🔍 Wikipedia Agent
A ReAct agent that searches Wikipedia to answer general-knowledge questions.

---

## Getting Started

### Prerequisites
- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/youtube-qa-bot.git
cd youtube-qa-bot

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
youtube-qa-bot/
├── app.py              # Streamlit UI
├── src/
│   └── helper.py       # LangChain logic (YouTube Q&A, pet names, agent)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Generative AI Embeddings (`gemini-embedding-001`) |
| Orchestration | LangChain |
| Vector store | FAISS |
| Transcripts | youtube-transcript-api |

---

## Notes

- YouTube videos must have captions/transcripts enabled.
- The free Google AI Studio tier has rate limits — if you hit quota errors, wait a moment and retry.
