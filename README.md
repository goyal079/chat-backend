# Chat Backend

A robust backend service for chat applications built with FastAPI and LangChain, featuring integration with OpenAI and Pinecone for vector storage.

## Features

- RESTful API endpoints using FastAPI
- Integration with OpenAI's latest models
- Vector storage with Pinecone
- LangChain for advanced language model operations

## Technical Overview

### API Endpoints

The application exposes two main API endpoints:

1. **Chat API (`/api/chat`)**

   - `POST /chat`: Send messages and receive AI responses
   - `GET /chat/history`: Retrieve conversation history
   - Request format:
     ```json
     {
       "message": "Your message here",
       "conversation_id": "optional-conversation-id"
     }
     ```

2. **Vector Store API (`/api/vector`)**
   - `POST /vector/store`: Store new documents
   - `GET /vector/search`: Search similar documents
   - Request format for search:
     ```json
     {
       "query": "Your search query",
       "top_k": 5
     }
     ```

### System Architecture

#### 1. Chat System

- Uses OpenAI's GPT models through LangChain
- Maintains conversation history
- Supports context-aware responses
- Configurable temperature and model settings

#### 2. Vector Storage System

- Pinecone for efficient vector storage
- Document embedding using OpenAI's embedding models
- Semantic search capabilities
- Scalable document storage

## Setup and Configuration

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key and environment

### Environment Variables

Create a `.env` file:

```plaintext
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
```

### Core Components

#### OpenAI Integration

```python
from openai import OpenAI
from langchain.chat_models import ChatOpenAI

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize LangChain's ChatOpenAI
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
```

### Installation

1. Clone the repository:

```bash
git clone [your-repository-url]
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

Start the server:

```bash
uvicorn main:app
```

```

Access the API at http://localhost:8000
```
