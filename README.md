# Hura Tourism Chatbot

A FastAPI backend for a tourism chatbot serving Kigali, Rwanda visitors.

## Features
- Answers tourist questions about Kigali
- Provides recommendations and insights
- Retrieval-Augmented Generation (RAG) architecture
- Mistral-7B CPU-optimized model

## Deployment
Deployed on Render: [https://hura-chatbot.onrender.com](https://hura-chatbot.onrender.com)

## API Endpoint
POST `/ask` with JSON body:
```json
{
  "text": "Your question here"
}

