#!/usr/bin/env python3
"""
Multi-Agent RAG Query Tool for ArXiv Daily with Conversation History

This script allows you to ask questions about papers in your database and other knowledge sources
using a multi-agent system that combines specialized expertise and maintains conversation history.
Compatible with Alibaba Qwen family models.
"""

import os
import argparse
import sys
import json
from typing import List, Dict, Any
import uuid
import time
import requests

try:
    from duckduckgo_search import DDGS
except ImportError:
    print("DuckDuckGo search package not found. Install with: pip install duckduckgo-search")
    DDGS = None

from arxiv_daily.rag.query_engine import RAGQueryEngine

class LLMClient:
    """Abstract base class for LLM clients"""
    
    def chat_completion(self, messages, model, response_format=None, temperature=0.3):
        """Generate a chat completion"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_embedding(self, text, model):
        """Get embedding for text"""
        raise NotImplementedError("Subclasses must implement this method")


class QwenClient(LLMClient):
    """Client for Alibaba Qwen models via DashScope API"""
    
    def __init__(self, api_key=None):
        """Initialize the Qwen client"""
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable or api_key parameter must be provided")
        
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages, model="qwen-max", response_format=None, temperature=0.3):
        """
        Generate a chat completion using Qwen models
        
        Args:
            messages: List of message objects
            model: Qwen model to use
            response_format: Format for the response (json or text)
            temperature: Temperature for generation
            
        Returns:
            Response object
        """
        # Map OpenAI-style message format to Qwen format
        qwen_messages = []
        for msg in messages:
            role = msg["role"]
            # Map 'system' role to 'system' for Qwen
            # Map 'user' role to 'user' for Qwen
            # Map 'assistant' role to 'assistant' for Qwen
            qwen_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        # Build request payload
        payload = {
            "model": model,
            "input": {
                "messages": qwen_messages
            },
            "parameters": {
                "temperature": temperature,
                "result_format": "message"
            }
        }
        
        # Add response format if specified
        if response_format and response_format.get("type") == "json_object":
            payload["parameters"]["response_format"] = {"type": "json_object"}
        
        # Make API request
        url = f"{self.base_url}/services/aigc/text-generation/generation"
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Create a response object that mimics OpenAI's response structure
        choices = [{
            "message": {
                "role": "assistant",
                "content": result["output"]["text"]
            },
            "index": 0
        }]
        
        return type('ChatCompletion', (), {
            'choices': choices,
            'usage': result.get("usage", {})
        })
    
    def get_embedding(self, text, model="text-embedding-v2"):
        """
        Get embedding for text using Qwen embedding models
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        url = f"{self.base_url}/services/embeddings/text-embedding/text-embedding"
        payload = {
            "model": model,
            "input": {
                "texts": [text]
            }
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        embeddings = result["output"]["embeddings"]
        
        # Create a response object that mimics OpenAI's response structure
        return type('EmbeddingResponse', (), {
            'data': [
                type('EmbeddingData', (), {
                    'embedding': embeddings[0]["embedding"],
                    'index': 0
                })
            ]
        })


class OpenAIClient(LLMClient):
    """Client for OpenAI models"""
    
    def __init__(self, api_key=None):
        """Initialize the OpenAI client"""
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def chat_completion(self, messages, model="gpt-4o-mini", response_format=None, temperature=0.3):
        """
        Generate a chat completion using OpenAI models
        
        Args:
            messages: List of message objects
            model: OpenAI model to use
            response_format: Format for the response (json or text)
            temperature: Temperature for generation
            
        Returns:
            Response object
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature
        )
    
    def get_embedding(self, text, model="text-embedding-ada-002"):
        """
        Get embedding for text using OpenAI embedding models
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        return self.client.embeddings.create(
            input=[text], 
            model=model
        )


def create_llm_client(provider="qwen"):
    """Create an LLM client based on the provider"""
    if provider.lower() == "qwen":
        return QwenClient()
    elif provider.lower() == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class MultiAgentQuerySystem:
    """Multi-agent system for answering questions with conversation history"""
    
    def __init__(self, model: str = "qwen-max", rag_model: str = None, web_search_enabled: bool = True, provider: str = "qwen"):
        """
        Initialize the multi-agent query system with conversation history
        
        Args:
            model: LLM model to use for the multi-agent system
            rag_model: LLM model to use for the RAG database queries (defaults to model if None)
            web_search_enabled: Whether to enable web search for questions not answered by the database
            provider: Model provider to use (qwen or openai)
        """
        self.provider = provider.lower()
        self.model = model
        self.rag_model = rag_model if rag_model else model
        
        # Use default models if provider is qwen
        if self.provider == "qwen":
            if "qwen" not in self.model.lower():
                print(f"Warning: Specified model {model} may not be compatible with Qwen provider")
            if self.rag_model and "qwen" not in self.rag_model.lower():
                print(f"Warning: Specified RAG model {rag_model} may not be compatible with Qwen provider")
        
        self.rag_engine = RAGQueryEngine(model_name=self.rag_model)
        self.client = create_llm_client(provider)
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())
        self.web_search_enabled = web_search_enabled
        
        print(f"Initializing multi-agent query system with {provider} model: {model}")
        if self.rag_model != model:
            print(f"RAG database using model: {self.rag_model}")
        print(f"Session ID: {self.session_id}")
        if web_search_enabled:
            print("Web search is enabled for questions not answered by your database")
    
    def search_web(self, query: str) -> List[Dict]:
        """Search the web for information using DuckDuckGo"""
        try:
            if not self.web_search_enabled:
                return []
                
            # Use DuckDuckGo search if available
            if DDGS is not None:
                print(f"Searching the web for: {query}")
                search_results = []
                
                # Create a DuckDuckGo search instance
                with DDGS() as ddgs:
                    # Get text search results (web pages)
                    results = list(ddgs.text(query, max_results=5))
                    
                    for result in results:
                        search_results.append({
                            "title": result.get("title", "Unknown"),
                            "snippet": result.get("body", "No description available"),
                            "url": result.get("href", ""),
                            "type": "web"
                        })
                    
                    # If fewer than 3 results, also try news search
                    if len(search_results) < 3:
                        news_results = list(ddgs.news(query, max_results=3))
                        for result in news_results:
                            if not any(r.get("url") == result.get("url") for r in search_results):
                                search_results.append({
                                    "title": result.get("title", "Unknown"),
                                    "snippet": result.get("body", "No description available"),
                                    "url": result.get("url", ""),
                                    "type": "web-news"
                                })
                
                if not search_results:
                    print("No web search results found")
                    return []
                
                print(f"Found {len(search_results)} web search results")
                return search_results
            else:
                print("DuckDuckGo search not available. Install with: pip install duckduckgo-search")
                return []
        except Exception as e:
            print(f"Web search error: {str(e)}")
            return []
    
    def evaluate_answer_sufficiency(self, question: str, rag_answer: str, rag_sources: List[Dict]) -> Dict:
        """
        Evaluate if the RAG answer is sufficient or if web search is needed
        
        Args:
            question: The original user question
            rag_answer: Answer from the RAG database
            rag_sources: Sources retrieved from the RAG database
            
        Returns:
            Dict with evaluation results
        """
        # Skip evaluation if web search is disabled
        if not self.web_search_enabled:
            return {
                "is_sufficient": True,
                "reason": "Web search is disabled"
            }
            
        # Check for obvious insufficiency indicators
        insufficiency_phrases = [
            "don't have enough information",
            "couldn't find",
            "insufficient information",
            "not enough information",
            "no information",
            "doesn't appear",
            "not mention",
            "information is limited",
            "cannot provide",
            "unable to find",
            "not explicitly mentioned",
            "no specific mention",
            "not clear from",
            "doesn't contain"
        ]
        
        # Quick check for obvious indicators
        if any(phrase in rag_answer.lower() for phrase in insufficiency_phrases):
            return {
                "is_sufficient": False,
                "reason": "RAG answer explicitly indicates insufficient information"
            }
            
        # If sources have low relevance scores
        relevant_sources = [s for s in rag_sources if s["relevance"] > 0.8]
        if not relevant_sources and rag_sources:
            return {
                "is_sufficient": False,
                "reason": "No highly relevant sources found in database"
            }
            
        # For more complex cases, use LLM to evaluate
        system_prompt = """You are an answer evaluator that determines if a response from a research database is sufficient.
        Analyze the question and answer to determine if the information provided is adequate and directly addresses the query.
        """
        
        user_prompt = f"""
        QUESTION: {question}
        
        ANSWER FROM DATABASE: {rag_answer}
        
        Indicate if this answer sufficiently addresses the question or if additional information should be sought.
        Return a JSON with:
        - "is_sufficient": boolean (true if answer is adequate, false if additional information is needed)
        - "reason": brief explanation of your decision
        """
        
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"Error in answer evaluation: {str(e)}")
            # Default to sufficient if evaluation fails
            return {
                "is_sufficient": True,
                "reason": f"Evaluation failed: {str(e)}"
            }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query with conversation history"""
        # Add the user's question to the conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        # First, query the RAG database
        print("Querying your research database...")
        rag_results = self.rag_engine.query(question)
        
        # Handle the tuple result from RAGQueryEngine (answer, retrieved_chunks)
        rag_answer, retrieved_chunks = rag_results
        
        # Format RAG results
        rag_sources = []
        for chunk in retrieved_chunks:
            rag_sources.append({
                "title": chunk.get("title", "Unknown paper"),
                "chunk_text": chunk.get("chunk_text", ""),
                "relevance": chunk.get("similarity", 0),
                "type": "database"
            })
        
        # Evaluate if RAG answer is sufficient
        evaluation = self.evaluate_answer_sufficiency(question, rag_answer, rag_sources)
        
        # Initialize web sources
        web_sources = []
        web_search_performed = False
        
        # If answer is insufficient and web search is enabled, search the web
        if not evaluation.get("is_sufficient", True) and self.web_search_enabled:
            print(f"Database answer insufficient: {evaluation.get('reason', 'Unknown reason')}")
            print("Searching the web for additional information...")
            web_sources = self.search_web(question)
            web_search_performed = True
        
        # Check if we have any sources (either from RAG or web)
        all_sources = rag_sources + web_sources
        if not all_sources:
            synthesized_answer = "I couldn't find any relevant information to answer your question."
            confidence = 0
        else:
            # Format conversation history for context
            conversation_context = ""
            if len(self.conversation_history) > 1:
                conversation_context = "Previous conversation:\n"
                for i, msg in enumerate(self.conversation_history[:-1]):
                    prefix = "User: " if msg["role"] == "user" else "Assistant: "
                    conversation_context += f"{prefix}{msg['content']}\n\n"
            
            # Format sources for the prompt
            sources_text = ""
            if rag_sources:
                sources_text += "Sources from user's research database:\n"
                for i, source in enumerate(rag_sources, 1):
                    sources_text += f"{i}. {source['title']}\n"
                    sources_text += f"   Relevance: {source['relevance']:.2f}\n"
                    sources_text += f"   Snippet: {source['chunk_text'][:300]}...\n\n"
            
            web_sources_text = ""
            if web_sources:
                web_sources_text = "Sources from web search:\n"
                for i, source in enumerate(web_sources, 1):
                    web_sources_text += f"{i}. {source['title']}\n"
                    web_sources_text += f"   URL: {source['url']}\n"
                    web_sources_text += f"   Snippet: {source['snippet'][:300]}...\n\n"
            
            # Create the system prompt
            system_prompt = f"""You are a research assistant helping answer questions about machine learning papers.
            
            CRITICAL INSTRUCTIONS:
            1. Prioritize information from the user's personal research database
            2. Maintain context from the conversation history
            3. Refer to papers from the user's database when relevant
            4. Be honest when information is not available
            5. Answers should be comprehensive but concise
            6. When web search was performed, indicate this in your answer
            
            Return a JSON with:
            - "answer": your response to the user's question
            - "confidence": level of confidence (0-10)
            - "sources_used": list of sources you referenced
            """
            
            # Create the user prompt
            user_prompt = f"""
            {conversation_context}
            
            Current question: {question}
            
            {sources_text}
            
            {"" if not web_sources_text else web_sources_text}
            
            RAG Database Answer: {rag_answer}
            
            Based on the available information and our conversation history, please answer the question.
            """
            
            # Query the LLM for the answer
            print("Synthesizing answer...")
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse the response
            content = response.choices[0].message.content
            result = json.loads(content)
            synthesized_answer = result.get("answer", "I couldn't provide an answer.")
            confidence = result.get("confidence", 0)
        
        # Add the assistant's response to the conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": synthesized_answer
        })
        
        # Format the final result
        return {
            "answer": synthesized_answer,
            "confidence": confidence,
            "sources": all_sources,
            "conversation_history": self.conversation_history.copy(),
            "rag_answer": rag_answer,
            "web_search_performed": web_search_performed,
            "evaluation": evaluation
        }

def format_answer(result, show_history=False):
    """Format the answer for display"""
    answer = result.get("answer", "")
    confidence = result.get("confidence", 0)
    sources = result.get("sources", [])
    web_search_performed = result.get("web_search_performed", False)
    
    output = f"\n{'='*80}\n"
    output += f" Confidence: {confidence}/10\n\n"
    output += f" Answer:\n{answer}\n\n"
    
    # Group sources by type
    db_sources = [s for s in sources if s.get("type", "") == "database"]
    web_sources = [s for s in sources if s.get("type", "") in ["web", "web-news"]]
    
    if db_sources:
        output += f" Sources from your research database:\n"
        for i, source in enumerate(db_sources, 1):
            title = source.get("title", "Unknown")
            relevance = source.get("relevance", 0)
            output += f"{i}. {title} (relevance: {relevance:.2f})\n"
        output += "\n"
    
    if web_sources:
        output += f" Web sources:\n"
        for i, source in enumerate(web_sources, 1):
            title = source.get("title", "Unknown")
            url = source.get("url", "")
            output += f"{i}. {title}\n   {url}\n"
        output += "\n"
    
    if web_search_performed:
        evaluation = result.get("evaluation", {})
        reason = evaluation.get("reason", "No specific reason provided")
        output += f" Web search was performed because: {reason}\n\n"
    
    if show_history:
        history = result.get("conversation_history", [])
        if len(history) > 2:  # Show history only if there's more than current Q&A
            output += f" Conversation History:\n"
            for i, msg in enumerate(history[:-2]):  # Skip current Q&A
                prefix = "You: " if msg["role"] == "user" else "Assistant: "
                # Truncate very long messages
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                output += f"{i+1}. {prefix}{content}\n"
            output += "\n"
    
    output += f"{'='*80}\n"
    return output

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent RAG Query Tool with Conversation History')
    parser.add_argument('--question', '-q', type=str, help='Question to ask')
    parser.add_argument('--model', '-m', type=str, default='qwen-max', 
                        help='LLM model to use for multi-agent system (default: qwen-max)')
    parser.add_argument('--rag-model', '-r', type=str, 
                        help='LLM model to use for RAG database queries (defaults to --model if not specified)')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='Run in interactive mode')
    parser.add_argument('--history', '-H', action='store_true',
                        help='Show conversation history in output')
    parser.add_argument('--no-web', action='store_true',
                        help='Disable web search fallback')
    parser.add_argument('--provider', '-p', type=str, default='qwen',
                        help='Model provider to use: qwen or openai (default: qwen)')
    
    args = parser.parse_args()
    
    # Check API keys based on provider
    if args.provider.lower() == 'qwen':
        if not os.environ.get('DASHSCOPE_API_KEY'):
            print("ERROR: DASHSCOPE_API_KEY environment variable is not set")
            print("Please set it with: export DASHSCOPE_API_KEY='your-api-key'")
            sys.exit(1)
    elif args.provider.lower() == 'openai':
        if not os.environ.get('OPENAI_API_KEY'):
            print("ERROR: OPENAI_API_KEY environment variable is not set")
            print("Please set it with: export OPENAI_API_KEY='your-api-key'")
            sys.exit(1)
    else:
        print(f"ERROR: Unsupported provider: {args.provider}")
        print("Supported providers: qwen, openai")
        sys.exit(1)
    
    try:
        # Initialize the multi-agent query system
        system = MultiAgentQuerySystem(
            model=args.model,
            rag_model=args.rag_model,
            web_search_enabled=not args.no_web,
            provider=args.provider
        )
        
        if args.interactive:
            # Run in interactive mode
            print("\n===== ArXiv Daily Multi-Agent RAG Query Tool =====")
            print("Ask questions to get answers from your paper database.")
            print("Conversation history is maintained between questions.")
            print("Type 'exit' or 'quit' to end the session.\n")
            
            while True:
                # Get user question
                question = input("Question: ")
                if question.lower() in ['exit', 'quit']:
                    break
                
                if not question.strip():
                    continue
                
                # Query the system
                print("\nProcessing your question...")
                result = system.query(question)
                
                # Print the answer
                print(format_answer(result, args.history))
                
        elif args.question:
            # Single question mode
            question = args.question
            print(f"Question: {question}")
            
            # Query the system
            print("Processing your question...")
            result = system.query(question)
            
            # Print the answer
            print(format_answer(result, args.history))
        
        else:
            print("Please provide a question with --question or use --interactive mode")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
