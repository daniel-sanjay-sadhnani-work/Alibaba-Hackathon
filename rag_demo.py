#!/usr/bin/env python3
"""
RAG Demo App - Compare model responses with and without RAG
"""

import os
import sqlite3
import gradio as gr
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from arxiv_daily.configs.config import Config

class RAGDemoApp:
    def __init__(self, db_path=Config.DB_PATH, model_name="gpt-4o-mini"):
        """Initialize demo app with database connection and OpenAI API"""
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("ERROR: OPENAI_API_KEY environment variable is not set")
            print("Please set it with: export OPENAI_API_KEY='your-api-key'")
            exit(1)
            
        self.db_path = db_path
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-ada-002"
        
        # Check if vector_chunks table exists
        self.rag_available = self._check_vector_table()
        if not self.rag_available:
            print("WARNING: vector_chunks table does not exist. RAG will not be available.")
            print("Will still demonstrate pure LLM capabilities")
    
    def _check_vector_table(self):
        """Check if vector_chunks table exists in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='vector_chunks'
            """)
            result = cursor.fetchone() is not None
            conn.close()
            return result
        except Exception as e:
            print(f"Error checking vector table: {str(e)}")
            return False
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text):
        """Get embedding for text using OpenAI's embedding API"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text], 
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_llm_response(self, query):
        """Get response from base LLM without RAG"""
        system_content = """You are a helpful research assistant specializing in machine learning, 
        computer vision, and AI research papers. Provide concise, accurate information 
        about research topics."""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve the most relevant chunks for the query from the database"""
        # Create a new connection for thread safety
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get embedding for the query
            query_embedding = self.get_embedding(query)
            
            # Get all chunks from the database
            cursor.execute("""
                SELECT c.id, c.paper_id, c.chunk_text, c.chunk_embedding, p.title 
                FROM vector_chunks c
                JOIN papers p ON c.paper_id = p.id
            """)
            chunks = cursor.fetchall()
            
            if not chunks:
                conn.close()
                return []
            
            # Calculate similarities and sort
            similarities = []
            for chunk in chunks:
                # Convert stored embedding from string to list
                chunk_embedding = np.array(eval(chunk['chunk_embedding']))
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    chunk_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append({
                    'chunk_id': chunk['id'],
                    'paper_id': chunk['paper_id'],
                    'title': chunk['title'],
                    'chunk_text': chunk['chunk_text'],
                    'similarity': similarity
                })
            
            # Sort by similarity (descending)
            sorted_chunks = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
            
            # Close connection before returning
            conn.close()
            
            # Return top_k results
            return sorted_chunks[:top_k]
            
        except Exception as e:
            conn.close()
            raise e
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_rag_answer(self, query, context_chunks):
        """Generate an answer to the query based on the retrieved context chunks"""
        # Prepare context from chunks
        context = "\n\n".join([
            f"Title: {chunk['title']}\nContent: {chunk['chunk_text']}"
            for chunk in context_chunks
        ])
        
        # Create prompt for LLM
        system_content = "You are a helpful research assistant that answers questions based only on the provided context."
        user_content = f"""
        Use ONLY the following context information to answer the question.
        If the necessary information is not in the context, say "I don't have enough information to answer this question."
        
        CONTEXT:
        {context}
        
        QUESTION: {query}
        """
        
        # Send to OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def compare_responses(self, query):
        """Compare responses with and without RAG"""
        # Get normal LLM response
        print(f"Getting LLM response for: {query}")
        llm_response = self.get_llm_response(query)
        
        # Get RAG response if available
        if self.rag_available:
            print(f"Getting RAG response for: {query}")
            try:
                # Retrieve chunks (creates new SQLite connection)
                chunks = self.retrieve_relevant_chunks(query)
                
                if not chunks:
                    rag_response = "No relevant information found in your paper database."
                    formatted_sources = "No papers found matching your query."
                else:
                    # Generate answer from chunks
                    rag_response = self.generate_rag_answer(query, chunks)
                    
                    # Format sources - deduplicate papers
                    formatted_sources = ""
                    unique_papers = {}
                    
                    for chunk in chunks:
                        paper_id = chunk['paper_id']
                        if paper_id not in unique_papers:
                            unique_papers[paper_id] = {
                                'title': chunk['title'],
                                'similarity': chunk['similarity']
                            }
                    
                    for i, (paper_id, info) in enumerate(unique_papers.items(), 1):
                        formatted_sources += f"{i}. {info['title']} (relevance: {info['similarity']:.2f})\n"
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                rag_response = f"Error generating RAG response: {str(e)}"
                formatted_sources = "No sources available due to error"
        else:
            rag_response = "RAG system is not available. Please initialize the vector database first with: python main.py --process-for-rag"
            formatted_sources = "No sources available - vector database not initialized"
        
        return llm_response, rag_response, formatted_sources

def create_interface():
    """Create Gradio interface"""
    # Create demo app
    app = RAGDemoApp()
    
    # Define Gradio interface
    with gr.Blocks(title="ArXiv Daily RAG Demo") as interface:
        gr.Markdown("""
        # ArXiv Daily RAG Demo
        
        This demo shows the difference between a language model with and without Retrieval-Augmented Generation (RAG).
        
        - **Standard LLM**: Uses only the model's pre-trained knowledge
        - **LLM + RAG**: Enhances responses using your personal ArXiv paper database
        
        Try asking questions about research papers in your database!
        """)
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are recent advancements in dataset distillation?",
                lines=2
            )
            submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Standard LLM Response")
                llm_output = gr.Textbox(label="", lines=12)
            
            with gr.Column():
                gr.Markdown("### LLM + RAG Response")
                rag_output = gr.Textbox(label="", lines=12)
        
        gr.Markdown("### Sources from Your ArXiv Database")
        sources_output = gr.Textbox(label="", lines=5)
        
        # Set up event handler
        submit_btn.click(
            fn=app.compare_responses,
            inputs=query_input,
            outputs=[llm_output, rag_output, sources_output]
        )
        
        gr.Markdown("""
        ## How It Works
        
        1. Your question is processed by both systems
        2. The standard LLM uses only its pre-trained knowledge
        3. The RAG system:
           - Converts your question to an embedding vector
           - Searches your paper database for relevant chunks
           - Provides these chunks as context to the LLM
           - Generates an answer grounded in your specific papers
        
        This demonstrates how RAG can provide more accurate, specific answers about papers in your database.
        """)
    
    return interface

if __name__ == "__main__":
    # Check for gradio
    try:
        import gradio as gr
    except ImportError:
        print("Please install gradio with: pip install gradio")
        exit(1)
    
    # Launch Gradio interface
    print("Starting ArXiv Daily RAG Demo...")
    interface = create_interface()
    interface.launch(share=False)
