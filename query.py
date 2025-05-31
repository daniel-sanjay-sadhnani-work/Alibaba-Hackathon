#!/usr/bin/env python3
"""
RAG Query Tool for ArXiv Daily
This script allows you to ask questions about papers in your database
and get answers based on the content using RAG (Retrieval-Augmented Generation).
"""

import os
import argparse
import sys
from pprint import pprint

from arxiv_daily.rag.query_engine import RAGQueryEngine
from arxiv_daily.configs.config import Config

def main():
    parser = argparse.ArgumentParser(description='Query papers in your ArXiv database using RAG')
    parser.add_argument('--question', '-q', type=str, help='Question to ask about papers in your database')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini', 
                        help='LLM model to use for answering (default: gpt-4o-mini)')
    parser.add_argument('--top-k', '-k', type=int, default=5, 
                        help='Number of top chunks to retrieve (default: 5)')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    try:
        # Initialize the RAG query engine
        print(f"Initializing RAG query engine with model: {args.model}")
        engine = RAGQueryEngine(db_path=Config.DB_PATH, model_name=args.model)
        
        if args.interactive:
            # Run in interactive mode
            print("\n===== ArXiv Daily RAG Query Tool =====")
            print("Ask questions about papers in your database.")
            print("Type 'exit' or 'quit' to end the session.\n")
            
            while True:
                # Get user question
                question = input("\033[1mQuestion:\033[0m ")
                if question.lower() in ['exit', 'quit']:
                    break
                
                if not question.strip():
                    continue
                
                # Query the engine
                print("\nSearching database and generating answer...")
                answer, chunks = engine.query(question, args.top_k)
                
                # Print the answer
                print("\n\033[1mAnswer:\033[0m")
                print(f"{answer}\n")
                
                # Show source information
                print("\033[1mSources:\033[0m")
                for i, chunk in enumerate(chunks, 1):
                    print(f"{i}. {chunk['title']} (similarity: {chunk['similarity']:.4f})")
                
                print("\n" + "-" * 50)
        
        elif args.question:
            # Single question mode
            question = args.question
            print(f"\nQuestion: {question}")
            
            # Query the engine
            print("Searching database and generating answer...")
            answer, chunks = engine.query(question, args.top_k)
            
            # Print the answer
            print("\nAnswer:")
            print(f"{answer}\n")
            
            # Show source information
            print("Sources:")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk['title']} (similarity: {chunk['similarity']:.4f})")
        
        else:
            print("Please provide a question with --question or use --interactive mode")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
