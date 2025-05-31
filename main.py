#!/usr/bin/env python3
"""
ArXiv Daily - Customized ArXiv Paper Feeds with PDF Processing and RAG Database

Main entry point for the application.
"""

import argparse
import schedule
import time
from datetime import datetime

from arxiv_daily.configs.config import Config
from arxiv_daily.api.arxiv_api import ArxivAPI
from arxiv_daily.processing.paper_processor import PaperProcessor
from arxiv_daily.processing.pdf_manager import PDFManager
from arxiv_daily.processing.llm_processor import LLMProcessor
from arxiv_daily.storage.database_manager import DatabaseManager
from arxiv_daily.utils.email_sender import EmailSender
from arxiv_daily.agents.filter_system import AgentFilterSystem, create_research_profile

def daily_job(specific_date=None):
    """
    Execute daily paper fetching, processing, and database management task.
    
    Args:
        specific_date: Optional specific date to query papers for
    """
    print("===== START: Daily ArXiv Paper Fetching Job =====")
    print(f"DEBUG: Running job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize database
    DatabaseManager.initialize_db()
    
    days_back = Config.DAYS_TO_LOOK_BACK
    time_period = "today" if days_back == 1 else f"the last {days_back} days"
    
    if specific_date:
        print(f"INFO: Fetching papers for specific date: {specific_date.strftime('%Y-%m-%d')}")
        time_period = specific_date.strftime('%Y-%m-%d')
    
    try:
        # Fetch papers
        if specific_date:
            papers = ArxivAPI.fetch_papers_by_date(specific_date)
        else:
            papers = ArxivAPI.fetch_papers()
        
        # Check if there are no papers for the specified period
        if not papers:
            print(f"INFO: No papers found for {time_period} (may be weekend or holiday in arXiv schedule)")
            content = PaperProcessor.generate_email_content([], found_papers=False, no_papers_today=True, time_period=time_period)
            EmailSender.send_email(content)
            return
        
        # Filter papers based on keywords or using multi-agent system
        if use_agent_filter:
            # Create research profile based on configured keywords
            profile = create_research_profile()
            
            # Create agent filter system
            filter_system = AgentFilterSystem(
                research_profile=profile,
                model=args.agent_model
            )
            
            # Use multi-agent system to filter papers
            filtered_papers = filter_system.filter_papers(papers)
        else:
            # Traditional keyword-based filtering
            filtered_papers = PaperProcessor.filter_papers(papers)
        
        # Store all matched papers in database
        stored_papers = []
        for paper in filtered_papers:
            if DatabaseManager.store_paper(paper):
                stored_papers.append(paper)
        
        print(f"INFO: Stored {len(stored_papers)} papers in the database")
        
        # Process PDFs and generate summaries
        if Config.DOWNLOAD_PDFS and Config.OPENAI_API_KEY:
            print("INFO: Beginning PDF download and summarization process")
            for paper in stored_papers:
                paper_id = paper.get('id')
                if not paper_id:
                    continue
                
                # Download PDF
                pdf_path = PDFManager.download_pdf(paper)
                if pdf_path:
                    # Update database with PDF path
                    DatabaseManager.update_paper_processed(paper_id, pdf_path)
                    
                    # Extract text from PDF
                    pdf_text = PDFManager.extract_text(pdf_path)
                    if pdf_text:
                        # Store text chunks for RAG
                        DatabaseManager.store_text_chunks(paper_id, pdf_text)
                        
                        # Generate summary with LLM if enabled
                        if use_llm_summary:
                            print(f"INFO: Generating LLM summary for paper {paper_id} using model {Config.OPENAI_MODEL}")
                            summary_result = LLMProcessor.summarize_text(pdf_text, paper)
                            if "summary" in summary_result and not "error" in summary_result:
                                # Store summary
                                DatabaseManager.store_summary(paper_id, summary_result["summary"])
                                paper["llm_summary"] = summary_result["summary"]
                else:
                    # Mark as processed even if download failed
                    DatabaseManager.update_paper_processed(paper_id)
        
        # Generate email
        if filtered_papers:
            print(f"INFO: Found {len(filtered_papers)} relevant papers")
            content = PaperProcessor.generate_email_content(filtered_papers, time_period=time_period)
        else:
            print(f"INFO: No relevant papers found matching keywords for {time_period}")
            content = PaperProcessor.generate_email_content([], found_papers=False, time_period=time_period)
        
        # Always send an email, regardless of whether papers were found
        EmailSender.send_email(content)
        
    except Exception as e:
        print(f"ERROR: An exception occurred during job execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("===== END: Daily ArXiv Paper Fetching Job =====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArXiv Papers Fetcher with PDF Processing and RAG Database')
    parser.add_argument('--date', type=str, help='Fetch papers for a specific date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=Config.DAYS_TO_LOOK_BACK, 
                        help=f'Number of days to look back (default: {Config.DAYS_TO_LOOK_BACK})')
    parser.add_argument('--schedule', action='store_true', 
                       help='Schedule the job to run daily at 9am Singapore time')
    parser.add_argument('--llm', type=str, 
                       help='Enable LLM summarization using specified model (e.g., gpt-4o-mini)')
    parser.add_argument('--process-for-rag', action='store_true',
                       help='Process existing papers for RAG database (create chunks and embeddings)')
    parser.add_argument('--agent-filter', action='store_true',
                       help='Use multi-agent system for semantic paper filtering instead of keyword matching')
    parser.add_argument('--agent-model', type=str, default='gpt-4o-mini',
                        help='LLM model to use for agent-based filtering (default: gpt-4o-mini)')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF downloads and processing')
    parser.add_argument('--no-email', action='store_true', help='Skip sending email')
    parser.add_argument('--keyword', type=str, help='Query database for papers matching a specific keyword')
    args = parser.parse_args()
    
    # Override config based on arguments
    if args.no_pdf:
        Config.DOWNLOAD_PDFS = False
        
    # Set LLM model if specified
    use_llm_summary = False
    if args.llm:
        if Config.OPENAI_API_KEY:
            use_llm_summary = True
            Config.OPENAI_MODEL = args.llm
            print(f"INFO: LLM summarization enabled using model: {args.llm}")
        else:
            print("WARNING: LLM summarization requested but OPENAI_API_KEY not found in environment")
    
    # Use agent-based filtering if specified
    use_agent_filter = False
    if args.agent_filter:
        if Config.OPENAI_API_KEY:
            use_agent_filter = True
            print(f"INFO: Multi-agent semantic filtering enabled using model: {args.agent_model}")
        else:
            print("WARNING: Agent filtering requested but OPENAI_API_KEY not found in environment")
        
    if args.process_for_rag:
        print("INFO: Processing papers for RAG database")
        from arxiv_daily.storage.vector_store import VectorStore
        vector_store = VectorStore(Config.DB_PATH)
        processed_count = vector_store.process_all_unprocessed_papers()
        print(f"INFO: Processed {processed_count} papers for RAG database")
        import sys
        sys.exit(0)
        
    if args.keyword:
        # Query for papers by keyword
        DatabaseManager.initialize_db()
        papers = DatabaseManager.get_papers_by_keyword(args.keyword)
        if papers:
            print(f"Found {len(papers)} papers matching keyword '{args.keyword}':")
            for i, paper in enumerate(papers):
                print(f"{i+1}. {paper['title']} (ID: {paper['id']})")
        else:
            print(f"No papers found matching keyword '{args.keyword}'")
    elif args.date:
        # Run job for specific date
        try:
            specific_date = datetime.strptime(args.date, '%Y-%m-%d')
            daily_job(specific_date)
        except ValueError:
            print("ERROR: Invalid date format. Please use YYYY-MM-DD format.")
    elif args.schedule:
        # Schedule daily job
        schedule_time = "08:00"  # Default to 8:00 AM Singapore time (arXiv update time)
        print(f"Scheduling daily job to run at {schedule_time}")
        schedule.every().day.at(schedule_time).do(daily_job)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nScheduled job terminated by user")
    else:
        # Run job immediately with default settings
        daily_job()
