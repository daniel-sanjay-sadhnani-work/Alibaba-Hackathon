import smtplib
from email.mime.text import MIMEText
from email.header import Header
import schedule
import time
import os
import re
import json
import sqlite3
import requests
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse
import feedparser
import PyPDF2
from typing import List, Dict, Optional, Tuple, Any

# For LLM integration (using OpenAI as example)
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configuration
class Config:
    EMAIL = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'xiaolingao987@gmail.com',
        'receiver_email': 'xiaolingao987@gmail.com',
        'sender_password': os.getenv('GMAIL_APP_PWD')
    }

    ARXIV_CATEGORIES = {
        'cs.LG': 'Machine Learning',
        'cs.AI': 'Artificial Intelligence',
        'cs.CV': 'Computer Vision',
        'stat.ML': 'Machine Learning (Statistics)'
    }

    KEYWORDS = [
        'dataset distillation', 'dataset compression',
        'dataset pruning', 'dataset condensation'
    ]
    
    # Number of days to look back for papers (user-configurable)
    DAYS_TO_LOOK_BACK = 5
    
    # Explanatory note about arXiv's schedule
    ARXIV_SCHEDULE_INFO = """
        <p><em>Note: ArXiv operates on US Eastern Time and new papers are 
        announced at 8:00 PM ET (8:00 AM Singapore time) from Sunday to Thursday.</em></p>
    """

    # Path settings
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR = BASE_DIR / "pdfs"
    DB_PATH = BASE_DIR / "arxiv_papers.db"
    
    # LLM settings
    LLM_PROVIDER = "openai"  # Options: "openai", "local", etc.
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = "gpt-4o"  # Or another model
    
    # PDF processing
    DOWNLOAD_PDFS = True
    MAX_PDF_SIZE_MB = 20  # Skip PDFs larger than this
    
    # Summary settings
    SUMMARY_PROMPT = """
    You are a helpful research assistant summarizing an academic paper.
    Please provide a concise summary of the paper with the following structure:
    
    1. Key Contributions (3-5 bullet points)
    2. Main Methodology (2-3 sentences)
    3. Results (2-3 sentences)
    4. Potential Applications (1-2 sentences)
    
    Be factual, specific, and focus on the novel aspects of this work.
    """
    
    # RAG Database settings
    VECTOR_CHUNK_SIZE = 1000
    VECTOR_CHUNK_OVERLAP = 200

class ArxivAPI:
    @staticmethod
    def construct_query(days_back: int = Config.DAYS_TO_LOOK_BACK, batch_size: int = 100, start_index: int = 0) -> str:
        """Construct arXiv API query URL with date filtering and timezone adjustment."""
        print(f"DEBUG: Constructing query with batch_size={batch_size}, start_index={start_index}, days_back={days_back}")
        base_url = 'http://export.arxiv.org/api/query?'
        
        # Get arXiv's "today" by adjusting for timezone difference between Singapore and US Eastern
        now_utc = datetime.utcnow()
        # Approximate EST/EDT (UTC-5 or UTC-4)
        # Singapore is UTC+8, so roughly 12-13 hours ahead
        arxiv_now = now_utc - timedelta(hours=12)  
        
        # Calculate the date range in arXiv's timezone
        end_date = arxiv_now
        start_date = end_date - timedelta(days=days_back-1)  # -1 because we want to include today
        
        # Format dates for arXiv API (YYYYMMDD format)
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        if days_back == 1:
            # For single day, use simpler query
            date_query = f"submittedDate:{end_date_str}"
            print(f"DEBUG: Looking for papers from arXiv date: {end_date_str}")
        else:
            # For date range, use between syntax
            date_query = f"submittedDate:[{start_date_str}000000 TO {end_date_str}235959]"
            print(f"DEBUG: Looking for papers from arXiv dates: {start_date_str} to {end_date_str}")
        
        search_query = (
            f'({date_query}) AND '
            '(cat:cs.* OR cat:math.* OR cat:stat.* OR '
            'cat:eess.* OR cat:physics.*)'
        )
        
        query_params = {
            'search_query': search_query,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
            'max_results': batch_size,
            'start': start_index
        }
        
        return base_url + urllib.parse.urlencode(query_params)

    @staticmethod
    def fetch_papers(max_results: int = 2000) -> List[Dict]:
        """Fetch papers from arXiv API for the configured time period."""
        days_back = Config.DAYS_TO_LOOK_BACK
        time_period = "today" if days_back == 1 else f"the last {days_back} days"
        
        print(f"DEBUG: Starting to fetch papers from {time_period} (max_results={max_results})")
        test_feed = feedparser.parse(ArxivAPI.construct_query(batch_size=1))
        if hasattr(test_feed, 'bozo_exception'):
            print(f"API Error: {test_feed.bozo_exception}")
            return []

        total_results = int(test_feed.feed.get('opensearch_totalresults', 0))
        print(f"DEBUG: Found {total_results} papers from {time_period}")
        
        # If no papers found, return empty list
        if total_results == 0:
            print(f"DEBUG: No papers found for {time_period} (may be weekend or holiday in arXiv schedule)")
            return []
            
        results_to_fetch = min(total_results, max_results)
        batch_size = min(results_to_fetch, 100)
        
        # Ensure batch_size is at least 1 to avoid range() error
        batch_size = max(batch_size, 1)
        
        all_papers = []
        for start_index in range(0, results_to_fetch, batch_size):
            current_batch = min(batch_size, results_to_fetch - start_index)
            print(f"DEBUG: Fetching batch of {current_batch} papers starting at index {start_index}")
            feed = feedparser.parse(ArxivAPI.construct_query(days_back=days_back, batch_size=current_batch, start_index=start_index))
            time.sleep(3)  # Respect rate limits
            
            for entry in feed.entries:
                paper = ArxivAPI._process_entry(entry)
                if paper:
                    all_papers.append(paper)
        
        print(f"DEBUG: Successfully fetched {len(all_papers)} papers")
        return all_papers

    @staticmethod
    def _process_entry(entry) -> Optional[Dict]:
        """Process a single arXiv entry."""
        try:
            return {
                'category': ', '.join(t.term for t in entry.tags),
                'title': entry.title.replace('\n', ' ').strip(),
                'authors': ', '.join(author.name for author in entry.authors),
                'abstract': entry.summary.replace('\n', ' ').strip(),
                'pdf_link': f"{entry.id.replace('abs', 'pdf')}.pdf",
                'abs_link': entry.id,
                'published': entry.published
            }
        except Exception as e:
            print(f"Error processing entry: {str(e)}")
            return None

class PaperProcessor:
    @staticmethod
    def filter_papers(papers: List[Dict]) -> List[Dict]:
        """Filter papers based on keywords."""
        print(f"DEBUG: Filtering {len(papers)} papers based on keywords")
        filtered = []
        for paper in papers:
            content = f"{paper['title']} {paper['abstract']}".lower()
            matched_keywords = []
            
            # Find all matching keywords for this paper
            for keyword in Config.KEYWORDS:
                if keyword.lower() in content:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                paper['matched_keywords'] = matched_keywords
                filtered.append(paper)
                print(f"DEBUG: Paper '{paper['title'][:50]}...' matched keywords: {matched_keywords}")
                
        print(f"DEBUG: Found {len(filtered)} papers matching keywords")
        return filtered

    @staticmethod
    def generate_email_content(papers: List[Dict], found_papers: bool = True, no_papers_today: bool = False) -> str:
        """Generate HTML email content with improved layout."""
        print("DEBUG: Generating email content")
        date_str = datetime.now().strftime("%Y-%m-%d")
        days_back = Config.DAYS_TO_LOOK_BACK
        time_period = "today" if days_back == 1 else f"the last {days_back} days"
        
        # Base CSS styles
        css = """
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }
            .header { background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
            .header h1 { margin: 0; color: #2c5282; font-size: 20px; }
            .header p { margin: 5px 0; }
            .info-box { background-color: #fffaf0; padding: 10px; border-left: 4px solid #ed8936; margin-bottom: 15px; font-size: 13px; }
            .paper-card { border: 1px solid #e2e8f0; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
            .paper-title { font-size: 16px; font-weight: bold; margin-top: 0; color: #2d3748; }
            .paper-meta { font-size: 13px; color: #4a5568; }
            .meta-group { margin-bottom: 5px; display: flex; }
            .meta-label { font-weight: bold; min-width: 100px; }
            .keywords-highlight { background-color: #FFFF00; padding: 2px 4px; border-radius: 3px; }
            .abstract { border-left: 3px solid #e2e8f0; padding-left: 10px; margin: 10px 0; font-size: 14px; line-height: 1.5; }
            .paper-links { font-size: 13px; }
            .paper-links a { color: #4299e1; text-decoration: none; margin-right: 15px; }
            .paper-links a:hover { text-decoration: underline; }
            .footer { margin-top: 20px; font-style: italic; text-align: center; color: #718096; }
        </style>
        """
        
        if no_papers_today:
            return f"""
                {css}
                <div class="header">
                    <h1>ArXiv Papers - {date_str}</h1>
                    <p><strong>No papers were published {time_period}</strong> in the categories you're monitoring.</p>
                </div>
                <div class="info-box">
                    <p>This is normal on weekends and holidays when arXiv typically doesn't post new papers.</p>
                    <p>{Config.ARXIV_SCHEDULE_INFO.strip()}</p>
                    <p><strong>Keywords searched:</strong> {', '.join(Config.KEYWORDS)}</p>
                </div>
                <div class="footer">
                    <p>Have a <strong>great day</strong>!</p>
                </div>
            """
        
        if not found_papers:
            return f"""
                {css}
                <div class="header">
                    <h1>ArXiv Papers - {date_str}</h1>
                    <p>Papers were published {time_period}, but <strong>none matched your keywords</strong>.</p>
                </div>
                <div class="info-box">
                    <p>{Config.ARXIV_SCHEDULE_INFO.strip()}</p>
                    <p><strong>Keywords searched:</strong> {', '.join(Config.KEYWORDS)}</p>
                </div>
                <div class="footer">
                    <p>Have a <strong>great day</strong>!</p>
                </div>
            """
        
        # Build paper cards
        paper_cards = []
        for i, paper in enumerate(papers, 1):
            abstract = paper['abstract'][:300] + '...' if len(paper['abstract']) > 300 else paper['abstract']
            
            # Format matched keywords with highlighting
            keywords_html = ''
            if 'matched_keywords' in paper and paper['matched_keywords']:
                keywords_list = []
                for keyword in paper['matched_keywords']:
                    keywords_list.append(f'<span class="keywords-highlight">{keyword}</span>')
                keywords_html = ', '.join(keywords_list)
            
            paper_cards.append(f"""
                <div class="paper-card">
                    <h3 class="paper-title">{i}. {paper['title']}</h3>
                    
                    <div class="paper-meta">
                        <div class="meta-group">
                            <span class="meta-label">Authors:</span>
                            <span>{paper['authors']}</span>
                        </div>
                        
                        <div class="meta-group">
                            <span class="meta-label">Categories:</span>
                            <span>{paper['category']}</span>
                        </div>
                        
                        <div class="meta-group">
                            <span class="meta-label">Published:</span>
                            <span>{paper['published'].split('T')[0]}</span>
                        </div>
                        
                        <div class="meta-group">
                            <span class="meta-label">Matched Keywords:</span>
                            <span>{keywords_html}</span>
                        </div>
                    </div>
                    
                    <div class="abstract">
                        {abstract}
                    </div>
                    
                    <div class="paper-links">
                        <a href="{paper['pdf_link']}" target="_blank">PDF</a>
                        <a href="{paper['abs_link']}" target="_blank">Abstract Page</a>
                    </div>
                </div>
            """)
        
        # Assemble the complete email
        return f"""
            {css}
            <div class="header">
                <h1>ArXiv Papers - {date_str}</h1>
                <p>Found <strong>{len(papers)}</strong> papers from {time_period} matching your keywords.</p>
            </div>
            
            <div class="info-box">
                <p><strong>Keywords searched:</strong> {', '.join(Config.KEYWORDS)}</p>
                <p>{Config.ARXIV_SCHEDULE_INFO.strip()}</p>
            </div>
            
            {''.join(paper_cards)}
            
            <div class="footer">
                <p>Have a <strong>great day</strong>!</p>
            </div>
        """

class EmailSender:
    @staticmethod
    def send_email(content: str) -> bool:
        """Send email with paper content."""
        print("DEBUG: Preparing to send email")
        date_str = datetime.now().strftime("%Y%m%d")
        msg = MIMEText(content, 'html', 'utf-8')
        msg['Subject'] = Header(f"arXiv论文推送_{date_str}", 'utf-8')
        msg['From'] = Config.EMAIL['sender_email']
        msg['To'] = Config.EMAIL['receiver_email']

        try:
            print(f"DEBUG: Connecting to SMTP server {Config.EMAIL['smtp_server']}:{Config.EMAIL['smtp_port']}")
            with smtplib.SMTP(Config.EMAIL['smtp_server'], Config.EMAIL['smtp_port']) as server:
                server.starttls()
                print("DEBUG: Logging into email account")
                server.login(Config.EMAIL['sender_email'], Config.EMAIL['sender_password'])
                print("DEBUG: Sending email")
                server.sendmail(
                    Config.EMAIL['sender_email'],
                    Config.EMAIL['receiver_email'],
                    msg.as_string()
                )
            print("DEBUG: Email sent successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to send email: {str(e)}")
            return False

class PDFManager:
    """Handles downloading and processing of PDFs from arXiv"""
    
    @staticmethod
    def ensure_pdf_dir():
        """Ensure PDF storage directory exists"""
        Config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_pdf_path(paper_id: str) -> Path:
        """Generate filesystem path for paper PDF"""
        return Config.PDF_DIR / f"{paper_id}.pdf"
    
    @staticmethod
    def download_pdf(paper: Dict) -> Optional[Path]:
        """Download PDF for a paper and return path if successful"""
        pdf_url = paper.get('pdf_link')
        paper_id = paper.get('id')
        
        if not pdf_url or not paper_id:
            print(f"ERROR: Missing PDF URL or ID for paper {paper.get('title', 'Unknown')}")
            return None
            
        pdf_path = PDFManager.get_pdf_path(paper_id)
        
        # Skip if already downloaded
        if pdf_path.exists():
            print(f"INFO: PDF already exists for {paper_id}")
            return pdf_path
            
        try:
            print(f"INFO: Downloading PDF for {paper_id} from {pdf_url}")
            PDFManager.ensure_pdf_dir()
            
            # Stream download to check size before saving entire file
            with requests.get(pdf_url, stream=True) as r:
                r.raise_for_status()
                
                # Check content length if available
                content_length = r.headers.get('Content-Length')
                if content_length and int(content_length) > Config.MAX_PDF_SIZE_MB * 1024 * 1024:
                    print(f"WARN: PDF too large ({int(content_length) // (1024*1024)}MB), skipping")
                    return None
                
                # Download file
                with open(pdf_path, 'wb') as f:
                    size = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        size += len(chunk)
                        # Check size during download
                        if size > Config.MAX_PDF_SIZE_MB * 1024 * 1024:
                            print(f"WARN: PDF download exceeded size limit, aborting")
                            f.close()
                            pdf_path.unlink(missing_ok=True)
                            return None
                        f.write(chunk)
            
            print(f"INFO: Successfully downloaded PDF to {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"ERROR: Failed to download PDF for {paper_id}: {str(e)}")
            # Clean up partial download if it exists
            pdf_path.unlink(missing_ok=True)
            return None
    
    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract text content from PDF file"""
        if not pdf_path.exists():
            print(f"ERROR: PDF file does not exist: {pdf_path}")
            return ""
            
        try:
            text = ""
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            return text
            
        except Exception as e:
            print(f"ERROR: Failed to extract text from PDF {pdf_path}: {str(e)}")
            return ""

class LLMProcessor:
    """Handles interactions with LLM for summarizing papers"""
    
    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def summarize_text(text: str, metadata: Dict) -> Dict:
        """
        Summarize paper text using LLM
        
        Args:
            text: The extracted text from the PDF
            metadata: Paper metadata (title, authors, etc)
            
        Returns:
            Dict with summary content
        """
        # Early return if no text or API key
        if not text or not Config.OPENAI_API_KEY:
            return {"error": "Missing text or API key"}
        
        # Set OpenAI API key
        openai.api_key = Config.OPENAI_API_KEY
        
        # Truncate text if too long (OpenAI has token limits)
        # Very roughly estimating 4 chars per token
        max_chars = 32000  # ~8k tokens for context
        if len(text) > max_chars:
            print(f"WARN: Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars]
        
        # Construct prompt with paper metadata
        system_prompt = Config.SUMMARY_PROMPT
        user_prompt = f"""
        Title: {metadata.get('title', 'Unknown')}
        Authors: {metadata.get('authors', 'Unknown')}
        Categories: {metadata.get('categories', 'Unknown')}
        
        Abstract:
        {metadata.get('summary', 'No abstract available')}
        
        Paper Content:
        {text}
        """
        
        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual output
                max_tokens=1000   # Limit response length
            )
            
            summary = response.choices[0].message.content
            return {
                "summary": summary,
                "paper_id": metadata.get('id'),
                "title": metadata.get('title'),
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"ERROR: LLM API call failed: {str(e)}")
            return {"error": str(e), "paper_id": metadata.get('id')}

class DatabaseManager:
    """Manages the SQLite database for storing papers and their summaries"""
    
    @staticmethod
    def initialize_db():
        """Create the database tables if they don't exist"""
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Papers table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT,
                    categories TEXT,
                    published TEXT,
                    abstract TEXT,
                    pdf_url TEXT,
                    web_url TEXT,
                    download_path TEXT,
                    processed BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                ''')
                
                # Summaries table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
                ''')
                
                # Keywords table for tracking which papers match which keywords
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS keyword_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    match_location TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
                ''')
                
                # Text chunks for RAG
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
                ''')
                
                conn.commit()
                print("INFO: Database initialized successfully")
                
        except Exception as e:
            print(f"ERROR: Failed to initialize database: {str(e)}")
    
    @staticmethod
    def store_paper(paper: Dict) -> bool:
        """Store paper metadata in database"""
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Check if paper already exists
                cursor.execute("SELECT id FROM papers WHERE id = ?", (paper.get('id'),))
                if cursor.fetchone():
                    print(f"INFO: Paper {paper.get('id')} already in database")
                    return True
                
                # Insert paper
                cursor.execute('''
                INSERT INTO papers (
                    id, title, authors, categories, published, abstract, 
                    pdf_url, web_url, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    paper.get('id', ''),
                    paper.get('title', ''),
                    json.dumps(paper.get('authors', [])),
                    json.dumps(paper.get('categories', [])),
                    paper.get('published', ''),
                    paper.get('summary', ''),
                    paper.get('pdf_link', ''),
                    paper.get('abs_link', ''),
                    datetime.now().isoformat()
                ))
                
                # Store keyword matches
                for keyword in Config.KEYWORDS:
                    locations = []
                    
                    # Check title
                    if keyword.lower() in paper.get('title', '').lower():
                        locations.append('title')
                        
                    # Check abstract
                    if keyword.lower() in paper.get('summary', '').lower():
                        locations.append('abstract')
                    
                    # If matches found, store them
                    for location in locations:
                        cursor.execute('''
                        INSERT INTO keyword_matches (paper_id, keyword, match_location, created_at)
                        VALUES (?, ?, ?, ?)
                        ''', (
                            paper.get('id', ''),
                            keyword,
                            location,
                            datetime.now().isoformat()
                        ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"ERROR: Failed to store paper in database: {str(e)}")
            return False
    
    @staticmethod
    def update_paper_processed(paper_id: str, pdf_path: Optional[Path] = None) -> bool:
        """Mark paper as processed and update download path"""
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                
                if pdf_path:
                    cursor.execute(
                        "UPDATE papers SET processed = 1, download_path = ? WHERE id = ?",
                        (str(pdf_path), paper_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE papers SET processed = 1 WHERE id = ?",
                        (paper_id,)
                    )
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"ERROR: Failed to update paper status: {str(e)}")
            return False
    
    @staticmethod
    def store_summary(paper_id: str, summary: str) -> bool:
        """Store paper summary in database"""
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO summaries (paper_id, summary_text, created_at)
                VALUES (?, ?, ?)
                ''', (
                    paper_id,
                    summary,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"ERROR: Failed to store summary: {str(e)}")
            return False
    
    @staticmethod
    def store_text_chunks(paper_id: str, text: str) -> bool:
        """Split text into chunks and store for RAG"""
        # Skip if empty text
        if not text:
            return False
            
        try:
            # Split text into chunks with overlap
            chunks = []
            text_length = len(text)
            chunk_size = Config.VECTOR_CHUNK_SIZE
            overlap = Config.VECTOR_CHUNK_OVERLAP
            
            for i in range(0, text_length, chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) < 50:  # Skip very small chunks
                    continue
                chunks.append((paper_id, chunk_text, len(chunks), datetime.now().isoformat()))
            
            # Store chunks in database
            with sqlite3.connect(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                
                cursor.executemany('''
                INSERT INTO text_chunks (paper_id, chunk_text, chunk_index, created_at)
                VALUES (?, ?, ?, ?)
                ''', chunks)
                
                conn.commit()
                print(f"INFO: Stored {len(chunks)} text chunks for paper {paper_id}")
                return True
                
        except Exception as e:
            print(f"ERROR: Failed to store text chunks: {str(e)}")
            return False
    
    @staticmethod
    def get_papers_by_keyword(keyword: str) -> List[Dict]:
        """Retrieve papers matching a specific keyword"""
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT p.* FROM papers p
                JOIN keyword_matches km ON p.id = km.paper_id
                WHERE km.keyword = ?
                ORDER BY p.published DESC
                ''', (keyword,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"ERROR: Failed to retrieve papers by keyword: {str(e)}")
            return []

def daily_job(specific_date: Optional[datetime] = None):
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
            content = PaperProcessor.generate_email_content([], found_papers=False, no_papers_today=True)
            EmailSender.send_email(content)
            return
        
        # Filter papers based on keywords
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
                        
                        # Generate summary with LLM
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
            content = PaperProcessor.generate_email_content(filtered_papers)
        else:
            print(f"INFO: No relevant papers found matching keywords for {time_period}")
            content = PaperProcessor.generate_email_content([], found_papers=False)
        
        # Always send an email, regardless of whether papers were found
        EmailSender.send_email(content)
        
    except Exception as e:
        print(f"ERROR: An exception occurred during job execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("===== END: Daily ArXiv Paper Fetching Job =====")

if __name__ == '__main__':
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='ArXiv Papers Fetcher with PDF Processing and RAG Database')
    parser.add_argument('--date', type=str, help='Specific date to query (YYYY-MM-DD format)')
    parser.add_argument('--keyword', type=str, help='Query database for papers matching a specific keyword')
    parser.add_argument('--schedule', action='store_true', help='Run as a scheduled daily job')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF downloads and processing')
    parser.add_argument('--no-email', action='store_true', help='Skip sending email')
    args = parser.parse_args()
    
    # Override config based on arguments
    if args.no_pdf:
        Config.DOWNLOAD_PDFS = False
    
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
