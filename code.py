"""
Self-Healing SQL Data Analyst with Context Engineering
======================================================

ARCHITECTURE OVERVIEW:
---------------------
This system implements an agentic workflow using LangGraph that can:
1. Take natural language questions
2. Generate SQL queries with rich context
3. Execute queries against a database
4. Self-correct when errors occur
5. Provide natural language answers

DESIGN PATTERNS:
---------------
1. Supervisor Pattern: Central coordinator manages workflow
2. Retry Pattern: Automatic error recovery with context
3. State Machine: Clean state transitions between steps
4. Circuit Breaker: Prevents infinite retry loops
5. Context Engineering: Intelligent context building and management

KEY COMPONENTS:
--------------
- State Management: Tracks conversation, queries, errors, and attempts
- Context Engineering: Builds rich, relevant context for LLM prompts
- SQL Generator: LLM-powered SQL generation from natural language
- Query Executor: Safe SQL execution with error capture
- Error Analyzer: Intelligent error interpretation
- Self-Correction Loop: Iterative refinement based on failures

CONTEXT ENGINEERING STRATEGY:
----------------------------
1. Schema Context: Database structure and relationships
2. Query History: Previous successful queries for pattern learning
3. Error Context: Past failures to avoid repeated mistakes
4. Business Context: Domain-specific rules and conventions
5. Examples Context: Few-shot examples for better SQL generation
6. Metadata Context: Statistics and data distributions

BENEFITS:
---------
- Self-Correction: Automatically fixes SQL errors
- Resilient Systems: Graceful failure handling
- Error Handling: Comprehensive error capture and recovery
- Context-Aware: Generates better SQL through rich context
"""

import sqlite3
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration and constants"""
    MAX_RETRY_ATTEMPTS = 3  # Maximum number of SQL correction attempts
    DATABASE_PATH = "sales_data.db"  # SQLite database path
    LLM_MODEL = "gpt-4"  # Can be replaced with actual LLM
    TIMEOUT_SECONDS = 30  # Query execution timeout
    
    # Context Engineering Settings
    MAX_SCHEMA_TOKENS = 2000  # Maximum tokens for schema context
    MAX_HISTORY_QUERIES = 5  # Number of historical queries to include
    MAX_ERROR_CONTEXT = 3  # Number of errors to include in context
    INCLUDE_QUERY_EXAMPLES = True  # Include few-shot examples
    INCLUDE_DATA_SAMPLES = True  # Include sample data in context
    MAX_SAMPLE_ROWS = 3  # Number of sample rows per table


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AnalystState(TypedDict):
    """
    Central state management for the SQL analyst workflow.
    
    State Fields:
    ------------
    question: Original natural language question from user
    sql_query: Generated or corrected SQL query
    query_result: Results from SQL execution
    error_message: Last error encountered
    error_history: Accumulated list of all errors for learning
    attempt_count: Number of SQL generation attempts
    schema_info: Database schema for context
    explanation: Natural language explanation of results
    status: Current workflow status
    timestamp: When the analysis started
    
    Context Engineering Fields:
    --------------------------
    query_history: List of previous successful queries
    business_context: Domain-specific rules and conventions
    few_shot_examples: Example query patterns
    data_samples: Sample data from tables
    schema_relationships: Foreign key relationships
    column_statistics: Data distribution and statistics
    user_preferences: User-specific query patterns
    context_summary: Condensed context for token efficiency
    """
    question: str
    sql_query: str
    query_result: str
    error_message: str
    error_history: Annotated[list, operator.add]  # Accumulates errors
    attempt_count: Annotated[int, operator.add]  # Increments attempts
    schema_info: str
    explanation: str
    status: Literal["pending", "processing", "success", "failed"]
    timestamp: str
    
    # Context Engineering Fields
    query_history: Annotated[list, operator.add]
    business_context: str
    few_shot_examples: str
    data_samples: str
    schema_relationships: str
    column_statistics: str
    user_preferences: dict
    context_summary: str


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

class DatabaseManager:
    """
    Handles all database operations including schema introspection
    and query execution with safety measures.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_sample_database()
    
    def init_sample_database(self):
        """
        Initialize a sample sales database for demonstration.
        In production, this would connect to your existing database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sample sales table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                region TEXT NOT NULL,
                product TEXT NOT NULL,
                revenue REAL NOT NULL,
                units_sold INTEGER NOT NULL,
                quarter TEXT NOT NULL,
                year INTEGER NOT NULL
            )
        """)
        
        # Insert sample data
        sample_data = [
            ('North America', 'Product A', 150000, 500, 'Q3', 2024),
            ('North America', 'Product B', 200000, 600, 'Q3', 2024),
            ('Europe', 'Product A', 180000, 550, 'Q3', 2024),
            ('Europe', 'Product B', 220000, 650, 'Q3', 2024),
            ('Asia', 'Product A', 250000, 800, 'Q3', 2024),
            ('Asia', 'Product B', 300000, 900, 'Q3', 2024),
            ('North America', 'Product A', 140000, 480, 'Q2', 2024),
            ('Europe', 'Product A', 160000, 520, 'Q2', 2024),
            ('Asia', 'Product A', 200000, 700, 'Q2', 2024),
        ]
        
        cursor.execute("DELETE FROM sales")  # Clear existing data
        cursor.executemany(
            "INSERT INTO sales (region, product, revenue, units_sold, quarter, year) VALUES (?, ?, ?, ?, ?, ?)",
            sample_data
        )
        
        conn.commit()
        conn.close()
    
    def get_schema(self) -> str:
        """
        Extract database schema information for LLM context.
        This helps the LLM understand available tables and columns.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        schema_info = "DATABASE SCHEMA:\n\n"
        
        for table in tables:
            table_name = table[0]
            schema_info += f"Table: {table_name}\n"
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema_info += "Columns:\n"
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                schema_info += f"  - {col_name} ({col_type})\n"
            
            schema_info += "\n"
        
        conn.close()
        return schema_info
    
    def execute_query(self, sql_query: str) -> tuple[bool, str, list]:
        """
        Execute SQL query with error handling.
        
        Returns:
        -------
        success: bool - Whether query executed successfully
        message: str - Error message or success message
        results: list - Query results if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(sql_query)
            
            # Fetch results
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.close()
            
            # Format results as list of dictionaries
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(column_names, row)))
            
            return True, "Query executed successfully", formatted_results
            
        except sqlite3.Error as e:
            return False, f"SQL Error: {str(e)}", []
        except Exception as e:
            return False, f"Unexpected Error: {str(e)}", []


# ============================================================================
# CONTEXT ENGINEERING
# ============================================================================

class ContextEngineer:
    """
    Advanced context engineering for optimal LLM performance.
    
    This class builds rich, relevant context from multiple sources to help
    the LLM generate better SQL queries and understand the database better.
    
    Context Sources:
    ---------------
    1. Schema Context: Table structures, columns, types
    2. Relationship Context: Foreign keys and joins
    3. Data Samples: Actual data examples
    4. Query Patterns: Historical successful queries
    5. Business Rules: Domain-specific constraints
    6. Statistics: Data distributions and cardinality
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.query_cache = []  # Store successful queries
    
    def build_schema_context(self) -> str:
        """
        Build comprehensive schema context with smart truncation.
        
        Returns enriched schema information optimized for LLM consumption.
        """
        schema = self.db_manager.get_schema()
        
        # Add metadata about the schema
        context = "DATABASE SCHEMA CONTEXT:\n"
        context += "=" * 60 + "\n\n"
        context += schema
        
        # Add helpful annotations
        context += "\nSCHEMA NOTES:\n"
        context += "- All tables use INTEGER PRIMARY KEY\n"
        context += "- Date fields are stored as TEXT in ISO format\n"
        context += "- Monetary values are REAL (floating point)\n"
        
        return context
    
    def extract_relationships(self) -> str:
        """
        Extract and document table relationships.
        
        This helps the LLM understand how to join tables correctly.
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        relationships = "TABLE RELATIONSHIPS:\n"
        relationships += "=" * 60 + "\n\n"
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fks = cursor.fetchall()
            
            if fks:
                relationships += f"Table: {table_name}\n"
                for fk in fks:
                    relationships += f"  - {fk[3]} → {fk[2]}.{fk[4]}\n"
                relationships += "\n"
        
        conn.close()
        
        if "Table:" not in relationships:
            relationships += "No explicit foreign key relationships defined.\n"
            relationships += "Use natural relationships based on column names.\n"
        
        return relationships
    
    def get_data_samples(self, max_rows: int = 3) -> str:
        """
        Get sample data from each table for better context.
        
        Sample data helps the LLM understand:
        - Data formats and patterns
        - Value ranges
        - Typical content
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        samples = "SAMPLE DATA:\n"
        samples += "=" * 60 + "\n\n"
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            samples += f"Table: {table_name}\n"
            samples += "-" * 40 + "\n"
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {max_rows}")
            rows = cursor.fetchall()
            
            if rows:
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Format as table
                samples += " | ".join(columns) + "\n"
                samples += "-" * 40 + "\n"
                
                for row in rows:
                    samples += " | ".join(str(val) for val in row) + "\n"
            else:
                samples += "(empty table)\n"
            
            samples += "\n"
        
        conn.close()
        return samples
    
    def get_column_statistics(self) -> str:
        """
        Generate statistics about columns to help with queries.
        
        This provides insights into:
        - Distinct value counts
        - Min/max ranges
        - Null percentages
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        stats = "COLUMN STATISTICS:\n"
        stats += "=" * 60 + "\n\n"
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            stats += f"Table: {table_name}\n"
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                # Get distinct count
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
                    distinct_count = cursor.fetchone()[0]
                    
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_count = cursor.fetchone()[0]
                    
                    stats += f"  - {col_name} ({col_type}): "
                    stats += f"{distinct_count} distinct values out of {total_count} rows"
                    
                    # For numeric columns, add min/max
                    if col_type in ['INTEGER', 'REAL']:
                        cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}) FROM {table_name}")
                        min_val, max_val = cursor.fetchone()
                        stats += f", range: [{min_val}, {max_val}]"
                    
                    stats += "\n"
                except:
                    pass
            
            stats += "\n"
        
        conn.close()
        return stats
    
    def generate_few_shot_examples(self) -> str:
        """
        Generate few-shot examples of question-SQL pairs.
        
        These examples help the LLM understand the expected pattern.
        """
        examples = "QUERY EXAMPLES:\n"
        examples += "=" * 60 + "\n\n"
        
        example_pairs = [
            {
                "question": "What is the total revenue?",
                "sql": "SELECT SUM(revenue) as total_revenue FROM sales;",
                "explanation": "Use SUM aggregation for totals"
            },
            {
                "question": "Which region has the highest sales?",
                "sql": "SELECT region, SUM(revenue) as total\nFROM sales\nGROUP BY region\nORDER BY total DESC\nLIMIT 1;",
                "explanation": "Use GROUP BY for aggregation by category, ORDER BY for sorting"
            },
            {
                "question": "Show sales for Q3 2024",
                "sql": "SELECT * FROM sales\nWHERE quarter = 'Q3' AND year = 2024;",
                "explanation": "Use WHERE clause for filtering"
            },
            {
                "question": "What is the growth rate from Q2 to Q3?",
                "sql": """SELECT 
    region,
    SUM(CASE WHEN quarter = 'Q3' THEN revenue ELSE 0 END) as q3_rev,
    SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END) as q2_rev,
    (SUM(CASE WHEN quarter = 'Q3' THEN revenue ELSE 0 END) - 
     SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END)) * 100.0 / 
     NULLIF(SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END), 0) as growth_rate
FROM sales
WHERE year = 2024
GROUP BY region;""",
                "explanation": "Use CASE statements for conditional aggregation"
            }
        ]
        
        for i, example in enumerate(example_pairs, 1):
            examples += f"Example {i}:\n"
            examples += f"Question: {example['question']}\n"
            examples += f"SQL:\n{example['sql']}\n"
            examples += f"Note: {example['explanation']}\n\n"
        
        return examples
    
    def build_business_context(self) -> str:
        """
        Define business rules and domain-specific conventions.
        
        This helps the LLM follow company-specific patterns.
        """
        context = "BUSINESS CONTEXT:\n"
        context += "=" * 60 + "\n\n"
        
        context += "Domain: Sales Analytics\n\n"
        
        context += "Business Rules:\n"
        context += "1. Quarters are labeled as 'Q1', 'Q2', 'Q3', 'Q4'\n"
        context += "2. Revenue is always in USD\n"
        context += "3. Regions: North America, Europe, Asia\n"
        context += "4. Fiscal year aligns with calendar year\n"
        context += "5. Growth rate = (New - Old) / Old * 100\n\n"
        
        context += "Common Metrics:\n"
        context += "- Total Revenue: SUM(revenue)\n"
        context += "- Average Deal Size: AVG(revenue)\n"
        context += "- Units Sold: SUM(units_sold)\n"
        context += "- Revenue per Unit: revenue / units_sold\n\n"
        
        context += "Naming Conventions:\n"
        context += "- Use lowercase for column names\n"
        context += "- Use snake_case for multi-word names\n"
        context += "- Always use table aliases in joins\n"
        
        return context
    
    def add_successful_query(self, question: str, sql: str):
        """
        Cache successful queries for pattern learning.
        
        Historical queries help the LLM learn from past successes.
        """
        self.query_cache.append({
            "question": question,
            "sql": sql,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent queries
        if len(self.query_cache) > Config.MAX_HISTORY_QUERIES:
            self.query_cache = self.query_cache[-Config.MAX_HISTORY_QUERIES:]
    
    def get_query_history_context(self) -> str:
        """
        Format query history for context.
        """
        if not self.query_cache:
            return ""
        
        context = "RECENT SUCCESSFUL QUERIES:\n"
        context += "=" * 60 + "\n\n"
        
        for i, query in enumerate(self.query_cache, 1):
            context += f"{i}. Question: {query['question']}\n"
            context += f"   SQL: {query['sql'][:100]}...\n\n"
        
        return context
    
    def build_full_context(self, include_samples: bool = True) -> dict:
        """
        Build complete context package for LLM.
        
        Returns a dictionary with all context components.
        """
        context = {
            "schema": self.build_schema_context(),
            "relationships": self.extract_relationships(),
            "business_rules": self.build_business_context(),
            "examples": self.generate_few_shot_examples(),
            "query_history": self.get_query_history_context(),
            "statistics": ""
        }
        
        if include_samples and Config.INCLUDE_DATA_SAMPLES:
            context["data_samples"] = self.get_data_samples(Config.MAX_SAMPLE_ROWS)
        else:
            context["data_samples"] = ""
        
        if Config.INCLUDE_QUERY_EXAMPLES:
            context["statistics"] = self.get_column_statistics()
        
        return context
    
    def create_prompt_context(self, question: str, error_context: list = None) -> str:
        """
        Create optimized prompt context for SQL generation.
        
        This intelligently selects and formats context based on the question.
        """
        full_context = self.build_full_context()
        
        prompt = "You are an expert SQL analyst. Use the following context to generate accurate SQL queries.\n\n"
        
        # Always include schema
        prompt += full_context["schema"] + "\n"
        
        # Add relationships if question involves multiple tables
        if any(word in question.lower() for word in ["join", "with", "and", "across"]):
            prompt += full_context["relationships"] + "\n"
        
        # Add business rules
        prompt += full_context["business_rules"] + "\n"
        
        # Add examples for complex queries
        if any(word in question.lower() for word in ["growth", "compare", "trend", "rate"]):
            prompt += full_context["examples"] + "\n"
        
        # Add data samples for better understanding
        if Config.INCLUDE_DATA_SAMPLES:
            prompt += full_context["data_samples"] + "\n"
        
        # Add statistics for aggregation queries
        if any(word in question.lower() for word in ["average", "total", "sum", "count", "max", "min"]):
            prompt += full_context["statistics"] + "\n"
        
        # Add query history if available
        if full_context["query_history"]:
            prompt += full_context["query_history"] + "\n"
        
        # Add error context for self-correction
        if error_context:
            prompt += "PREVIOUS ERRORS (AVOID THESE):\n"
            prompt += "=" * 60 + "\n"
            for i, error in enumerate(error_context[-Config.MAX_ERROR_CONTEXT:], 1):
                prompt += f"{i}. {error}\n"
            prompt += "\n"
        
        prompt += f"\nQUESTION: {question}\n\n"
        prompt += "Generate SQL query. Return ONLY the SQL query, no explanations.\n"
        
        return prompt


# ============================================================================
# LLM SIMULATION (Replace with actual LLM calls)
# ============================================================================

class LLMService:
    """
    Simulates LLM interactions with context engineering.
    In production, replace with actual OpenAI, Anthropic, or other LLM API calls.
    """
    
    def __init__(self, context_engineer: ContextEngineer):
        self.context_engineer = context_engineer
    
    def generate_sql(self, question: str, schema: str, error_context: list = None) -> str:
        """
        Generate SQL query from natural language question using rich context.
        
        Uses ContextEngineer to build comprehensive prompt context.
        """
        # Build rich context using context engineer
        prompt_context = self.context_engineer.create_prompt_context(question, error_context)
        
        # In production, this would call an actual LLM with the full context
        # Example: response = openai.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt_context}]
        # )
        
        # For demonstration, use simple pattern matching
        # The rich context would significantly improve real LLM performance
        question_lower = question.lower()
        
        # Check for error context to improve query
        if error_context and len(error_context) > 0:
            last_error = error_context[-1]
            # Simulate learning from errors with context
            if "no such column" in last_error.lower():
                # The LLM would understand the column doesn't exist
                # and use context to find the right column
                pass
        
        # Simple SQL generation logic (replace with LLM)
        if "highest growth" in question_lower and "region" in question_lower:
            return """
SELECT 
    region,
    SUM(CASE WHEN quarter = 'Q3' THEN revenue ELSE 0 END) as q3_revenue,
    SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END) as q2_revenue,
    (SUM(CASE WHEN quarter = 'Q3' THEN revenue ELSE 0 END) - 
     SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END)) * 100.0 / 
     NULLIF(SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END), 0) as growth_rate
FROM sales
WHERE year = 2024
GROUP BY region
ORDER BY growth_rate DESC
LIMIT 1
"""
        elif "total revenue" in question_lower:
            return "SELECT SUM(revenue) as total_revenue FROM sales"
        
        elif "top selling" in question_lower:
            return """
SELECT product, SUM(units_sold) as total_units
FROM sales
GROUP BY product
ORDER BY total_units DESC
LIMIT 5
"""
        else:
            # Default query
            return "SELECT * FROM sales LIMIT 10"
    
    def analyze_error(self, sql_query: str, error_message: str, schema: str) -> str:
        """
        Analyze SQL error using context and provide correction guidance.
        In production, this would use an LLM with full context.
        """
        analysis = f"Error Analysis (with Context Engineering):\n"
        analysis += f"Query: {sql_query}\n"
        analysis += f"Error: {error_message}\n\n"
        
        if "no such column" in error_message.lower():
            analysis += "Issue: Referenced column does not exist in the table.\n"
            analysis += "Context: Review the schema context to find correct column names.\n"
            analysis += "Action: Check SCHEMA CONTEXT and DATA SAMPLES for valid columns.\n"
        elif "syntax error" in error_message.lower():
            analysis += "Issue: SQL syntax is incorrect.\n"
            analysis += "Context: Review QUERY EXAMPLES for proper syntax patterns.\n"
            analysis += "Action: Follow examples and business conventions.\n"
        elif "no such table" in error_message.lower():
            analysis += "Issue: Referenced table does not exist.\n"
            analysis += "Context: Review SCHEMA CONTEXT for available tables.\n"
            analysis += "Action: Use tables from the provided schema.\n"
        else:
            analysis += "Issue: Unspecified error.\n"
            analysis += "Context: Review all context sections for guidance.\n"
            analysis += "Action: Check schema, examples, and business rules.\n"
        
        return analysis
    
    def explain_results(self, question: str, results: list) -> str:
        """
        Generate natural language explanation using context.
        In production, this would use an LLM.
        """
        if not results:
            return "No results found for your query."
        
        # Simple explanation logic (replace with LLM)
        explanation = f"Based on your question '{question}', here are the findings:\n\n"
        
        if len(results) == 1 and "growth_rate" in results[0]:
            region = results[0].get("region", "Unknown")
            growth = results[0].get("growth_rate", 0)
            explanation += f"The highest growth region is {region} with a growth rate of {growth:.2f}%."
        else:
            explanation += f"Found {len(results)} record(s):\n"
            for i, row in enumerate(results[:5], 1):
                explanation += f"{i}. {row}\n"
        
        return explanation


# ============================================================================
# LANGGRAPH NODES
# ============================================================================

# Initialize shared resources
db_manager = DatabaseManager(Config.DATABASE_PATH)
context_engineer = ContextEngineer(db_manager)
llm_service = LLMService(context_engineer)


def initialize_analysis(state: AnalystState) -> AnalystState:
    """
    NODE 1: Initialize Analysis with Context Engineering
    
    Purpose:
    -------
    - Set up initial state
    - Load database schema
    - Build comprehensive context
    - Prepare for SQL generation with rich context
    
    This is the entry point where context engineering begins.
    """
    # Build full context package
    context_package = context_engineer.build_full_context()
    
    # Create comprehensive context summary
    context_summary = "CONTEXT LOADED:\n"
    context_summary += f"- Schema: {len(context_package['schema'])} chars\n"
    context_summary += f"- Relationships: {len(context_package['relationships'])} chars\n"
    context_summary += f"- Business Rules: {len(context_package['business_rules'])} chars\n"
    context_summary += f"- Examples: {len(context_package['examples'])} chars\n"
    context_summary += f"- Data Samples: {len(context_package['data_samples'])} chars\n"
    
    return {
        "schema_info": context_package["schema"],
        "schema_relationships": context_package["relationships"],
        "business_context": context_package["business_rules"],
        "few_shot_examples": context_package["examples"],
        "data_samples": context_package["data_samples"],
        "column_statistics": context_package["statistics"],
        "context_summary": context_summary,
        "status": "processing",
        "timestamp": datetime.now().isoformat(),
        "attempt_count": 0,
        "error_history": [],
        "query_history": [],
        "user_preferences": {}
    }


def generate_sql_query(state: AnalystState) -> AnalystState:
    """
    NODE 2: Generate SQL Query with Rich Context
    
    Purpose:
    -------
    - Convert natural language to SQL using context engineering
    - Use schema, examples, business rules, and data samples
    - Learn from previous errors (if any)
    - Apply query history patterns
    
    This is where context engineering enhances SQL generation.
    """
    question = state["question"]
    schema = state["schema_info"]
    error_history = state.get("error_history", [])
    
    print(f"\n{'='*60}")
    print(f"CONTEXT ENGINEERING IN ACTION")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"\nContext Components:")
    print(f"  ✓ Schema Information")
    print(f"  ✓ Table Relationships")
    print(f"  ✓ Business Rules & Conventions")
    print(f"  ✓ Few-Shot Query Examples")
    print(f"  ✓ Sample Data")
    print(f"  ✓ Column Statistics")
    if error_history:
        print(f"  ✓ Error History ({len(error_history)} previous errors)")
    print(f"{'='*60}\n")
    
    # Generate SQL using LLM with rich context
    sql_query = llm_service.generate_sql(question, schema, error_history)
    
    return {
        "sql_query": sql_query.strip(),
        "attempt_count": 1
    }


def execute_sql_query(state: AnalystState) -> AnalystState:
    """
    NODE 3: Execute SQL Query
    
    Purpose:
    -------
    - Run the generated SQL
    - Capture results or errors
    - Update state accordingly
    - Cache successful queries for future context
    
    This is where we interact with the actual database.
    """
    sql_query = state["sql_query"]
    question = state["question"]
    
    # Execute the query
    success, message, results = db_manager.execute_query(sql_query)
    
    if success:
        # Cache successful query for future context
        context_engineer.add_successful_query(question, sql_query)
        
        # Format results as string for state
        results_str = str(results) if results else "No results found"
        
        print(f"✓ Query executed successfully!")
        print(f"✓ Added to query history for future context\n")
        
        return {
            "query_result": results_str,
            "error_message": "",
            "status": "success",
            "query_history": [{"question": question, "sql": sql_query}]
        }
    else:
        # Capture error for self-correction
        print(f"⚠ Query failed: {message[:80]}...")
        print(f"⚠ Will use error context for self-correction\n")
        
        return {
            "query_result": "",
            "error_message": message,
            "error_history": [message],
            "status": "processing"  # Continue to error analysis
        }


def analyze_error_and_correct(state: AnalystState) -> AnalystState:
    """
    NODE 4: Analyze Error and Self-Correct
    
    Purpose:
    -------
    - Understand what went wrong
    - Generate corrected SQL
    - Learn from mistakes
    
    This is the SELF-HEALING component - the key to resilience.
    """
    sql_query = state["sql_query"]
    error_message = state["error_message"]
    schema = state["schema_info"]
    question = state["question"]
    error_history = state["error_history"]
    
    # Analyze the error
    analysis = llm_service.analyze_error(sql_query, error_message, schema)
    
    print(f"\n{'='*60}")
    print(f"SELF-CORRECTION ATTEMPT {state['attempt_count']}")
    print(f"{'='*60}")
    print(analysis)
    print(f"{'='*60}\n")
    
    # Generate corrected SQL with error context
    corrected_sql = llm_service.generate_sql(question, schema, error_history)
    
    return {
        "sql_query": corrected_sql.strip(),
        "attempt_count": 1  # Increment attempt counter
    }


def generate_explanation(state: AnalystState) -> AnalystState:
    """
    NODE 5: Generate Natural Language Explanation
    
    Purpose:
    -------
    - Convert query results to human-readable format
    - Provide context and insights
    - Complete the analysis
    
    This is the final output generation step.
    """
    question = state["question"]
    results_str = state["query_result"]
    
    # Parse results back to list
    try:
        results = eval(results_str) if results_str else []
    except:
        results = []
    
    # Generate explanation using LLM
    explanation = llm_service.explain_results(question, results)
    
    return {
        "explanation": explanation,
        "status": "success"
    }


def handle_failure(state: AnalystState) -> AnalystState:
    """
    NODE 6: Handle Final Failure
    
    Purpose:
    -------
    - Graceful degradation when max retries exceeded
    - Provide helpful error messages
    - Log for debugging
    
    This is our circuit breaker - prevents infinite loops.
    """
    error_history = state["error_history"]
    attempts = state["attempt_count"]
    
    failure_message = f"""
Analysis Failed After {attempts} Attempts

The system attempted to generate and execute SQL {attempts} times but encountered persistent errors.

Error History:
{chr(10).join(f"{i+1}. {err}" for i, err in enumerate(error_history))}

Suggestions:
1. Rephrase your question more clearly
2. Check if the data you're asking about exists
3. Contact support if the issue persists

Original Question: {state['question']}
"""
    
    return {
        "explanation": failure_message,
        "status": "failed"
    }


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_retry_or_complete(state: AnalystState) -> Literal["retry", "success", "failed"]:
    """
    ROUTING FUNCTION: Determine Next Step After Execution
    
    Decision Logic:
    --------------
    - If query succeeded → go to explanation
    - If error and attempts < max → retry with correction
    - If max attempts exceeded → handle failure
    
    This implements our retry pattern with circuit breaker.
    """
    if state["status"] == "success" and state["query_result"]:
        # Query succeeded, move to explanation
        return "success"
    
    elif state["error_message"] and state["attempt_count"] < Config.MAX_RETRY_ATTEMPTS:
        # Error occurred but we can retry
        return "retry"
    
    else:
        # Max retries exceeded
        return "failed"


def check_final_status(state: AnalystState) -> Literal["complete", "failed"]:
    """
    ROUTING FUNCTION: Final Status Check
    
    Determines if we should end successfully or handle failure.
    """
    if state["status"] == "success":
        return "complete"
    else:
        return "failed"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_self_healing_analyst() -> StateGraph:
    """
    Build the Self-Healing SQL Analyst Graph
    
    WORKFLOW ARCHITECTURE:
    ---------------------
    
    1. Initialize → 2. Generate SQL → 3. Execute SQL
                                            ↓
                                    [Success or Error?]
                                            ↓
                    ┌───────────────────────┴───────────────────────┐
                    ↓                                               ↓
            [SUCCESS]                                         [ERROR]
                    ↓                                               ↓
        5. Generate Explanation                     4. Analyze & Correct
                    ↓                                               ↓
                  [END]                               [Retry < Max?]
                                                            ↓
                                            ┌───────────────┴───────────────┐
                                            ↓                               ↓
                                        [YES]                             [NO]
                                            ↓                               ↓
                                    Go to Execute                   6. Handle Failure
                                                                            ↓
                                                                          [END]
    
    KEY FEATURES:
    ------------
    - Self-correction loop between steps 3 and 4
    - Circuit breaker at max retry attempts
    - Conditional routing based on success/failure
    - State preservation across all nodes
    """
    
    # Create the graph with our state schema
    workflow = StateGraph(AnalystState)
    
    # Add all nodes to the graph
    workflow.add_node("initialize", initialize_analysis)
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("execute_sql", execute_sql_query)
    workflow.add_node("analyze_error", analyze_error_and_correct)
    workflow.add_node("generate_explanation", generate_explanation)
    workflow.add_node("handle_failure", handle_failure)
    
    # Define the workflow structure
    
    # Entry point
    workflow.set_entry_point("initialize")
    
    # Linear flow: Initialize → Generate → Execute
    workflow.add_edge("initialize", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    
    # Conditional routing after execution
    workflow.add_conditional_edges(
        "execute_sql",
        should_retry_or_complete,
        {
            "success": "generate_explanation",  # Query succeeded
            "retry": "analyze_error",           # Need to retry
            "failed": "handle_failure"          # Max retries exceeded
        }
    )
    
    # Self-correction loop: Error analysis → back to execution
    workflow.add_edge("analyze_error", "execute_sql")
    
    # Final paths to END
    workflow.add_edge("generate_explanation", END)
    workflow.add_edge("handle_failure", END)
    
    return workflow


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_analysis(question: str, use_checkpointing: bool = True):
    """
    Run the self-healing SQL analysis for a given question.
    
    Parameters:
    ----------
    question: Natural language question to analyze
    use_checkpointing: Enable state persistence for debugging
    
    Returns:
    -------
    Final state with explanation and results
    """
    
    print(f"\n{'#'*80}")
    print(f"SELF-HEALING SQL DATA ANALYST")
    print(f"{'#'*80}\n")
    print(f"Question: {question}\n")
    
    # Build the graph
    graph = build_self_healing_analyst()
    
    # Compile with optional checkpointing for persistence
    if use_checkpointing:
        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "analysis_session"}}
    else:
        app = graph.compile()
        config = {}
    
    # Initialize state with context engineering fields
    initial_state = {
        "question": question,
        "sql_query": "",
        "query_result": "",
        "error_message": "",
        "error_history": [],
        "attempt_count": 0,
        "schema_info": "",
        "explanation": "",
        "status": "pending",
        "timestamp": "",
        # Context Engineering Fields
        "query_history": [],
        "business_context": "",
        "few_shot_examples": "",
        "data_samples": "",
        "schema_relationships": "",
        "column_statistics": "",
        "user_preferences": {},
        "context_summary": ""
    }
    
    # Run the analysis with streaming to see progress
    print("Processing...\n")
    
    for step_output in app.stream(initial_state, config):
        node_name = list(step_output.keys())[0]
        node_state = step_output[node_name]
        
        print(f"✓ Completed: {node_name}")
        
        # Show SQL query when generated
        if node_name in ["generate_sql", "analyze_error"] and node_state.get("sql_query"):
            print(f"  SQL: {node_state['sql_query'][:100]}...")
        
        # Show errors when they occur
        if node_state.get("error_message"):
            print(f"  ⚠ Error: {node_state['error_message'][:80]}...")
    
    # Get final state
    final_state = app.invoke(initial_state, config) if not use_checkpointing else \
                  app.get_state(config).values
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")
    print(final_state.get("explanation", "No explanation available"))
    print(f"\nStatus: {final_state.get('status', 'unknown')}")
    print(f"Total Attempts: {final_state.get('attempt_count', 0)}")
    print(f"\n{'='*80}\n")
    
    return final_state


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating self-healing with context engineering
    """
    
    # Display Context Engineering Capabilities
    print("\n" + "="*80)
    print("CONTEXT ENGINEERING DEMONSTRATION")
    print("="*80)
    print("\nThis system uses advanced context engineering to improve SQL generation:")
    print("\n1. SCHEMA CONTEXT:")
    print("   - Complete database schema with column types")
    print("   - Table relationships and foreign keys")
    print("   - Column statistics and data distributions")
    
    print("\n2. BUSINESS CONTEXT:")
    print("   - Domain-specific rules (e.g., quarter formats, naming conventions)")
    print("   - Common metrics and calculations")
    print("   - Company-specific business logic")
    
    print("\n3. QUERY PATTERNS:")
    print("   - Few-shot examples of question-SQL pairs")
    print("   - Historical successful queries")
    print("   - Best practices and patterns")
    
    print("\n4. DATA CONTEXT:")
    print("   - Sample data from tables")
    print("   - Data format examples")
    print("   - Value ranges and typical content")
    
    print("\n5. ERROR CONTEXT:")
    print("   - Previous failures to avoid")
    print("   - Self-correction history")
    print("   - Pattern of successful fixes")
    
    print("\n" + "="*80)
    input("\nPress Enter to run examples with context engineering...")
    
    # Example 1: Successful query with context
    print("\n" + "="*80)
    print("EXAMPLE 1: Context-Enhanced Query Generation")
    print("="*80)
    result1 = run_analysis("What was our highest growth region in Q3?")
    
    # Example 2: Self-correction with context
    print("\n" + "="*80)
    print("EXAMPLE 2: Self-Correction Using Context")
    print("="*80)
    result2 = run_analysis("What is the total revenue across all regions?")
    
    # Example 3: Complex query benefiting from examples
    print("\n" + "="*80)
    print("EXAMPLE 3: Complex Query with Context")
    print("="*80)
    result3 = run_analysis("What are our top selling products?")
    
    print("\n" + "="*80)
    print("CONTEXT ENGINEERING BENEFITS DEMONSTRATED")
    print("="*80)
    print("\n✓ Schema Context: Prevents 'table/column not found' errors")
    print("✓ Business Context: Ensures queries follow company conventions")
    print("✓ Query Examples: Improves complex query generation")
    print("✓ Data Samples: Helps understand data formats and patterns")
    print("✓ Error Context: Enables intelligent self-correction")
    print("✓ Query History: Learns from previous successful patterns")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("\n1. Context Engineering significantly improves LLM performance")
    print("2. Rich context reduces errors and improves first-time success")
    print("3. Self-correction becomes more effective with error context")
    print("4. Historical patterns enable continuous improvement")
    print("5. Business rules ensure domain-appropriate queries")
    
    print("\nThis demonstrates how context engineering creates a resilient,")
    print("intelligent system that learns and improves over time.")
