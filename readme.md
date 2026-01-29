# Self-Healing SQL Data Analyst

A resilient, agentic AI system built with LangGraph that automatically corrects SQL errors and provides intelligent data analysis.

## 🎯 Overview

This system demonstrates the gold standard for "Agentic" workflows: **Self-Correction**. It takes natural language questions, generates SQL queries, executes them, and if errors occur, intelligently analyzes and corrects them automatically.

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                             │
│              "What was our highest growth                    │
│               region in Q3?"                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 1: Initialize Analysis                                 │
│  - Load database schema                                      │
│  - Set up initial state                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 2: Generate SQL Query                                  │
│  - Convert natural language to SQL                           │
│  - Use schema context                                        │
│  - Consider error history (if any)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 3: Execute SQL Query                                   │
│  - Run query against database                                │
│  - Capture results or errors                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
                  ┌────┴────┐
                  │SUCCESS? │
                  └────┬────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
       YES                           NO
        │                             │
        ▼                             ▼
┌──────────────────┐    ┌──────────────────────────────────┐
│  NODE 5:         │    │  NODE 4: Analyze Error           │
│  Generate        │    │  - Understand what went wrong    │
│  Explanation     │    │  - Generate corrected SQL        │
│                  │    │  - Learn from mistakes           │
└────────┬─────────┘    └────────┬─────────────────────────┘
         │                       │
         │                       │
         │              ┌────────┴────────┐
         │              │ RETRY < MAX?    │
         │              └────────┬────────┘
         │                       │
         │              ┌────────┴────────┐
         │             YES               NO
         │              │                 │
         │              │                 ▼
         │              │    ┌──────────────────────┐
         │              │    │  NODE 6: Handle      │
         │              │    │  Failure             │
         │              │    │  - Circuit breaker   │
         │              └───▶│  - Log errors        │
         │                   └──────────┬───────────┘
         │                              │
         ▼                              │
    ┌────────┐                          │
    │  END   │◀─────────────────────────┘
    └────────┘
```

### State Flow

The system maintains a comprehensive state through the entire workflow:

```python
{
    "question": str,           # User's natural language question
    "sql_query": str,          # Generated SQL query
    "query_result": str,       # Execution results
    "error_message": str,      # Last error encountered
    "error_history": list,     # All errors for learning
    "attempt_count": int,      # Number of attempts made
    "schema_info": str,        # Database schema context
    "explanation": str,        # Final natural language answer
    "status": str,             # Current workflow status
    "timestamp": str           # When analysis started
}
```

## 🚀 Key Features

### 1. Self-Correction
- Automatically detects SQL errors
- Analyzes error messages
- Generates corrected queries
- Learns from previous attempts

### 2. Resilient Systems
- Graceful failure handling
- Circuit breaker prevents infinite loops
- Maximum retry attempts configurable
- Comprehensive error logging

### 3. Error Handling
- Captures all database errors
- Provides contextual error analysis
- Maintains error history for learning
- Suggests improvements

## 💻 Usage

### Basic Example

```python
from self_healing_sql_analyst import run_analysis

# Ask a natural language question
result = run_analysis("What was our highest growth region in Q3?")

# The system will:
# 1. Generate SQL
# 2. Execute it
# 3. If error occurs, self-correct and retry
# 4. Return natural language explanation
```

### With Checkpointing

```python
result = run_analysis(
    question="What is the total revenue?",
    use_checkpointing=True  # Enable state persistence
)
```

## 🔧 Configuration

Edit the `Config` class to customize behavior:

```python
class Config:
    MAX_RETRY_ATTEMPTS = 3      # Maximum correction attempts
    DATABASE_PATH = "sales_data.db"  # Database location
    LLM_MODEL = "gpt-4"         # LLM model to use
    TIMEOUT_SECONDS = 30        # Query timeout
```

## 📊 Example Scenarios

### Scenario 1: Successful First Attempt

```
Question: "What is the total revenue?"
Generated SQL: SELECT SUM(revenue) as total_revenue FROM sales
Status: SUCCESS (1 attempt)
Result: Total revenue is $1,500,000
```

### Scenario 2: Self-Correction After Error

```
Question: "Show me sales by invalid_column"
Attempt 1: SELECT invalid_column FROM sales
Error: SQL Error: no such column: invalid_column
Attempt 2: SELECT * FROM sales  (corrected)
Status: SUCCESS (2 attempts)
Result: Here are all sales records...
```

### Scenario 3: Circuit Breaker Triggered

```
Question: "Complex query with persistent errors"
Attempt 1: Error - syntax error
Attempt 2: Error - still syntax error
Attempt 3: Error - still syntax error
Status: FAILED (3 attempts)
Result: Analysis failed. Error history provided with suggestions.
```

## 🎓 Design Patterns Used

### 1. Supervisor Pattern
Central coordinator manages workflow and delegates to specialized nodes.

### 2. Retry Pattern
Automatic retry with exponential learning from failures.

### 3. State Machine
Clean state transitions between workflow steps.

### 4. Circuit Breaker
Prevents infinite loops with maximum retry limit.

## 🔌 Integration Points

### Replace LLM Service
```python
class LLMService:
    @staticmethod
    def generate_sql(question, schema, error_context):
        # Replace with actual OpenAI/Anthropic API call
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Database schema: {schema}"},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
```

### Connect to Production Database
```python
class DatabaseManager:
    def __init__(self, connection_string):
        # Replace SQLite with PostgreSQL/MySQL
        self.engine = create_engine(connection_string)
```

## 📈 Benefits

### For Development
- Rapid prototyping of SQL queries
- Automatic error detection and correction
- Reduced debugging time
- Learning from historical errors

### For Production
- Resilient data analysis pipelines
- Self-healing capabilities
- Comprehensive error logging
- Graceful degradation

### For Users
- Natural language interface
- No SQL knowledge required
- Immediate, corrected results
- Clear explanations

## 🧪 Testing

The system includes comprehensive testing capabilities:

```python
# Test individual nodes
def test_sql_generation():
    state = {"question": "What is total revenue?"}
    result = generate_sql_query(state)
    assert "SELECT" in result["sql_query"]

# Test full workflow
def test_self_correction():
    result = run_analysis("Complex question")
    assert result["status"] in ["success", "failed"]
    assert result["attempt_count"] <= Config.MAX_RETRY_ATTEMPTS
```

## 🔍 Monitoring and Debugging

### Enable Verbose Logging
```python
# The system prints detailed progress
for step in app.stream(initial_state):
    print(f"Node: {step}")
    print(f"State: {step.values}")
```

### Checkpoint Analysis
```python
# Review state at each step
for state in app.get_state_history(config):
    print(f"Attempt: {state.values['attempt_count']}")
    print(f"Error: {state.values['error_message']}")
```

## 🎯 Real-World Applications

1. **Business Intelligence**: Natural language queries for executives
2. **Data Science**: Quick data exploration without SQL
3. **Customer Support**: Automated data retrieval for support teams
4. **Reporting**: Self-service analytics for non-technical users
5. **Audit & Compliance**: Automated data validation queries

## 🚦 Performance Considerations

- **Caching**: Implement query result caching for repeated questions
- **Parallelization**: Execute multiple queries simultaneously
- **Rate Limiting**: Control LLM API usage
- **Database Optimization**: Use query optimization and indexing

## 🔒 Security Best Practices

1. **SQL Injection Prevention**: Validate and sanitize all generated SQL
2. **Access Control**: Implement role-based database access
3. **Query Limits**: Restrict query complexity and execution time
4. **Audit Logging**: Track all queries and their sources

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📞 Support

For issues or questions, please open a GitHub issue or contact the maintainers.

---

**Built with LangGraph** - The framework for building stateful, agentic AI applications.
