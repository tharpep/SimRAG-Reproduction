# Demo Logs

This directory contains logs from RAG and tuning demos, organized by type.

## Directory Structure:
```
logs/
├── README.md              # This file
├── rag/
│   └── rag_results.log    # RAG demo results
└── tuning/
    └── tuning_results.log # Tuning demo results
```

## Log Formats:

**RAG Results:**
```
2025-01-04 15:30:00 | llama3.2:1b | 2.34s | Q: What is Docker?...
A:     Docker is a platform for developing, shipping, and running applications in
    containers. It allows you to package your application and its dependencies
    into a lightweight, portable container that can run consistently across
    different environments...
CONTEXT: Retrieved 3 documents
  Doc 1 (score: 0.856): Docker is a platform for developing, shipping, and running applications in containers. It provides the ability to package and run an application...
  Doc 2 (score: 0.743): Containerization allows applications to run consistently across different environments. Docker containers include everything needed...
  Doc 3 (score: 0.692): Docker enables developers to separate their applications from their infrastructure, allowing for faster software delivery...
```

**Tuning Results:**
```
2025-01-04 15:45:00 | llama3.2:1b v1.0 | 45.2s | Loss: 2.3456 | Demo training - quick mode
```

## View Logs:
```bash
# View RAG results (Windows)
type logs\rag\rag_results.log

# View tuning results (Windows)
type logs\tuning\tuning_results.log

# View recent entries only (last 20 lines)
powershell "Get-Content logs\rag\rag_results.log | Select-Object -Last 20"
```

## What Gets Logged:
- **RAG Demos**: Questions, answers, response times, model used
- **Tuning Demos**: Model versions, training times, final loss, notes

## Notes:
- Logs are automatically created when you run demos
- Organized by demo type for easy navigation
- Simple, readable format
- **Auto-rotation**: Logs rotate when they reach 1MB (keeps 3 backups)
- Logs are git-ignored (won't be committed)
