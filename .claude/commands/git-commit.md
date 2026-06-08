---
description: Quick git commit workflow
allowed-tools: Bash(git:*), Bash(pytest:*)
---

# Quick Commit

1. ReFactor code
2. Run tests: `pytest -v`
3. Update README.md
4. If tests pass:
   - Show status: `git status`
   - Stage all: `git add .`
   - Ask for commit message
   - Commit: `git commit -m "{message}"`
   - Push: `git push`
5. Done!