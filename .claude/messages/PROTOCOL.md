# Inter-Agent Communication Protocol

File-based message passing between Claude Code subagents.
No daemon — agents use `.claude/agent_message.py` to send/check/reply.

## Directory Structure

```
.claude/messages/
├── PROTOCOL.md              # This file
├── disciplined-coder/
│   ├── inbox/               # Messages sent TO this agent
│   └── sent/                # Archive of messages sent BY this agent
├── e2e-code-tester/
│   ├── inbox/
│   └── sent/
├── research-scout/          # ...same structure
├── research-experiment-planner/
└── academic-content-writer/
```

## Message Format

Each message is a JSON file named `{timestamp_ms}-{short_uuid}.json`:

```json
{
  "id": "1781219574569-96b77f0f",
  "from": "disciplined-coder",
  "to": "e2e-code-tester",
  "type": "task_request",
  "subject": "Short summary (one line)",
  "body": "Detailed description...",
  "context": {
    "files": ["src/api/auth.py"],
    "branch": "main",
    "related_msg_id": "msg-id"
  },
  "timestamp": "2026-06-11T23:12:54Z",
  "status": "unread"
}
```

## Message Types

| Type | Sender | Meaning |
|------|--------|---------|
| `task_request` | Any agent | "Please do this task" |
| `status_update` | Any agent | "Here's what I finished" |
| `question` | Any agent | "I need clarification" |
| `handoff` | Any agent | "Transferring ownership of this work" |
| `reply` | Any agent | Response to a previous message |

## CLI Reference

```bash
# Send a message
python .claude/agent_message.py send \
  --to AGENT_NAME \
  --type TYPE \
  --subject "..." \
  --body "..." \
  [--context '{"key": "value"}'] \
  [--from-agent SENDER]

# Check inbox
python .claude/agent_message.py check [--agent AGENT] [--unread-only]

# Read a message
python .claude/agent_message.py read MSG_ID [--agent AGENT]

# Mark as read
python .claude/agent_message.py mark-read MSG_ID [--agent AGENT]

# Reply to a message
python .claude/agent_message.py reply MSG_ID --body "..." [--agent AGENT]

# List all agents and unread counts
python .claude/agent_message.py list-agents
```

## Agent Communication Rules

### 1. Check inbox at session start
Before starting assigned work, check your inbox:
```bash
python .claude/agent_message.py check --agent YOUR_NAME --unread-only
```

### 2. Process messages FIFO
Read oldest unread first. Mark as "read" after processing. Reply if action is needed from the sender.

### 3. Notify downstream on completion
When you finish a task, consider: does another agent need to act on this?
- If yes, send a `status_update` or `task_request` to them.
- Include `context.files` and `context.branch` so they know what changed.

### 4. Ask questions via message
If you hit an ambiguity and the sender is another agent (not the orchestrator),
send a `question` type message rather than blocking.

### 5. Never delete messages
Messages are the audit trail. Mark as "read"/"replied" but never delete.
Archive old messages by moving to `archive/` subdirectory if the inbox grows large.

## Common Agent Notification Triggers

| When this happens... | This agent notifies... | Type |
|---------------------|----------------------|------|
| disciplined-coder finishes a feature | e2e-code-tester | `task_request` |
| e2e-code-tester finds a bug | disciplined-coder | `task_request` |
| research-scout finds a paper/dataset | research-experiment-planner | `status_update` |
| research-experiment-planner designs experiment | disciplined-coder | `handoff` |
| disciplined-coder needs algorithm details | research-experiment-planner | `question` |
| Any agent completes a milestone | academic-content-writer | `status_update` |
| e2e-code-tester validates a module | disciplined-coder (if bugs) or academic-content-writer (if milestone) | `reply` or `status_update` |
