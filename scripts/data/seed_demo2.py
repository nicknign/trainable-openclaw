import json, sys, os
sys.path.insert(0, '/data/wangye/trainable-openclaw')
from trainable_openclaw.logging.conversation_store import ConversationStore

with open('data/seed_data.json', encoding='utf-8') as f:
    convs = json.load(f)

store = ConversationStore('data/conversations.db')
for c in convs:
    scores = [m['quality_score'] for m in c['messages'] if m['role'] == 'assistant']
    avg = sum(scores) / len(scores) if scores else None
    sid = store.create_session(c['user_id'], model=c['model'], metadata={'avg_quality': avg, 'source': 'demo_seed'})
    for m in c['messages']:
        store.add_message(sid, m['role'], m['content'],
            metadata={'quality_score': m.get('quality_score'), 'simulated_feedback': m.get('feedback')},
            stop_reason='stop' if m['role'] == 'assistant' else None)

stats = store.get_statistics()
print(f"Sessions: {stats['total_sessions']}")
print(f"Messages: {stats['total_messages']}")
print(f"Users: {stats['total_users']}")
print(f"Roles: {stats['role_distribution']}")
store.close()
print('Done')
