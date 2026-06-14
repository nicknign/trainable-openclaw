"""Verification: good trajectories must score higher than bad ones."""
import json
from trainable_openclaw.training.rubric_rules import RubricRuleEngine


def tc(name, args, status="success"):
    return {"name": name, "arguments": args, "result": {"status": status}}


engine = RubricRuleEngine()

good1 = {
    'conversations': [
        {'role': 'user', 'content': 'Hi, I am Alice Chen, zip 94102. Cancel my pending order.'},
        {'role': 'assistant', 'content': 'Let me look up your account.'},
        {'role': 'tool', 'content': json.dumps([tc('find_user_id_by_name_zip', {'first_name': 'Alice', 'last_name': 'Chen', 'zip': '94102'})])},
        {'role': 'assistant', 'content': 'You are Alice Chen, ID U001. Let me check orders.'},
        {'role': 'tool', 'content': json.dumps([tc('get_order_details', {'order_id': 'O002'})])},
        {'role': 'assistant', 'content': 'Order O002 for Running Shoes is pending. Would you like me to cancel it?'},
        {'role': 'user', 'content': 'Yes, please.'},
        {'role': 'assistant', 'content': 'Cancelled order O002. Refund $131.99 to PayPal. Anything else?'},
        {'role': 'tool', 'content': json.dumps([tc('cancel_pending_order', {'order_id': 'O002'})])},
        {'role': 'user', 'content': 'Thank you! Perfect.'},
    ],
    'status': 'complete',
}

good2 = {
    'conversations': [
        {'role': 'user', 'content': 'Hi, Bob Williams, U002. Return the fitness watch from order O003.'},
        {'role': 'assistant', 'content': 'Let me verify and check order O003.'},
        {'role': 'tool', 'content': json.dumps([tc('get_user_details', {'user_id': 'U002'})])},
        {'role': 'assistant', 'content': 'Confirmed Bob Williams. Looking at O003...'},
        {'role': 'tool', 'content': json.dumps([tc('get_order_details', {'order_id': 'O003'})])},
        {'role': 'assistant', 'content': 'Would you like me to return the Smart Fitness Watch (I004)?'},
        {'role': 'user', 'content': 'Yes please.'},
        {'role': 'assistant', 'content': 'Return processed. Refund $199.99 to card 5678. Auth RMA-O003. Anything else?'},
        {'role': 'tool', 'content': json.dumps([tc('return_delivered_order_items', {'order_id': 'O003', 'item_ids': ['I004'], 'payment_method': 'credit_card'})])},
        {'role': 'user', 'content': 'Thanks! All done.'},
    ],
    'status': 'complete',
}

good3 = {
    'conversations': [
        {'role': 'user', 'content': 'Hi, Alice Chen, zip 94102. What is my gift card balance?'},
        {'role': 'assistant', 'content': 'Let me look up your account.'},
        {'role': 'tool', 'content': json.dumps([tc('find_user_id_by_name_zip', {'first_name': 'Alice', 'last_name': 'Chen', 'zip': '94102'})])},
        {'role': 'assistant', 'content': 'Found account U001. Getting details...'},
        {'role': 'tool', 'content': json.dumps([tc('get_user_details', {'user_id': 'U001'})])},
        {'role': 'assistant', 'content': 'Gift card balance: $50.00. Anything else?'},
        {'role': 'user', 'content': 'No, thanks!'},
    ],
    'status': 'complete',
}

bad1 = {
    'conversations': [
        {'role': 'user', 'content': 'Cancel my order.'},
        {'role': 'assistant', 'content': 'OK, cancelling.'},
        {'role': 'tool', 'content': json.dumps([{'name': 'cancel_pending_order', 'result': {'status': 'error', 'message': 'Order not found'}}])},
        {'role': 'user', 'content': 'You did not even check who I am.'},
        {'role': 'assistant', 'content': 'Sorry, I cannot help you.'},
    ],
    'status': 'give_up',
}

bad2 = {
    'conversations': [
        {'role': 'user', 'content': 'I need help with my order.'},
        {'role': 'assistant', 'content': 'Here is a list of products.'},
        {'role': 'tool', 'content': json.dumps([tc('list_all_product_types', {})])},
        {'role': 'user', 'content': 'I asked about my ORDER, not products!'},
        {'role': 'assistant', 'content': 'Let me calculate 2+2.'},
        {'role': 'tool', 'content': json.dumps([tc('calculate', {'expression': '2+2'})])},
        {'role': 'user', 'content': 'Stop! Transfer me to a human.'},
        {'role': 'assistant', 'content': 'Transferring.'},
        {'role': 'tool', 'content': json.dumps([tc('transfer_to_human_agents', {'summary': 'confused'})])},
    ],
    'status': 'give_up',
}

bad3 = {
    'conversations': [
        {'role': 'user', 'content': 'Cancel my order please.'},
        {'role': 'assistant', 'content': 'OK.'},
    ],
    'status': 'give_up',
}

good_scores = [engine.compute_reward(t) for t in [good1, good2, good3]]
bad_scores = [engine.compute_reward(t) for t in [bad1, bad2, bad3]]

print("Good scores:", [f"{s:.3f}" for s in good_scores])
print("Bad scores: ", [f"{s:.3f}" for s in bad_scores])
print(f"Avg good: {sum(good_scores)/len(good_scores):.3f}")
print(f"Avg bad:  {sum(bad_scores)/len(bad_scores):.3f}")

for gi, g in enumerate(good_scores):
    for bi, b in enumerate(bad_scores):
        if g <= b:
            print(f"FAIL: good[{gi}]={g:.3f} <= bad[{bi}]={b:.3f}")
            raise SystemExit(1)

print("PASS: All good trajectories score higher than all bad trajectories")
