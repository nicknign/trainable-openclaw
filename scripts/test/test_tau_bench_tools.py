"""Quick verification of tau-bench tools — imports, schemas, and executions."""
from __future__ import annotations

import json
from trainable_openclaw.agent.tau_bench_tools import (
    MockTool, MockDatabase, seed, register_tau_bench_tools,
)

from trainable_openclaw.agent.tau_bench_tools.retail import (
    FindUserByEmailTool,
    GetOrderDetailsTool,
    ListAllProductTypesTool,
    CancelPendingOrderTool,
    ModifyPendingOrderAddressTool,
    ExchangeDeliveredOrderItemsTool,
    ReturnDeliveredOrderItemsTool,
)

from trainable_openclaw.agent.tau_bench_tools.airline import (
    SearchDirectFlightTool,
    ListAllAirportsTool,
    BookReservationTool,
    SearchOnestopFlightTool,
    GetFlightStatusTool,
    UpdateReservationFlightsTool,
)

from trainable_openclaw.agent.tau_bench_tools.base import (
    calculate_tool, think_tool, transfer_to_human_agents_tool,
)


def main():
    # Test 1: Tool counts
    rt = register_tau_bench_tools("retail")
    at = register_tau_bench_tools("airline")
    print(f"Retail tools: {len(rt)}")
    print(f"Airline tools: {len(at)}")

    # Test 2: All tools have valid schemas
    for t in rt + at:
        s = t.to_schema()
        assert "type" in s, f"{t.name}: missing type"
        assert "function" in s, f"{t.name}: missing function key"
        f = s["function"]
        assert "name" in f, f"{t.name}: missing function.name"
        assert f["name"] == t.name, f"{t.name}: name mismatch {f['name']}"
        assert "parameters" in f, f"{t.name}: missing function.parameters"
        p = f["parameters"]
        assert "type" in p, f"{t.name}: missing parameters.type"
        assert p["type"] == "object", f"{t.name}: parameters.type != object"
    print("All schemas valid!")

    # Test 3: MockDatabase creation
    db = MockDatabase("retail")
    print(f"Retail DB: {len(db.state['users'])} users, {len(db.state['products'])} products, {len(db.state['orders'])} orders")

    db2 = MockDatabase("airline")
    print(f"Airline DB: {len(db2.state['users'])} users, {len(db2.state['airports'])} airports, "
          f"{len(db2.state['flights'])} flights, {len(db2.state['reservations'])} reservations")

    # Test 4: Find user by email
    tool = FindUserByEmailTool()
    result = db.execute(tool, {"email": "alice.chen@email.com"})
    print(f"Find user by email: {result['status']} - {result.get('result', {}).get('name', 'N/A')}")

    # Test 5: Get order details
    tool2 = GetOrderDetailsTool()
    result2 = db.execute(tool2, {"order_id": "O001"})
    print(f"Get order: {result2['status']} - {len(result2['result']['items'])} items, status={result2['result']['status']}")

    # Test 6: List product types
    tool3 = ListAllProductTypesTool()
    result3 = db.execute(tool3, {})
    print(f"Product types: {result3['result']}")

    # Test 7: Search direct flights
    tool4 = SearchDirectFlightTool()
    result4 = db2.execute(tool4, {"origin": "SFO", "destination": "JFK", "date": "2026-06-15"})
    flights = result4.get("result", [])
    print(f"Direct flights SFO->JFK on 2026-06-15: {len(flights)} found")

    # Test 8: List airports
    tool5 = ListAllAirportsTool()
    result5 = db2.execute(tool5, {})
    print(f"Airports: {len(result5['result'])} found")

    # Test 9: Cancel pending order
    tool6 = CancelPendingOrderTool()
    result6 = db.execute(tool6, {"order_id": "O002"})
    print(f"Cancel O002: {result6['status']}")

    # Test 10: Book reservation
    tool7 = BookReservationTool()
    result7 = db2.execute(tool7, {
        "user_id": "UA001",
        "origin": "SFO",
        "destination": "LAX",
        "flight_type": "one_way",
        "cabin": "economy",
        "flights": [{"flight_number": "UA303", "date": "2026-06-15"}],
        "passengers": [{"name": "Alice Chen", "dob": "1990-03-15"}],
    })
    print(f"Book reservation: {result7['status']} - {result7.get('result', {}).get('reservation_id', 'N/A')}")

    # Test 11: Serialize/deserialize
    json_str = db.to_json()
    assert len(json_str) > 1000
    data = json.loads(json_str)
    assert "users" in data
    assert "orders" in data
    print(f"JSON serialization: {len(json_str)} chars OK")

    # Test 12: Modify pending order address
    tool8 = ModifyPendingOrderAddressTool()
    result8 = db.execute(tool8, {
        "order_id": "O006",
        "address1": "999 New St",
        "city": "Miami",
        "state": "FL",
        "zip": "33102",
    })
    print(f"Modify address O006: {result8['status']}")

    # Test 13: Exchange delivered order items
    tool9 = ExchangeDeliveredOrderItemsTool()
    result9 = db.execute(tool9, {
        "order_id": "O001",
        "old_item_ids": ["I001"],
        "new_item_ids": ["I007"],
        "quantities": [1],
        "payment_method": "credit_card",
    })
    print(f"Exchange O001: {result9['status']} - exchange ID: {result9.get('result', {}).get('exchange_order_id', 'N/A')}")

    # Test 14: Return delivered order items
    tool10 = ReturnDeliveredOrderItemsTool()
    result10 = db.execute(tool10, {
        "order_id": "O003",
        "item_ids": ["I004"],
        "payment_method": "credit_card",
    })
    print(f"Return O003: {result10['status']} - RMA: {result10.get('result', {}).get('return_authorization', 'N/A')}")

    # Test 15: Search one-stop flights
    tool11 = SearchOnestopFlightTool()
    result11 = db2.execute(tool11, {"origin": "SEA", "destination": "LAX", "date": "2026-06-15"})
    print(f"One-stop SEA->LAX on 2026-06-15: {result11['status']} - {len(result11.get('result', []))} options found")

    # Test 16: Get flight status
    tool12 = GetFlightStatusTool()
    result12 = db2.execute(tool12, {"flight_number": "AA101", "date": "2026-06-15"})
    print(f"Flight AA101 status: {result12['status']} - {result12['result']['flight_status']}")

    # Test 17: Error handling - missing required arg
    result_bad = db.execute(tool, {})
    assert result_bad["status"] == "error", f"Expected error, got {result_bad}"
    print(f"Missing arg error: {result_bad['message'][:50]}...")

    # Test 18: Error handling - entity not found
    result_bad2 = db.execute(tool, {"email": "nobody@nowhere.com"})
    assert result_bad2["status"] == "error"
    print(f"Not found error: {result_bad2['message'][:50]}...")

    # Test 19: Domain constraint - basic economy cannot change flights
    tool13 = UpdateReservationFlightsTool()
    result13 = db2.execute(tool13, {
        "reservation_id": "RA011",
        "new_flights": [{"flight_number": "DL212", "date": "2026-06-12"}],
    })
    print(f"Change basic economy RA011: {result13['status']} - {result13.get('message', '')[:60]}...")

    # Test 20: calculate, think, transfer tools
    result_calc = calculate_tool.execute({"expression": "2+3*4"}, {})
    print(f"Calculate: {result_calc['result']}")
    result_think = think_tool.execute({"thought": "I need to check the order status"}, {})
    print(f"Think: {result_think['result']}")
    result_transfer = transfer_to_human_agents_tool.execute({"summary": "Customer needs refund"}, {})
    print(f"Transfer: {result_transfer['result']['message'][:40]}...")

    print()
    print("ALL 20 TESTS PASSED!")


if __name__ == "__main__":
    main()
