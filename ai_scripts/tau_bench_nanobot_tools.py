"""Tau-bench tools as nanobot Tool subclasses.

Place this file in nanobot/agent/tools/ for auto-discovery by nanobot's
ToolLoader.  All retail + airline domain tools are wrapped, plus shared
utilities (calculate, think, transfer_to_human_agents).

The mock database is seeded on first use (module-level singleton, one per
scenario).  Each tool knows which scenario it belongs to.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# nanobot tools dir is inside the project at
#   {project_root}/nanobot-0.2.1/nanobot/agent/tools/
# So parents[4] from this file is the project root.
_project_root = Path(__file__).resolve().parents[4]
# Fallback: the known absolute path on the remote server
if not _project_root.is_dir() or not (_project_root / "trainable_openclaw").is_dir():
    _project_root = Path("/data/wangye/trainable-openclaw")
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase, seed  # noqa: E402

from nanobot.agent.tools.base import Tool, tool_parameters  # noqa: E402
from nanobot.agent.tools.schema import (  # noqa: E402
    ArraySchema,
    IntegerSchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

# ── shared mock DB ─────────────────────────────────────────────────────────
_dbs: dict[str, MockDatabase] = {}


def _db(scenario: str) -> MockDatabase:
    if scenario not in _dbs:
        _dbs[scenario] = MockDatabase(scenario)
    return _dbs[scenario]


# ── helper: run sync tool against db ───────────────────────────────────────
def _run(tool_name: str, scenario: str, **kwargs):
    """Execute a single tau-bench tool by name and return result dict."""
    from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools

    db = _db(scenario)
    state = db.state
    for t in register_tau_bench_tools(scenario):
        if t.name == tool_name:
            try:
                return t.execute(kwargs, state)
            except Exception as exc:
                return {"status": "error", "message": str(exc)}
    return {"status": "error", "message": f"Tool {tool_name} not found in {scenario}"}


# ── shared utility tools ───────────────────────────────────────────────────

class CalculateTool(Tool):
    name = "calculate"
    description = (
        "Evaluate a mathematical expression. "
        "Supports +, -, *, /, parentheses, and decimal numbers."
    )
    parameters = tool_parameters_schema(
        expression=StringSchema(description="Mathematical expression to evaluate, e.g. '2+3*4'."),
        required=["expression"],
    )

    async def execute(self, **kwargs) -> str:
        expr = kwargs.get("expression", "").strip()
        if not re.fullmatch(r"[\d+\-*/().%\s]+", expr):
            return _error("Invalid expression")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return _ok({"result": str(result)})
        except Exception as exc:
            return _error(f"Calculation error: {exc}")


class ThinkTool(Tool):
    name = "think"
    description = "Use this tool to think through a problem step by step. The tool does nothing — it simply records your thought process."
    parameters = tool_parameters_schema(
        thought=StringSchema(description="Your internal reasoning or thought process."),
        required=["thought"],
    )

    async def execute(self, **kwargs) -> str:
        return _ok({"result": "Thought recorded."})


class TransferToHumanTool(Tool):
    name = "transfer_to_human_agents"
    description = (
        "Transfer the conversation to a human agent. Use this when you cannot "
        "resolve the issue or when the user explicitly requests a human. "
        "Provide a summary of the situation."
    )
    parameters = tool_parameters_schema(
        summary=StringSchema(description="Summary of the issue, actions taken, and what the human agent needs to do."),
        required=["summary"],
    )

    async def execute(self, **kwargs) -> str:
        return _ok({"message": f"Transferred to human agent. Summary: {kwargs.get('summary', '')}"})


# ── retail tools ───────────────────────────────────────────────────────────

class GetUserOrdersTool(Tool):
    name = "get_user_orders"
    description = "Get all orders for a specific user by their user ID. Returns a list of order summaries with status and dates."
    parameters = tool_parameters_schema(
        user_id=StringSchema(description="The user ID to look up orders for."),
        required=["user_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_user_orders", "retail", **kwargs))


class FindUserIdByNameZipTool(Tool):
    name = "find_user_id_by_name_zip"
    description = "Find user IDs by matching the user's first name, last name, and zip code. Returns matching user records."
    parameters = tool_parameters_schema(
        first_name=StringSchema(description="User's first name."),
        last_name=StringSchema(description="User's last name."),
        zip_code=StringSchema(description="User's zip code."),
        required=["first_name", "last_name", "zip_code"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("find_user_id_by_name_zip", "retail", **kwargs))


class FindUserIdByEmailTool(Tool):
    name = "find_user_id_by_email"
    description = "Find a user by their email address. Returns the user record if found."
    parameters = tool_parameters_schema(
        email=StringSchema(description="The user's email address."),
        required=["email"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("find_user_id_by_email", "retail", **kwargs))


class GetUserDetailsTool(Tool):
    name = "get_user_details"
    description = "Get full user profile."
    parameters = tool_parameters_schema(
        user_id=StringSchema(description="The user ID to look up."),
        required=["user_id"],
    )
    async def execute(self, **kwargs) -> str:
        uid = kwargs.get("user_id", "")
        # Retail users: U001-U015; Airline users: UA001-UA015
        scenario = "airline" if uid.startswith("UA") else "retail"
        return _json(_run("get_user_details", scenario, **kwargs))


class GetOrderDetailsTool(Tool):
    name = "get_order_details"
    description = "Get full order details including items, shipping address, payment, and status."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to look up."),
        required=["order_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_order_details", "retail", **kwargs))


class GetProductDetailsTool(Tool):
    name = "get_product_details"
    description = "Get product details including description, price range, and available variants."
    parameters = tool_parameters_schema(
        product_id=StringSchema(description="The product ID to look up."),
        required=["product_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_product_details", "retail", **kwargs))


class GetItemDetailsTool(Tool):
    name = "get_item_details"
    description = "Get item variant details including price, availability, and parent product info."
    parameters = tool_parameters_schema(
        item_id=StringSchema(description="The item variant ID to look up."),
        required=["item_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_item_details", "retail", **kwargs))


class ListAllProductTypesTool(Tool):
    name = "list_all_product_types"
    description = "Return a list of all available product categories/types."
    parameters = tool_parameters_schema(required=[], description="No parameters required.")

    async def execute(self, **kwargs) -> str:
        return _json(_run("list_all_product_types", "retail"))


class ModifyPendingOrderAddressTool(Tool):
    name = "modify_pending_order_address"
    description = "Update the shipping address for a pending order. Only works on orders with 'pending' status."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to modify."),
        address1=StringSchema(description="New shipping address line 1."),
        address2=StringSchema(description="New shipping address line 2 (optional)."),
        city=StringSchema(description="New shipping city."),
        state=StringSchema(description="New shipping state (2-letter code)."),
        zip_code=StringSchema(description="New shipping zip code."),
        required=["order_id", "address1", "city", "state", "zip_code"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("modify_pending_order_address", "retail", **kwargs))


class ModifyPendingOrderItemsTool(Tool):
    name = "modify_pending_order_items"
    description = "Replace all items in a pending order with the specified items. Only works on orders with 'pending' status."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to modify."),
        items=ArraySchema(
            items=ObjectSchema(
                item_id=StringSchema(description="The item variant ID to add."),
                quantity=IntegerSchema(description="Quantity of this item."),
                required=["item_id", "quantity"],
            ),
            description="List of item objects with item_id and quantity.",
        ),
        required=["order_id", "items"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("modify_pending_order_items", "retail", **kwargs))


class ModifyPendingOrderPaymentTool(Tool):
    name = "modify_pending_order_payment"
    description = "Change the payment method for a pending order. Only works on orders with 'pending' status."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to modify."),
        payment_method=StringSchema(description="New payment method type (e.g., 'credit_card', 'paypal', 'gift_card')."),
        required=["order_id", "payment_method"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("modify_pending_order_payment", "retail", **kwargs))


class ModifyUserAddressTool(Tool):
    name = "modify_user_address"
    description = "Update the default address for a user account."
    parameters = tool_parameters_schema(
        user_id=StringSchema(description="The user ID to modify."),
        address1=StringSchema(description="New address line 1."),
        address2=StringSchema(description="New address line 2 (optional)."),
        city=StringSchema(description="New city."),
        state=StringSchema(description="New state (2-letter code)."),
        zip_code=StringSchema(description="New zip code."),
        required=["user_id", "address1", "city", "state", "zip_code"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("modify_user_address", "retail", **kwargs))


class CancelPendingOrderTool(Tool):
    name = "cancel_pending_order"
    description = "Cancel a pending order. Only works on orders with 'pending' status. Cancelled or delivered orders cannot be cancelled."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to cancel."),
        required=["order_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("cancel_pending_order", "retail", **kwargs))


class ExchangeDeliveredOrderItemsTool(Tool):
    name = "exchange_delivered_order_items"
    description = "Exchange items from a delivered order for new items. Only works on orders with 'delivered' status."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to exchange items from."),
        old_item_ids=ArraySchema(
            items=StringSchema(description="Item ID to exchange."),
            description="List of item IDs to exchange.",
        ),
        new_item_ids=ArraySchema(
            items=StringSchema(description="New item ID to receive."),
            description="List of new item IDs to receive.",
        ),
        required=["order_id", "old_item_ids", "new_item_ids"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("exchange_delivered_order_items", "retail", **kwargs))


class ReturnDeliveredOrderItemsTool(Tool):
    name = "return_delivered_order_items"
    description = "Return items from a delivered order. Only works on orders with 'delivered' status."
    parameters = tool_parameters_schema(
        order_id=StringSchema(description="The order ID to return items from."),
        item_ids=ArraySchema(
            items=StringSchema(description="Item ID to return."),
            description="List of item IDs to return.",
        ),
        required=["order_id", "item_ids"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("return_delivered_order_items", "retail", **kwargs))


# ── airline tools ──────────────────────────────────────────────────────────

class GetUserReservationsTool(Tool):
    name = "get_user_reservations"
    description = "Get all reservations for a specific user by their user ID. Returns a list of reservation summaries with status and dates."
    parameters = tool_parameters_schema(
        user_id=StringSchema(description="The user ID to look up reservations for."),
        required=["user_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_user_reservations", "airline", **kwargs))


class BookReservationTool(Tool):
    name = "book_reservation"
    description = "Book a flight reservation. Searches for available flights and creates a reservation."
    parameters = tool_parameters_schema(
        user_id=StringSchema(description="The user ID making the reservation."),
        origin=StringSchema(description="Origin airport code (e.g., 'SFO')."),
        destination=StringSchema(description="Destination airport code (e.g., 'JFK')."),
        date=StringSchema(description="Flight date in YYYY-MM-DD format."),
        flight_number=StringSchema(description="The flight number to book."),
        cabin=StringSchema(description="Cabin class: 'business', 'economy', or 'basic_economy'."),
        passengers=ArraySchema(
            items=ObjectSchema(
                name=StringSchema(description="Passenger full name."),
                dob=StringSchema(description="Passenger date of birth (YYYY-MM-DD)."),
                required=["name", "dob"],
            ),
            description="List of passenger objects.",
        ),
        bags=IntegerSchema(description="Number of checked bags."),
        required=["user_id", "origin", "destination", "date", "flight_number", "cabin", "passengers"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("book_reservation", "airline", **kwargs))


class CancelReservationTool(Tool):
    name = "cancel_reservation"
    description = "Cancel an existing reservation. Cannot cancel already cancelled reservations."
    parameters = tool_parameters_schema(
        reservation_id=StringSchema(description="The reservation ID to cancel."),
        required=["reservation_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("cancel_reservation", "airline", **kwargs))


class GetReservationDetailsTool(Tool):
    name = "get_reservation_details"
    description = "Get full reservation details including flights, passengers, bags, and payment."
    parameters = tool_parameters_schema(
        reservation_id=StringSchema(description="The reservation ID to look up."),
        required=["reservation_id"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_reservation_details", "airline", **kwargs))


class ListAllAirportsTool(Tool):
    name = "list_all_airports"
    description = "Return a list of all airports with codes and names."
    parameters = tool_parameters_schema(required=[], description="No parameters required.")
    async def execute(self, **kwargs) -> str:
        return _json(_run("list_all_airports", "airline"))


class SearchDirectFlightTool(Tool):
    name = "search_direct_flight"
    description = "Search for direct flights between two airports on a given date. Returns available flights with pricing and seat availability."
    parameters = tool_parameters_schema(
        origin=StringSchema(description="Origin airport code (e.g., 'SFO')."),
        destination=StringSchema(description="Destination airport code (e.g., 'JFK')."),
        date=StringSchema(description="Flight date in YYYY-MM-DD format."),
        required=["origin", "destination", "date"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("search_direct_flight", "airline", **kwargs))


class SearchOnestopFlightTool(Tool):
    name = "search_onestop_flight"
    description = "Search for flights with exactly one stop/connection between two airports on a given date."
    parameters = tool_parameters_schema(
        origin=StringSchema(description="Origin airport code (e.g., 'SFO')."),
        destination=StringSchema(description="Destination airport code (e.g., 'JFK')."),
        date=StringSchema(description="Flight date in YYYY-MM-DD format."),
        required=["origin", "destination", "date"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("search_onestop_flight", "airline", **kwargs))


class SendCertificateTool(Tool):
    name = "send_certificate"
    description = "Send a travel certificate or voucher to a user by email."
    parameters = tool_parameters_schema(
        user_id=StringSchema(description="The user ID to send certificate to."),
        amount=IntegerSchema(description="Certificate amount in dollars."),
        required=["user_id", "amount"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("send_certificate", "airline", **kwargs))


class UpdateReservationBaggagesTool(Tool):
    name = "update_reservation_baggages"
    description = "Update the baggage allowance for a reservation."
    parameters = tool_parameters_schema(
        reservation_id=StringSchema(description="The reservation ID to update."),
        total_bags=IntegerSchema(description="New total number of checked bags."),
        nonfree_bags=IntegerSchema(description="New number of non-free checked bags."),
        required=["reservation_id", "total_bags", "nonfree_bags"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("update_reservation_baggages", "airline", **kwargs))


class UpdateReservationFlightsTool(Tool):
    name = "update_reservation_flights"
    description = "Change the flights on an existing reservation. Basic economy fares cannot be changed."
    parameters = tool_parameters_schema(
        reservation_id=StringSchema(description="The reservation ID to update."),
        flights=ArraySchema(
            items=ObjectSchema(
                flight_number=StringSchema(description="New flight number."),
                date=StringSchema(description="New flight date (YYYY-MM-DD)."),
                cabin=StringSchema(description="New cabin class."),
                required=["flight_number", "date", "cabin"],
            ),
            description="List of new flights to replace existing ones.",
        ),
        required=["reservation_id", "flights"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("update_reservation_flights", "airline", **kwargs))


class UpdateReservationPassengersTool(Tool):
    name = "update_reservation_passengers"
    description = "Update the passenger information on a reservation."
    parameters = tool_parameters_schema(
        reservation_id=StringSchema(description="The reservation ID to update."),
        passengers=ArraySchema(
            items=ObjectSchema(
                name=StringSchema(description="Passenger full name."),
                dob=StringSchema(description="Passenger date of birth (YYYY-MM-DD)."),
                required=["name", "dob"],
            ),
            description="List of updated passenger objects.",
        ),
        required=["reservation_id", "passengers"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("update_reservation_passengers", "airline", **kwargs))


class GetFlightStatusTool(Tool):
    name = "get_flight_status"
    description = "Get the current status of a specific flight on a given date."
    parameters = tool_parameters_schema(
        flight_number=StringSchema(description="The flight number to check."),
        date=StringSchema(description="Date in YYYY-MM-DD format."),
        required=["flight_number", "date"],
    )
    async def execute(self, **kwargs) -> str:
        return _json(_run("get_flight_status", "airline", **kwargs))


# ── result helpers ─────────────────────────────────────────────────────────

def _ok(data: dict) -> str:
    import json as _json_mod
    return _json_mod.dumps({"status": "success", **data}, ensure_ascii=False)


def _error(msg: str) -> str:
    import json as _json_mod
    return _json_mod.dumps({"status": "error", "message": msg}, ensure_ascii=False)


def _json(data: dict) -> str:
    import json as _json_mod
    return _json_mod.dumps(data, ensure_ascii=False, default=str)
