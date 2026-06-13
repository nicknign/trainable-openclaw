"""
Tau-bench retail domain tools.

14 domain-specific tools + 3 shared utilities (calculate, think,
transfer_to_human_agents) implemented against the mock database engine.

Each tool is a ``MockTool`` subclass that reads/writes ``db_state`` — a
plain Python dict seeded by ``mock_db.seed("retail")``.
"""

from __future__ import annotations

import logging
import copy
from typing import Any

from trainable_openclaw.agent.tau_bench_tools.base import (
    MockTool,
    calculate_tool,
    think_tool,
    transfer_to_human_agents_tool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_user(db_state: dict, user_id: str) -> dict | None:
    for u in db_state["users"]:
        if u["user_id"] == user_id:
            return u
    return None


def _find_order(db_state: dict, order_id: str) -> dict | None:
    for o in db_state["orders"]:
        if o["order_id"] == order_id:
            return o
    return None


def _find_product(db_state: dict, product_id: str) -> dict | None:
    for p in db_state["products"]:
        if p["product_id"] == product_id:
            return p
    return None


def _find_item_variant(db_state: dict, item_id: str) -> dict | None:
    for p in db_state["products"]:
        for v in p.get("variants", []):
            if v["item_id"] == item_id:
                return {
                    **v,
                    "product_id": p["product_id"],
                    "product_name": p["name"],
                    "product_type": p["type"],
                }
    return None


def _build_item_name(db_state: dict, item_id: str) -> str:
    """Build a human-readable item name from item_id."""
    v = _find_item_variant(db_state, item_id)
    if v:
        return f"{v['product_name']} ({v['variant_name']})"
    return f"Item {item_id}"


def _calculate_order_total(items: list[dict]) -> float:
    return sum(it.get("unit_price", 0) * it.get("quantity", 0) for it in items)


# ---------------------------------------------------------------------------
# 0. get_user_orders
# ---------------------------------------------------------------------------

class GetUserOrdersTool(MockTool):
    name = "get_user_orders"
    description = "Get all orders for a specific user by their user ID. Returns a list of order summaries."
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user ID (e.g. 'U001').",
            },
        },
        "required": ["user_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        user = _find_user(db_state, arguments["user_id"])
        if user is None:
            return {"status": "error", "message": f"User '{arguments['user_id']}' not found"}
        orders = [
            {
                "order_id": o["order_id"],
                "status": o["status"],
                "items": [{"name": it["name"], "quantity": it["quantity"], "unit_price": it["unit_price"]} for it in o.get("items", [])],
                "created_at": o.get("created_at", ""),
                "total": o.get("payment", {}).get("amount", 0.0),
            }
            for o in db_state["orders"]
            if o["user_id"] == arguments["user_id"]
        ]
        return {"status": "success", "result": orders}


# ---------------------------------------------------------------------------
# 1. find_user_id_by_name_zip
# ---------------------------------------------------------------------------

class FindUserByNameZipTool(MockTool):
    name = "find_user_id_by_name_zip"
    description = (
        "Find user IDs by matching the user's first name, last name, and "
        "zip code. Returns a list of matching user records."
    )
    parameters = {
        "type": "object",
        "properties": {
            "first_name": {
                "type": "string",
                "description": "The user's first name (case-insensitive partial match).",
            },
            "last_name": {
                "type": "string",
                "description": "The user's last name (case-insensitive partial match).",
            },
            "zip": {
                "type": "string",
                "description": "The user's 5-digit zip code.",
            },
        },
        "required": ["first_name", "last_name", "zip"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        fn = arguments["first_name"].strip().lower()
        ln = arguments["last_name"].strip().lower()
        zp = arguments["zip"].strip()
        matches = []
        for u in db_state["users"]:
            if (
                fn in u["first_name"].lower()
                and ln in u["last_name"].lower()
                and u.get("address", {}).get("zip") == zp
            ):
                matches.append({
                    "user_id": u["user_id"],
                    "name": u["name"],
                    "email": u["email"],
                    "address": u.get("address"),
                })
        if not matches:
            return {"status": "error", "message": f"No user found for {arguments['first_name']} {arguments['last_name']}, zip {zp}"}
        return {"status": "success", "result": matches}


# ---------------------------------------------------------------------------
# 2. find_user_id_by_email
# ---------------------------------------------------------------------------

class FindUserByEmailTool(MockTool):
    name = "find_user_id_by_email"
    description = "Find a user by their email address. Returns the user record if found."
    parameters = {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The user's email address (exact match).",
            },
        },
        "required": ["email"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        email = arguments["email"].strip().lower()
        for u in db_state["users"]:
            if u["email"].lower() == email:
                return {
                    "status": "success",
                    "result": {
                        "user_id": u["user_id"],
                        "name": u["name"],
                        "email": u["email"],
                        "address": u.get("address"),
                    },
                }
        return {"status": "error", "message": f"No user found with email '{arguments['email']}'"}


# ---------------------------------------------------------------------------
# 3. get_user_details
# ---------------------------------------------------------------------------

class GetUserDetailsTool(MockTool):
    name = "get_user_details"
    description = "Get full user profile including address and payment methods."
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user ID (e.g. 'U001').",
            },
        },
        "required": ["user_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        user = _find_user(db_state, arguments["user_id"])
        if user is None:
            return {"status": "error", "message": f"User '{arguments['user_id']}' not found"}
        # Return a copy to prevent accidental mutation
        safe = {k: v for k, v in user.items()}
        # mask sensitive payment details
        if "payment_methods" in safe:
            safe["payment_methods"] = [
                {
                    "type": pm["type"],
                    "last_four": pm.get("last_four", "N/A"),
                    "brand": pm.get("brand", "N/A"),
                }
                for pm in safe["payment_methods"]
            ]
        return {"status": "success", "result": safe}


# ---------------------------------------------------------------------------
# 4. get_order_details
# ---------------------------------------------------------------------------

class GetOrderDetailsTool(MockTool):
    name = "get_order_details"
    description = "Get full order details including items, shipping address, payment, and status."
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID (e.g. 'O001').",
            },
        },
        "required": ["order_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order = _find_order(db_state, arguments["order_id"])
        if order is None:
            return {"status": "error", "message": f"Order '{arguments['order_id']}' not found"}
        return {"status": "success", "result": dict(order)}


# ---------------------------------------------------------------------------
# 5. get_product_details
# ---------------------------------------------------------------------------

class GetProductDetailsTool(MockTool):
    name = "get_product_details"
    description = "Get product details including description, price range, and available variants."
    parameters = {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "The product ID (e.g. 'P001').",
            },
        },
        "required": ["product_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        product = _find_product(db_state, arguments["product_id"])
        if product is None:
            return {"status": "error", "message": f"Product '{arguments['product_id']}' not found"}
        # Return product info with variant summary
        result = dict(product)
        result["variant_summary"] = [
            {"item_id": v["item_id"], "variant_name": v["variant_name"], "price": v["price"], "available": v["available"]}
            for v in product.get("variants", [])
        ]
        return {"status": "success", "result": result}


# ---------------------------------------------------------------------------
# 6. get_item_details
# ---------------------------------------------------------------------------

class GetItemDetailsTool(MockTool):
    name = "get_item_details"
    description = "Get item variant details including price, availability, and parent product info."
    parameters = {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The item variant ID (e.g. 'I001').",
            },
        },
        "required": ["item_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        item = _find_item_variant(db_state, arguments["item_id"])
        if item is None:
            return {"status": "error", "message": f"Item '{arguments['item_id']}' not found"}
        return {"status": "success", "result": item}


# ---------------------------------------------------------------------------
# 7. list_all_product_types
# ---------------------------------------------------------------------------

class ListAllProductTypesTool(MockTool):
    name = "list_all_product_types"
    description = "Return a list of all available product categories/types."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        types = sorted({p["type"] for p in db_state["products"]})
        return {"status": "success", "result": types}


# ---------------------------------------------------------------------------
# 8. modify_pending_order_address
# ---------------------------------------------------------------------------

class ModifyPendingOrderAddressTool(MockTool):
    name = "modify_pending_order_address"
    description = (
        "Update the shipping address for a pending order. "
        "Only works on orders with 'pending' status."
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID (e.g. 'O001')."},
            "address1": {"type": "string", "description": "Primary street address."},
            "address2": {"type": "string", "description": "Apartment, suite, unit, etc. (optional)."},
            "city": {"type": "string", "description": "City name."},
            "state": {"type": "string", "description": "Two-letter state/province code."},
            "zip": {"type": "string", "description": "Postal code."},
            "country": {"type": "string", "description": "Country name (default 'USA')."},
        },
        "required": ["order_id", "address1", "city", "state", "zip"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order = _find_order(db_state, arguments["order_id"])
        if order is None:
            return {"status": "error", "message": f"Order '{arguments['order_id']}' not found"}
        if order["status"] != "pending":
            return {"status": "error", "message": f"Order '{arguments['order_id']}' is '{order['status']}', not 'pending'"}

        order["shipping_address"] = {
            "address1": arguments["address1"],
            "address2": arguments.get("address2", ""),
            "city": arguments["city"],
            "state": arguments["state"],
            "zip": arguments["zip"],
            "country": arguments.get("country", "USA"),
        }
        return {
            "status": "success",
            "result": {"order_id": order["order_id"], "shipping_address": order["shipping_address"]},
        }


# ---------------------------------------------------------------------------
# 9. modify_pending_order_items
# ---------------------------------------------------------------------------

class ModifyPendingOrderItemsTool(MockTool):
    name = "modify_pending_order_items"
    description = (
        "Replace all items in a pending order with the specified items. "
        "Only works on orders with 'pending' status. "
        "Provide parallel lists of item_ids and quantities."
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID (e.g. 'O001')."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of item variant IDs to include in the order.",
            },
            "quantities": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Quantities for each item (must match length of item_ids).",
            },
        },
        "required": ["order_id", "item_ids", "quantities"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order_id = arguments["order_id"]
        item_ids = arguments["item_ids"]
        quantities = arguments["quantities"]

        order = _find_order(db_state, order_id)
        if order is None:
            return {"status": "error", "message": f"Order '{order_id}' not found"}
        if order["status"] != "pending":
            return {"status": "error", "message": f"Order '{order_id}' is '{order['status']}', not 'pending'"}
        if len(item_ids) != len(quantities):
            return {"status": "error", "message": "item_ids and quantities must have the same length"}

        new_items = []
        for iid, qty in zip(item_ids, quantities):
            variant = _find_item_variant(db_state, iid)
            if variant is None:
                return {"status": "error", "message": f"Item '{iid}' not found"}
            if qty < 0:
                return {"status": "error", "message": f"Quantity for '{iid}' cannot be negative"}
            # Filter out zero-quantity items
            if qty == 0:
                continue
            new_items.append({
                "item_id": iid,
                "product_id": variant["product_id"],
                "name": _build_item_name(db_state, iid),
                "quantity": qty,
                "unit_price": variant["price"],
            })

        order["items"] = new_items
        # Update payment amount
        new_total = _calculate_order_total(new_items)
        if "payment" in order:
            order["payment"]["amount"] = round(new_total + 10.0, 2)  # rough tax/shipping estimate

        return {
            "status": "success",
            "result": {"order_id": order_id, "items": order["items"]},
        }


# ---------------------------------------------------------------------------
# 10. modify_pending_order_payment
# ---------------------------------------------------------------------------

class ModifyPendingOrderPaymentTool(MockTool):
    name = "modify_pending_order_payment"
    description = (
        "Change the payment method for a pending order. "
        "Only works on orders with 'pending' status."
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID."},
            "payment_type": {
                "type": "string",
                "enum": ["credit_card", "gift_card", "paypal"],
                "description": "Payment method type.",
            },
            "payment_details": {
                "type": "object",
                "description": "Payment-specific details (e.g. {'last_four': '1234', 'brand': 'Visa'}). Optional.",
            },
        },
        "required": ["order_id", "payment_type"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order = _find_order(db_state, arguments["order_id"])
        if order is None:
            return {"status": "error", "message": f"Order '{arguments['order_id']}' not found"}
        if order["status"] != "pending":
            return {"status": "error", "message": f"Order '{arguments['order_id']}' is '{order['status']}', not 'pending'"}

        ptype = arguments["payment_type"]
        details = arguments.get("payment_details", {})
        order["payment"] = {
            "method": ptype,
            "amount": order.get("payment", {}).get("amount", 0.0),
        }
        order["payment"].update(details)
        return {
            "status": "success",
            "result": {"order_id": order["order_id"], "payment": order["payment"]},
        }


# ---------------------------------------------------------------------------
# 11. modify_user_address
# ---------------------------------------------------------------------------

class ModifyUserAddressTool(MockTool):
    name = "modify_user_address"
    description = "Update the default address for a user account."
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "The user ID."},
            "address1": {"type": "string", "description": "Primary street address."},
            "address2": {"type": "string", "description": "Apartment, suite, unit, etc. (optional)."},
            "city": {"type": "string", "description": "City name."},
            "state": {"type": "string", "description": "Two-letter state/province code."},
            "zip": {"type": "string", "description": "Postal code."},
            "country": {"type": "string", "description": "Country name (default 'USA')."},
        },
        "required": ["user_id", "address1", "city", "state", "zip"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        user = _find_user(db_state, arguments["user_id"])
        if user is None:
            return {"status": "error", "message": f"User '{arguments['user_id']}' not found"}

        user["address"] = {
            "address1": arguments["address1"],
            "address2": arguments.get("address2", ""),
            "city": arguments["city"],
            "state": arguments["state"],
            "zip": arguments["zip"],
            "country": arguments.get("country", "USA"),
        }
        return {
            "status": "success",
            "result": {"user_id": user["user_id"], "address": user["address"]},
        }


# ---------------------------------------------------------------------------
# 12. cancel_pending_order
# ---------------------------------------------------------------------------

class CancelPendingOrderTool(MockTool):
    name = "cancel_pending_order"
    description = (
        "Cancel a pending order. Only works on orders with 'pending' status. "
        "Cancelled orders cannot be reactivated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID."},
        },
        "required": ["order_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order = _find_order(db_state, arguments["order_id"])
        if order is None:
            return {"status": "error", "message": f"Order '{arguments['order_id']}' not found"}
        if order["status"] != "pending":
            return {"status": "error", "message": f"Order '{arguments['order_id']}' is '{order['status']}', not 'pending'"}

        from datetime import datetime
        order["status"] = "cancelled"
        order["cancelled_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "status": "success",
            "result": {"order_id": order["order_id"], "status": "cancelled"},
        }


# ---------------------------------------------------------------------------
# 13. exchange_delivered_order_items
# ---------------------------------------------------------------------------

class ExchangeDeliveredOrderItemsTool(MockTool):
    name = "exchange_delivered_order_items"
    description = (
        "Exchange items from a delivered order for new items. "
        "Only works on orders with 'delivered' status. "
        "Returns a new exchange order if successful."
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The original order ID."},
            "old_item_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Item IDs to exchange (must be present in the order).",
            },
            "new_item_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "New item IDs to receive in exchange.",
            },
            "quantities": {
                "type": "array", "items": {"type": "integer"},
                "description": "Quantities for each new item.",
            },
            "payment_method": {
                "type": "string",
                "description": "Payment method for any price difference (e.g. 'credit_card').",
            },
        },
        "required": ["order_id", "old_item_ids", "new_item_ids", "quantities", "payment_method"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order = _find_order(db_state, arguments["order_id"])
        if order is None:
            return {"status": "error", "message": f"Order '{arguments['order_id']}' not found"}
        if order["status"] != "delivered":
            return {"status": "error", "message": f"Order '{arguments['order_id']}' is '{order['status']}', not 'delivered'"}

        old_ids = set(arguments["old_item_ids"])
        new_ids = arguments["new_item_ids"]
        qties = arguments["quantities"]
        if len(new_ids) != len(qties):
            return {"status": "error", "message": "new_item_ids and quantities must have the same length"}

        # Verify old items exist in the order
        order_item_ids = {it["item_id"] for it in order["items"]}
        missing = old_ids - order_item_ids
        if missing:
            return {"status": "error", "message": f"Items not in order: {missing}"}

        # Build new items
        new_items = []
        new_total = 0.0
        for iid, qty in zip(new_ids, qties):
            variant = _find_item_variant(db_state, iid)
            if variant is None:
                return {"status": "error", "message": f"Item '{iid}' not found"}
            if qty <= 0:
                return {"status": "error", "message": f"Quantity for '{iid}' must be positive"}
            new_items.append({
                "item_id": iid,
                "product_id": variant["product_id"],
                "name": _build_item_name(db_state, iid),
                "quantity": qty,
                "unit_price": variant["price"],
            })
            new_total += variant["price"] * qty

        # Calculate price difference
        old_total = sum(
            it["unit_price"] * it["quantity"]
            for it in order["items"]
            if it["item_id"] in old_ids
        )
        price_diff = round(new_total - old_total, 2)

        # Create exchange order
        from datetime import datetime
        exchange_order_id = f"{arguments['order_id']}-EX"
        db_state["orders"].append({
            "order_id": exchange_order_id,
            "user_id": order["user_id"],
            "status": "pending",
            "items": new_items,
            "shipping_address": dict(order["shipping_address"]),
            "payment": {
                "method": arguments["payment_method"],
                "amount": round(max(price_diff, 0) + 5.0, 2),  # price diff + shipping
            },
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "exchange_for": arguments["order_id"],
            "price_difference": price_diff,
        })

        return {
            "status": "success",
            "result": {
                "exchange_order_id": exchange_order_id,
                "original_order_id": arguments["order_id"],
                "items_exchanged": list(old_ids),
                "new_items": new_items,
                "price_difference": price_diff,
            },
        }


# ---------------------------------------------------------------------------
# 14. return_delivered_order_items
# ---------------------------------------------------------------------------

class ReturnDeliveredOrderItemsTool(MockTool):
    name = "return_delivered_order_items"
    description = (
        "Return items from a delivered order. Only works on orders with "
        "'delivered' status. Returns a return authorization and estimated refund."
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The original order ID."},
            "item_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Item IDs to return (must be present in the order).",
            },
            "payment_method": {
                "type": "string",
                "description": "Payment method for the refund (e.g. 'credit_card').",
            },
        },
        "required": ["order_id", "item_ids", "payment_method"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        order = _find_order(db_state, arguments["order_id"])
        if order is None:
            return {"status": "error", "message": f"Order '{arguments['order_id']}' not found"}
        if order["status"] != "delivered":
            return {"status": "error", "message": f"Order '{arguments['order_id']}' is '{order['status']}', not 'delivered'"}

        return_ids = set(arguments["item_ids"])
        order_item_ids = {it["item_id"] for it in order["items"]}
        missing = return_ids - order_item_ids
        if missing:
            return {"status": "error", "message": f"Items not in order: {missing}"}

        # Calculate refund
        refund = 0.0
        returned_items = []
        for it in order["items"]:
            if it["item_id"] in return_ids:
                refund += it["unit_price"] * it["quantity"]
                returned_items.append(it)

        from datetime import datetime
        return_auth = f"RMA-{arguments['order_id']}-{datetime.utcnow().strftime('%Y%m%d')}"

        return {
            "status": "success",
            "result": {
                "return_authorization": return_auth,
                "order_id": arguments["order_id"],
                "returned_items": returned_items,
                "estimated_refund": round(refund, 2),
                "refund_method": arguments["payment_method"],
                "instructions": "Print the return label from your account. Drop off at any authorized location within 30 days.",
            },
        }


# ---------------------------------------------------------------------------
# Tool registry for the retail domain
# ---------------------------------------------------------------------------

def _make_retail_tools() -> list[MockTool]:
    """Instantiate all 15 retail-domain tools + 3 shared utilities."""
    return [
        GetUserOrdersTool(),
        FindUserByNameZipTool(),
        FindUserByEmailTool(),
        GetUserDetailsTool(),
        GetOrderDetailsTool(),
        GetProductDetailsTool(),
        GetItemDetailsTool(),
        ListAllProductTypesTool(),
        ModifyPendingOrderAddressTool(),
        ModifyPendingOrderItemsTool(),
        ModifyPendingOrderPaymentTool(),
        ModifyUserAddressTool(),
        CancelPendingOrderTool(),
        ExchangeDeliveredOrderItemsTool(),
        ReturnDeliveredOrderItemsTool(),
        calculate_tool,
        think_tool,
        transfer_to_human_agents_tool,
    ]


# Module-level export for convenience
retail_tools: list[MockTool] = _make_retail_tools()

# Re-export individual tool classes for direct import
__all__ = [
    "GetUserOrdersTool",
    "FindUserByNameZipTool",
    "FindUserByEmailTool",
    "GetUserDetailsTool",
    "GetOrderDetailsTool",
    "GetProductDetailsTool",
    "GetItemDetailsTool",
    "ListAllProductTypesTool",
    "ModifyPendingOrderAddressTool",
    "ModifyPendingOrderItemsTool",
    "ModifyPendingOrderPaymentTool",
    "ModifyUserAddressTool",
    "CancelPendingOrderTool",
    "ExchangeDeliveredOrderItemsTool",
    "ReturnDeliveredOrderItemsTool",
    "retail_tools",
]
