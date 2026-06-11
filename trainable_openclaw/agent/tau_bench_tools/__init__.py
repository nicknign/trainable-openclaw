"""
Tau-bench mock tools for nanobot agent training.

Provides 28+ mock tools across two domains (retail and airline) backed by
an in-memory thread-safe database engine.

Usage::

    from trainable_openclaw.agent.tau_bench_tools import (
        MockTool, MockDatabase, register_tau_bench_tools,
    )

    db = MockDatabase("retail")
    tools = register_tau_bench_tools("retail")

    # Execute a tool
    result = db.execute(tools[0], {"first_name": "Alice", "last_name": "Chen", "zip": "94102"})

    # Get nanobot-compatible schemas
    schemas = [t.to_schema() for t in tools]
"""

from trainable_openclaw.agent.tau_bench_tools.base import MockTool
from trainable_openclaw.agent.tau_bench_tools.mock_db import MockDatabase, seed
from trainable_openclaw.agent.tau_bench_tools.registry import register_tau_bench_tools

from trainable_openclaw.agent.tau_bench_tools.retail import (  # noqa: F401 — explicit re-exports
    FindUserByNameZipTool,
    FindUserByEmailTool,
    GetUserDetailsTool as RetailGetUserDetailsTool,
    GetOrderDetailsTool,
    GetProductDetailsTool,
    GetItemDetailsTool,
    ListAllProductTypesTool,
    ModifyPendingOrderAddressTool,
    ModifyPendingOrderItemsTool,
    ModifyPendingOrderPaymentTool,
    ModifyUserAddressTool,
    CancelPendingOrderTool,
    ExchangeDeliveredOrderItemsTool,
    ReturnDeliveredOrderItemsTool,
    retail_tools,
)

from trainable_openclaw.agent.tau_bench_tools.airline import (  # noqa: F401 — explicit re-exports
    BookReservationTool,
    CancelReservationTool,
    GetReservationDetailsTool,
    AirlineGetUserDetailsTool,
    ListAllAirportsTool,
    SearchDirectFlightTool,
    SearchOnestopFlightTool,
    SendCertificateTool,
    UpdateReservationBaggagesTool,
    UpdateReservationFlightsTool,
    UpdateReservationPassengersTool,
    GetFlightStatusTool,
    airline_tools,
)

__all__ = [
    # Base
    "MockTool",
    # Database
    "MockDatabase",
    "seed",
    # Registry
    "register_tau_bench_tools",
    # Retail tools
    "FindUserByNameZipTool",
    "FindUserByEmailTool",
    "RetailGetUserDetailsTool",
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
    # Airline tools
    "BookReservationTool",
    "CancelReservationTool",
    "GetReservationDetailsTool",
    "AirlineGetUserDetailsTool",
    "ListAllAirportsTool",
    "SearchDirectFlightTool",
    "SearchOnestopFlightTool",
    "SendCertificateTool",
    "UpdateReservationBaggagesTool",
    "UpdateReservationFlightsTool",
    "UpdateReservationPassengersTool",
    "GetFlightStatusTool",
    "airline_tools",
]
