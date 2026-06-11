"""
Tau-bench airline domain tools.

12 domain-specific tools + 3 shared utilities (calculate, think,
transfer_to_human_agents) implemented against the mock database engine.

Each tool is a ``MockTool`` subclass that reads/writes ``db_state`` — a
plain Python dict seeded by ``mock_db.seed("airline")``.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
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

def _find_airline_user(db_state: dict, user_id: str) -> dict | None:
    for u in db_state["users"]:
        if u["user_id"] == user_id:
            return u
    return None


def _find_reservation(db_state: dict, reservation_id: str) -> dict | None:
    for r in db_state["reservations"]:
        if r["reservation_id"] == reservation_id:
            return r
    return None


def _find_flight(db_state: dict, flight_number: str) -> dict | None:
    return db_state.get("flights", {}).get(flight_number)


def _parse_time(t: str) -> tuple[int, int, int]:
    """Parse a time string like '08:00', '16:30', '07:15+1' → (hour, minute, day_offset)."""
    day_off = 0
    if "+" in t:
        t, off = t.split("+", 1)
        day_off = int(off)
    parts = t.split(":")
    return int(parts[0]), int(parts[1]), day_off


def _connection_ok(arrival_time: str, departure_time: str, same_date: bool = True) -> bool:
    """Check if a connection is feasible (1–8 hours layover)."""
    ah, am, ado = _parse_time(arrival_time)
    dh, dm, ddo = _parse_time(departure_time)
    # If same_date, adjust day offsets
    if same_date and ddo == 0 and ah > dh:
        ddo = 1  # departure next day relative to arrival
    arr_min = ah * 60 + am + ado * 24 * 60
    dep_min = dh * 60 + dm + ddo * 24 * 60
    gap = dep_min - arr_min
    return 60 <= gap <= 480  # 1-8 hours layover


def _get_available_cabins(flight: dict, date: str) -> list[str]:
    """Return list of cabin classes with >0 seats on the given date."""
    seats = flight.get("available_seats", {}).get(date, {})
    return [c for c, s in seats.items() if s > 0]


def _get_seat_count(flight: dict, date: str, cabin: str) -> int:
    return flight.get("available_seats", {}).get(date, {}).get(cabin, 0)


def _decrement_seats(flight: dict, date: str, cabin: str, count: int = 1) -> bool:
    seats = flight.get("available_seats", {}).get(date, {})
    if seats.get(cabin, 0) < count:
        return False
    seats[cabin] -= count
    return True


# ---------------------------------------------------------------------------
# 1. book_reservation
# ---------------------------------------------------------------------------

class BookReservationTool(MockTool):
    name = "book_reservation"
    description = (
        "Book a flight reservation. Searches for available flights and creates "
        "a reservation. Pass a list of flight segments (flight_number + date) "
        "and passenger information."
    )
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "The user ID making the booking."},
            "origin": {"type": "string", "description": "Origin airport code (e.g. 'SFO')."},
            "destination": {"type": "string", "description": "Destination airport code (e.g. 'JFK')."},
            "flight_type": {
                "type": "string",
                "enum": ["round_trip", "one_way"],
                "description": "Whether booking is round-trip or one-way.",
            },
            "cabin": {
                "type": "string",
                "enum": ["business", "economy", "basic_economy"],
                "description": "Cabin class for the booking.",
            },
            "flights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string", "description": "Flight number, e.g. 'AA101'."},
                        "date": {"type": "string", "description": "Flight date in YYYY-MM-DD format."},
                    },
                    "required": ["flight_number", "date"],
                },
                "description": "List of flight segments to book.",
            },
            "passengers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full passenger name."},
                        "dob": {"type": "string", "description": "Date of birth (YYYY-MM-DD)."},
                    },
                    "required": ["name", "dob"],
                },
                "description": "List of passengers on the reservation.",
            },
            "payment": {
                "type": "object",
                "description": "Payment details with 'method' and optional 'last_four'.",
            },
        },
        "required": ["user_id", "origin", "destination", "flight_type", "cabin", "flights", "passengers"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err

        user_id = arguments["user_id"]
        cabin = arguments["cabin"]
        flights_arg = arguments["flights"]
        passengers = arguments["passengers"]
        flight_type = arguments["flight_type"]
        destination = arguments["destination"]

        # Validate user
        user = _find_airline_user(db_state, user_id)
        if user is None:
            return {"status": "error", "message": f"User '{user_id}' not found"}

        # Validate flights
        booked_segments = []
        total_fare = 0.0
        for i, seg in enumerate(flights_arg):
            fn = seg["flight_number"]
            date = seg["date"]
            flight = _find_flight(db_state, fn)
            if flight is None:
                return {"status": "error", "message": f"Flight '{fn}' not found"}
            if date not in flight.get("dates", []):
                return {"status": "error", "message": f"Flight '{fn}' does not operate on {date}"}
            if cabin not in flight.get("price", {}):
                return {"status": "error", "message": f"Cabin '{cabin}' not available on flight '{fn}'"}
            if _get_seat_count(flight, date, cabin) < len(passengers):
                return {
                    "status": "error",
                    "message": f"Not enough '{cabin}' seats on '{fn}' for {date} (need {len(passengers)})",
                }
            # Validate segment connectivity
            if i == 0 and flight["origin"] != arguments["origin"]:
                return {"status": "error", "message": f"First flight '{fn}' origin '{flight['origin']}' does not match '{arguments['origin']}'"}
            if i == len(flights_arg) - 1 and flight["destination"] != destination:
                return {"status": "error", "message": f"Last flight '{fn}' destination '{flight['destination']}' does not match '{destination}'"}
            if i > 0:
                prev = booked_segments[-1]
                if prev["flight"]["destination"] != flight["origin"]:
                    return {"status": "error", "message": f"Flight '{fn}' origin '{flight['origin']}' does not connect from previous destination '{prev['flight']['destination']}'"}

            fare = flight["price"][cabin] * len(passengers)
            total_fare += fare
            booked_segments.append({
                "flight_number": fn,
                "date": date,
                "origin": flight["origin"],
                "destination": flight["destination"],
                "cabin": cabin,
                "fare": fare / len(passengers),
            })
            # Decrement seats
            _decrement_seats(flight, date, cabin, len(passengers))

        # Check round-trip: last flight must end at origin
        if flight_type == "round_trip":
            last_dest = booked_segments[-1]["destination"]
            if last_dest != arguments["origin"]:
                return {"status": "error", "message": f"Round-trip must end at origin '{arguments['origin']}', but ends at '{last_dest}'"}

        # Create reservation
        reservation_id = f"RA{random.randint(100, 999)}"
        payment_info = arguments.get("payment", {"method": "credit_card"})
        reservation = {
            "reservation_id": reservation_id,
            "user_id": user_id,
            "flights": booked_segments,
            "passengers": passengers,
            "bags": {"total": len(passengers), "nonfree": 0},
            "payment": {**payment_info, "amount": round(total_fare, 2)},
            "status": "confirmed",
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        db_state["reservations"].append(reservation)

        return {
            "status": "success",
            "result": {
                "reservation_id": reservation_id,
                "flights": booked_segments,
                "passengers": passengers,
                "total_fare": round(total_fare, 2),
                "status": "confirmed",
            },
        }


# ---------------------------------------------------------------------------
# 2. cancel_reservation
# ---------------------------------------------------------------------------

class CancelReservationTool(MockTool):
    name = "cancel_reservation"
    description = "Cancel an existing reservation. Cannot cancel already cancelled reservations."
    parameters = {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string", "description": "The reservation ID (e.g. 'RA001')."},
        },
        "required": ["reservation_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        rid = arguments["reservation_id"]
        res = _find_reservation(db_state, rid)
        if res is None:
            return {"status": "error", "message": f"Reservation '{rid}' not found"}
        if res["status"] == "cancelled":
            return {"status": "error", "message": f"Reservation '{rid}' is already cancelled"}

        # Return seats to inventory
        for seg in res.get("flights", []):
            flight = _find_flight(db_state, seg["flight_number"])
            if flight:
                cabin = seg.get("cabin", "economy")
                seats = flight.get("available_seats", {}).get(seg["date"], {})
                seats[cabin] = seats.get(cabin, 0) + len(res.get("passengers", []))

        res["status"] = "cancelled"
        res["cancelled_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        return {"status": "success", "result": {"reservation_id": rid, "status": "cancelled"}}


# ---------------------------------------------------------------------------
# 3. get_reservation_details
# ---------------------------------------------------------------------------

class GetReservationDetailsTool(MockTool):
    name = "get_reservation_details"
    description = "Get full reservation details including flights, passengers, bags, and payment."
    parameters = {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string", "description": "The reservation ID."},
        },
        "required": ["reservation_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        res = _find_reservation(db_state, arguments["reservation_id"])
        if res is None:
            return {"status": "error", "message": f"Reservation '{arguments['reservation_id']}' not found"}
        return {"status": "success", "result": dict(res)}


# ---------------------------------------------------------------------------
# 4. get_user_details (airline)
# ---------------------------------------------------------------------------

class AirlineGetUserDetailsTool(MockTool):
    name = "get_user_details"
    description = "Get full user profile including loyalty tier, points, and travel documents."
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "The user ID (e.g. 'UA001')."},
        },
        "required": ["user_id"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        user = _find_airline_user(db_state, arguments["user_id"])
        if user is None:
            return {"status": "error", "message": f"User '{arguments['user_id']}' not found"}

        # Return safe copy with loyalty info
        safe = dict(user)
        return {"status": "success", "result": safe}


# ---------------------------------------------------------------------------
# 5. list_all_airports
# ---------------------------------------------------------------------------

class ListAllAirportsTool(MockTool):
    name = "list_all_airports"
    description = "Return a list of all airports with codes and names."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        airports = []
        for code, info in db_state["airports"].items():
            airports.append({
                "code": code,
                "name": info["name"],
                "city": info["city"],
                "country": info["country"],
            })
        airports.sort(key=lambda x: x["code"])
        return {"status": "success", "result": airports}


# ---------------------------------------------------------------------------
# 6. search_direct_flight
# ---------------------------------------------------------------------------

class SearchDirectFlightTool(MockTool):
    name = "search_direct_flight"
    description = (
        "Search for direct flights between two airports on a given date. "
        "Returns available flights with seat counts and prices by cabin."
    )
    parameters = {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Origin airport code (e.g. 'SFO')."},
            "destination": {"type": "string", "description": "Destination airport code (e.g. 'JFK')."},
            "date": {"type": "string", "description": "Flight date in YYYY-MM-DD format."},
        },
        "required": ["origin", "destination", "date"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        origin = arguments["origin"].upper()
        dest = arguments["destination"].upper()
        date = arguments["date"]

        # Verify airports exist
        airports = db_state["airports"]
        if origin not in airports:
            return {"status": "error", "message": f"Airport '{origin}' not found"}
        if dest not in airports:
            return {"status": "error", "message": f"Airport '{dest}' not found"}

        results = []
        for fn, flight in db_state.get("flights", {}).items():
            if flight["origin"] == origin and flight["destination"] == dest and date in flight.get("dates", []):
                seats = flight.get("available_seats", {}).get(date, {})
                results.append({
                    "flight_number": fn,
                    "airline": flight.get("airline", ""),
                    "origin": origin,
                    "destination": dest,
                    "departure_time": flight["departure_time"],
                    "arrival_time": flight["arrival_time"],
                    "date": date,
                    "available_seats": seats,
                    "prices": flight.get("price", {}),
                    "status": flight.get("status", {}).get(date, "on_time"),
                })

        if not results:
            return {"status": "error", "message": f"No direct flights found from {origin} to {dest} on {date}"}
        return {"status": "success", "result": results}


# ---------------------------------------------------------------------------
# 7. search_onestop_flight
# ---------------------------------------------------------------------------

class SearchOnestopFlightTool(MockTool):
    name = "search_onestop_flight"
    description = (
        "Search for flights with exactly one stop/connection between two airports "
        "on a given date. The layover must be 1-8 hours."
    )
    parameters = {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Origin airport code."},
            "destination": {"type": "string", "description": "Destination airport code."},
            "date": {"type": "string", "description": "Flight date in YYYY-MM-DD format."},
        },
        "required": ["origin", "destination", "date"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        origin = arguments["origin"].upper()
        dest = arguments["destination"].upper()
        date = arguments["date"]

        airports = db_state["airports"]
        if origin not in airports:
            return {"status": "error", "message": f"Airport '{origin}' not found"}
        if dest not in airports:
            return {"status": "error", "message": f"Airport '{dest}' not found"}

        flights = db_state.get("flights", {})
        results = []
        for fn1, f1 in flights.items():
            if f1["origin"] != origin or date not in f1.get("dates", []):
                continue
            f1_seats = f1.get("available_seats", {}).get(date, {})
            if not any(s > 0 for s in f1_seats.values()):
                continue

            for fn2, f2 in flights.items():
                if fn1 == fn2:
                    continue
                if f2["destination"] != dest:
                    continue
                if f1["destination"] != f2["origin"]:
                    continue

                # Check both flights operate on date (or next day for red-eye)
                date2 = date
                if f1["arrival_time"].endswith("+1"):
                    d = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
                    date2 = d.strftime("%Y-%m-%d")
                if date2 not in f2.get("dates", []):
                    continue

                # Check connection time (1-8 hours)
                if not _connection_ok(f1["arrival_time"], f2["departure_time"]):
                    continue

                f2_seats = f2.get("available_seats", {}).get(date2, {})
                if not any(s > 0 for s in f2_seats.values()):
                    continue

                # Calculate combined prices (cheapest available cabin)
                combined_price = {}
                for cabin in ["business", "economy", "basic_economy"]:
                    p1 = f1.get("price", {}).get(cabin)
                    p2 = f2.get("price", {}).get(cabin)
                    if p1 is not None and p2 is not None:
                        combined_price[cabin] = p1 + p2

                results.append({
                    "flights": [
                        {
                            "flight_number": fn1,
                            "origin": f1["origin"],
                            "destination": f1["destination"],
                            "departure_time": f1["departure_time"],
                            "arrival_time": f1["arrival_time"],
                            "date": date,
                            "airline": f1.get("airline", ""),
                            "available_seats": f1_seats,
                        },
                        {
                            "flight_number": fn2,
                            "origin": f2["origin"],
                            "destination": f2["destination"],
                            "departure_time": f2["departure_time"],
                            "arrival_time": f2["arrival_time"],
                            "date": date2,
                            "airline": f2.get("airline", ""),
                            "available_seats": f2_seats,
                        },
                    ],
                    "connection_airport": f1["destination"],
                    "prices": combined_price,
                })

        if not results:
            return {"status": "error", "message": f"No one-stop flights found from {origin} to {dest} on {date}"}
        return {"status": "success", "result": results}


# ---------------------------------------------------------------------------
# 8. send_certificate
# ---------------------------------------------------------------------------

class SendCertificateTool(MockTool):
    name = "send_certificate"
    description = "Send a travel certificate or voucher to a user by email."
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "The user ID to send the certificate to."},
            "amount": {"type": "number", "description": "Certificate amount in dollars."},
        },
        "required": ["user_id", "amount"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        user = _find_airline_user(db_state, arguments["user_id"])
        if user is None:
            return {"status": "error", "message": f"User '{arguments['user_id']}' not found"}
        amount = float(arguments["amount"])
        if amount <= 0:
            return {"status": "error", "message": "Certificate amount must be positive"}

        cert_code = f"CERT-{random.randint(100000, 999999)}"
        return {
            "status": "success",
            "result": {
                "certificate_code": cert_code,
                "user_id": user["user_id"],
                "email": user["email"],
                "amount": amount,
                "message": f"A ${amount:.2f} certificate ({cert_code}) has been sent to {user['email']}.",
            },
        }


# ---------------------------------------------------------------------------
# 9. update_reservation_baggages
# ---------------------------------------------------------------------------

class UpdateReservationBaggagesTool(MockTool):
    name = "update_reservation_baggages"
    description = "Update the baggage allowance for a reservation."
    parameters = {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string", "description": "The reservation ID."},
            "total_bags": {"type": "integer", "description": "Total number of checked bags.", "minimum": 0},
            "nonfree_bags": {
                "type": "integer",
                "description": "Number of bags that incur a fee (beyond free allowance).",
                "minimum": 0,
            },
        },
        "required": ["reservation_id", "total_bags", "nonfree_bags"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        res = _find_reservation(db_state, arguments["reservation_id"])
        if res is None:
            return {"status": "error", "message": f"Reservation '{arguments['reservation_id']}' not found"}
        total = int(arguments["total_bags"])
        nonfree = int(arguments["nonfree_bags"])
        if nonfree > total:
            return {"status": "error", "message": "nonfree_bags cannot exceed total_bags"}

        res["bags"] = {"total": total, "nonfree": nonfree}
        # Non-free bags incur a fee: $35/bag in economy, $0 in business
        bag_fee = 0.0
        for seg in res.get("flights", []):
            if seg.get("cabin") != "business":
                bag_fee += nonfree * 35.0

        return {
            "status": "success",
            "result": {
                "reservation_id": arguments["reservation_id"],
                "bags": res["bags"],
                "additional_bag_fee": round(bag_fee, 2),
            },
        }


# ---------------------------------------------------------------------------
# 10. update_reservation_flights
# ---------------------------------------------------------------------------

class UpdateReservationFlightsTool(MockTool):
    name = "update_reservation_flights"
    description = (
        "Change the flights on an existing reservation. "
        "Basic economy fares cannot be changed. Economy fares may incur a change fee. "
        "Business fares can be changed freely."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string", "description": "The reservation ID."},
            "new_flights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["flight_number", "date"],
                },
                "description": "Replacement flight segments.",
            },
        },
        "required": ["reservation_id", "new_flights"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        res = _find_reservation(db_state, arguments["reservation_id"])
        if res is None:
            return {"status": "error", "message": f"Reservation '{arguments['reservation_id']}' not found"}
        if res["status"] != "confirmed":
            return {"status": "error", "message": f"Reservation '{arguments['reservation_id']}' is '{res['status']}', not 'confirmed'"}

        # Check fare class rules
        for seg in res.get("flights", []):
            if seg.get("cabin") == "basic_economy":
                return {
                    "status": "error",
                    "message": "Cannot change flights on a basic economy reservation. Basic economy fares are non-changeable.",
                }

        # Release old seats
        for seg in res.get("flights", []):
            flight = _find_flight(db_state, seg["flight_number"])
            if flight:
                cabin = seg.get("cabin", "economy")
                seats = flight.get("available_seats", {}).get(seg["date"], {})
                seats[cabin] = seats.get(cabin, 0) + len(res.get("passengers", []))

        # Book new flights
        new_segments = []
        total_new_fare = 0.0
        for s in arguments["new_flights"]:
            fn = s["flight_number"]
            date = s["date"]
            flight = _find_flight(db_state, fn)
            if flight is None:
                return {"status": "error", "message": f"Flight '{fn}' not found"}
            if date not in flight.get("dates", []):
                return {"status": "error", "message": f"Flight '{fn}' does not operate on {date}"}
            cabin = res["flights"][0].get("cabin", "economy") if res["flights"] else "economy"
            if _get_seat_count(flight, date, cabin) < len(res.get("passengers", [])):
                return {"status": "error", "message": f"Not enough '{cabin}' seats on '{fn}' for {date}"}

            fare = flight["price"].get(cabin, 0) * len(res.get("passengers", []))
            total_new_fare += fare
            new_segments.append({
                "flight_number": fn,
                "date": date,
                "origin": flight["origin"],
                "destination": flight["destination"],
                "cabin": cabin,
                "fare": fare / max(1, len(res.get("passengers", []))),
            })
            _decrement_seats(flight, date, cabin, len(res.get("passengers", [])))

        # Compute change fee
        old_fare = sum(s.get("fare", 0) * len(res.get("passengers", [])) for s in res.get("flights", []))
        change_fee = 0.0
        if res["flights"][0].get("cabin") == "economy":
            change_fee = 75.0
        fare_diff = total_new_fare - old_fare
        total_change = round(fare_diff + change_fee, 2)

        res["flights"] = new_segments
        if total_change > 0:
            res["payment"]["amount"] = round(res["payment"]["amount"] + total_change, 2)

        return {
            "status": "success",
            "result": {
                "reservation_id": arguments["reservation_id"],
                "new_flights": new_segments,
                "change_fee": change_fee,
                "fare_difference": round(fare_diff, 2),
                "total_change": total_change,
            },
        }


# ---------------------------------------------------------------------------
# 11. update_reservation_passengers
# ---------------------------------------------------------------------------

class UpdateReservationPassengersTool(MockTool):
    name = "update_reservation_passengers"
    description = "Update the passenger information on a reservation."
    parameters = {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string", "description": "The reservation ID."},
            "passengers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full passenger name."},
                        "dob": {"type": "string", "description": "Date of birth (YYYY-MM-DD)."},
                    },
                    "required": ["name", "dob"],
                },
                "description": "Updated list of passengers.",
            },
        },
        "required": ["reservation_id", "passengers"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        res = _find_reservation(db_state, arguments["reservation_id"])
        if res is None:
            return {"status": "error", "message": f"Reservation '{arguments['reservation_id']}' not found"}
        if res["status"] != "confirmed":
            return {"status": "error", "message": f"Reservation '{arguments['reservation_id']}' is '{res['status']}', not 'confirmed'"}

        new_passengers = arguments["passengers"]
        old_count = len(res.get("passengers", []))
        res["passengers"] = new_passengers

        # Adjust bags proportionally
        if old_count > 0 and len(new_passengers) != old_count:
            ratio = len(new_passengers) / old_count
            bags = res.get("bags", {"total": 1, "nonfree": 0})
            res["bags"] = {
                "total": max(0, round(bags.get("total", 0) * ratio)),
                "nonfree": max(0, round(bags.get("nonfree", 0) * ratio)),
            }

        return {
            "status": "success",
            "result": {
                "reservation_id": arguments["reservation_id"],
                "passengers": new_passengers,
                "passenger_count": len(new_passengers),
            },
        }


# ---------------------------------------------------------------------------
# 12. get_flight_status
# ---------------------------------------------------------------------------

class GetFlightStatusTool(MockTool):
    name = "get_flight_status"
    description = "Get the current status of a specific flight on a given date."
    parameters = {
        "type": "object",
        "properties": {
            "flight_number": {"type": "string", "description": "The flight number (e.g. 'AA101')."},
            "date": {"type": "string", "description": "Flight date in YYYY-MM-DD format."},
        },
        "required": ["flight_number", "date"],
    }

    def execute(self, arguments: dict[str, Any], db_state: dict[str, Any]) -> dict[str, Any]:
        err = self._validate_args(arguments)
        if err:
            return err
        fn = arguments["flight_number"]
        date = arguments["date"]

        flight = _find_flight(db_state, fn)
        if flight is None:
            return {"status": "error", "message": f"Flight '{fn}' not found"}
        if date not in flight.get("dates", []):
            return {"status": "error", "message": f"Flight '{fn}' does not operate on {date}"}

        status = flight.get("status", {}).get(date, "on_time")
        seats = flight.get("available_seats", {}).get(date, {})
        return {
            "status": "success",
            "result": {
                "flight_number": fn,
                "date": date,
                "origin": flight["origin"],
                "destination": flight["destination"],
                "departure_time": flight["departure_time"],
                "arrival_time": flight["arrival_time"],
                "airline": flight.get("airline", ""),
                "flight_status": status,
                "available_seats": seats,
            },
        }


# ---------------------------------------------------------------------------
# Tool registry for the airline domain
# ---------------------------------------------------------------------------

def _make_airline_tools() -> list[MockTool]:
    """Instantiate all 12 airline-domain tools + 3 shared utilities."""
    return [
        BookReservationTool(),
        CancelReservationTool(),
        GetReservationDetailsTool(),
        AirlineGetUserDetailsTool(),
        ListAllAirportsTool(),
        SearchDirectFlightTool(),
        SearchOnestopFlightTool(),
        SendCertificateTool(),
        UpdateReservationBaggagesTool(),
        UpdateReservationFlightsTool(),
        UpdateReservationPassengersTool(),
        GetFlightStatusTool(),
        calculate_tool,
        think_tool,
        transfer_to_human_agents_tool,
    ]


# Module-level export for convenience
airline_tools: list[MockTool] = _make_airline_tools()

# Re-export individual tool classes for direct import
__all__ = [
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
