"""
In-memory mock database engine for tau-bench tools.

Provides seeded, thread-safe dict storage with realistic data for both
the airline and retail domains.  The database state is plain Python dicts
and lists — trivially JSON-serializable for persistence.

Usage::

    db = MockDatabase("retail")
    state = db.state   # reference to the in-memory dict
    db.execute(tool, {"user_id": "U001"})
    db.save("retail_state.json")
    db2 = MockDatabase.load("retail_state.json")
"""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data factories — each call returns a *fresh* dict
# ---------------------------------------------------------------------------


def _seed_retail_users() -> list[dict[str, Any]]:
    return [
        {
            "user_id": "U001",
            "first_name": "Alice",
            "last_name": "Chen",
            "name": "Alice Chen",
            "email": "alice.chen@email.com",
            "phone": "415-555-0101",
            "address": {
                "address1": "123 Main St",
                "address2": "Apt 4B",
                "city": "San Francisco",
                "state": "CA",
                "zip": "94102",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "1234", "brand": "Visa"},
                {"type": "paypal", "email": "alice.chen@email.com"},
            ],
            "member_since": "2024-01-15",
        },
        {
            "user_id": "U002",
            "first_name": "Bob",
            "last_name": "Williams",
            "name": "Bob Williams",
            "email": "bob.w@email.com",
            "phone": "212-555-0102",
            "address": {
                "address1": "456 Park Ave",
                "address2": "",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "5678", "brand": "Mastercard"},
            ],
            "member_since": "2023-06-20",
        },
        {
            "user_id": "U003",
            "first_name": "Carlos",
            "last_name": "Rodriguez",
            "name": "Carlos Rodriguez",
            "email": "carlos.r@email.com",
            "phone": "305-555-0103",
            "address": {
                "address1": "789 Ocean Dr",
                "address2": "",
                "city": "Miami",
                "state": "FL",
                "zip": "33101",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "9012", "brand": "Amex"},
            ],
            "member_since": "2024-03-10",
        },
        {
            "user_id": "U004",
            "first_name": "Diana",
            "last_name": "Park",
            "name": "Diana Park",
            "email": "diana.park@email.com",
            "phone": "312-555-0104",
            "address": {
                "address1": "321 Michigan Ave",
                "address2": "Unit 5",
                "city": "Chicago",
                "state": "IL",
                "zip": "60601",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "3456", "brand": "Visa"},
                {"type": "gift_card", "last_four": "9999", "brand": "Store"},
            ],
            "member_since": "2023-11-05",
        },
        {
            "user_id": "U005",
            "first_name": "Edward",
            "last_name": "Kim",
            "name": "Edward Kim",
            "email": "ed.kim@email.com",
            "phone": "206-555-0105",
            "address": {
                "address1": "654 Pine St",
                "address2": "",
                "city": "Seattle",
                "state": "WA",
                "zip": "98101",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "7890", "brand": "Mastercard"},
            ],
            "member_since": "2024-07-22",
        },
        {
            "user_id": "U006",
            "first_name": "Fatima",
            "last_name": "Hassan",
            "name": "Fatima Hassan",
            "email": "fatima.h@email.com",
            "phone": "617-555-0106",
            "address": {
                "address1": "87 Beacon St",
                "address2": "",
                "city": "Boston",
                "state": "MA",
                "zip": "02108",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "2468", "brand": "Visa"},
            ],
            "member_since": "2024-09-14",
        },
        {
            "user_id": "U007",
            "first_name": "George",
            "last_name": "Thompson",
            "name": "George Thompson",
            "email": "george.t@email.com",
            "phone": "303-555-0107",
            "address": {
                "address1": "1590 Broadway",
                "address2": "Suite 200",
                "city": "Denver",
                "state": "CO",
                "zip": "80202",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "1357", "brand": "Amex"},
            ],
            "member_since": "2023-02-28",
        },
        {
            "user_id": "U008",
            "first_name": "Hannah",
            "last_name": "Lee",
            "name": "Hannah Lee",
            "email": "hannah.lee@email.com",
            "phone": "213-555-0108",
            "address": {
                "address1": "2000 Sunset Blvd",
                "address2": "Apt 12A",
                "city": "Los Angeles",
                "state": "CA",
                "zip": "90028",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "8642", "brand": "Visa"},
                {"type": "paypal", "email": "hannah.lee@email.com"},
            ],
            "member_since": "2024-04-17",
        },
        {
            "user_id": "U009",
            "first_name": "Ivan",
            "last_name": "Petrov",
            "name": "Ivan Petrov",
            "email": "ivan.p@email.com",
            "phone": "832-555-0109",
            "address": {
                "address1": "4100 Westheimer Rd",
                "address2": "",
                "city": "Houston",
                "state": "TX",
                "zip": "77027",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "9753", "brand": "Mastercard"},
            ],
            "member_since": "2024-08-03",
        },
        {
            "user_id": "U010",
            "first_name": "Julia",
            "last_name": "Martinez",
            "name": "Julia Martinez",
            "email": "julia.m@email.com",
            "phone": "602-555-0110",
            "address": {
                "address1": "1 E Washington St",
                "address2": "Floor 14",
                "city": "Phoenix",
                "state": "AZ",
                "zip": "85004",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "1122", "brand": "Visa"},
            ],
            "member_since": "2023-12-08",
        },
        {
            "user_id": "U011",
            "first_name": "Kevin",
            "last_name": "O'Brien",
            "name": "Kevin O'Brien",
            "email": "kevin.ob@email.com",
            "phone": "503-555-0111",
            "address": {
                "address1": "720 SW Broadway",
                "address2": "",
                "city": "Portland",
                "state": "OR",
                "zip": "97205",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "3344", "brand": "Mastercard"},
            ],
            "member_since": "2024-05-30",
        },
        {
            "user_id": "U012",
            "first_name": "Lisa",
            "last_name": "Nakamura",
            "name": "Lisa Nakamura",
            "email": "lisa.n@email.com",
            "phone": "808-555-0112",
            "address": {
                "address1": "1450 Ala Moana Blvd",
                "address2": "Apt 2305",
                "city": "Honolulu",
                "state": "HI",
                "zip": "96814",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "5566", "brand": "Visa"},
                {"type": "credit_card", "last_four": "7788", "brand": "Amex"},
            ],
            "member_since": "2024-11-12",
        },
        {
            "user_id": "U013",
            "first_name": "Michael",
            "last_name": "Brown",
            "name": "Michael Brown",
            "email": "michael.b@email.com",
            "phone": "404-555-0113",
            "address": {
                "address1": "1200 Peachtree St",
                "address2": "",
                "city": "Atlanta",
                "state": "GA",
                "zip": "30309",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "9900", "brand": "Visa"},
            ],
            "member_since": "2023-08-15",
        },
        {
            "user_id": "U014",
            "first_name": "Nancy",
            "last_name": "Davis",
            "name": "Nancy Davis",
            "email": "nancy.d@email.com",
            "phone": "214-555-0114",
            "address": {
                "address1": "300 Reunion Blvd",
                "address2": "Suite 800",
                "city": "Dallas",
                "state": "TX",
                "zip": "75207",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "2233", "brand": "Mastercard"},
            ],
            "member_since": "2024-02-19",
        },
        {
            "user_id": "U015",
            "first_name": "Oscar",
            "last_name": "Garcia",
            "name": "Oscar Garcia",
            "email": "oscar.g@email.com",
            "phone": "210-555-0115",
            "address": {
                "address1": "111 Alamo Plaza",
                "address2": "",
                "city": "San Antonio",
                "state": "TX",
                "zip": "78205",
                "country": "USA",
            },
            "payment_methods": [
                {"type": "credit_card", "last_four": "4455", "brand": "Amex"},
                {"type": "paypal", "email": "oscar.g@email.com"},
            ],
            "member_since": "2024-10-07",
        },
    ]


def _seed_retail_products() -> list[dict[str, Any]]:
    return [
        {
            "product_id": "P001",
            "name": "Wireless Noise-Cancelling Headphones",
            "type": "Electronics",
            "description": (
                "Premium over-ear headphones with active noise cancellation, "
                "30-hour battery life, and Bluetooth 5.3 connectivity."
            ),
            "variants": [
                {"item_id": "I001", "variant_name": "Black", "price": 129.99, "available": True},
                {"item_id": "I002", "variant_name": "White", "price": 129.99, "available": True},
                {"item_id": "I003", "variant_name": "Navy Blue", "price": 139.99, "available": False},
            ],
        },
        {
            "product_id": "P002",
            "name": "Smart Fitness Watch",
            "type": "Electronics",
            "description": (
                "Water-resistant smartwatch with GPS, heart-rate monitor, "
                "sleep tracking, and 7-day battery life."
            ),
            "variants": [
                {"item_id": "I004", "variant_name": "Black/Small", "price": 199.99, "available": True},
                {"item_id": "I005", "variant_name": "Black/Large", "price": 199.99, "available": True},
                {"item_id": "I006", "variant_name": "Rose Gold/Small", "price": 219.99, "available": True},
            ],
        },
        {
            "product_id": "P003",
            "name": "Portable Bluetooth Speaker",
            "type": "Electronics",
            "description": (
                "Compact waterproof speaker with 360-degree sound, 12-hour "
                "playtime, and built-in microphone."
            ),
            "variants": [
                {"item_id": "I007", "variant_name": "Black", "price": 49.99, "available": True},
                {"item_id": "I008", "variant_name": "Blue", "price": 49.99, "available": True},
                {"item_id": "I009", "variant_name": "Red", "price": 54.99, "available": True},
            ],
        },
        {
            "product_id": "P004",
            "name": "Professional Running Shoes",
            "type": "Footwear",
            "description": (
                "Lightweight running shoes with responsive cushioning, "
                "breathable mesh upper, and durable rubber outsole."
            ),
            "variants": [
                {"item_id": "I010", "variant_name": "Men's 8", "price": 119.99, "available": True},
                {"item_id": "I011", "variant_name": "Men's 9", "price": 119.99, "available": True},
                {"item_id": "I012", "variant_name": "Men's 10", "price": 119.99, "available": True},
                {"item_id": "I013", "variant_name": "Women's 7", "price": 119.99, "available": True},
                {"item_id": "I014", "variant_name": "Women's 8", "price": 119.99, "available": False},
            ],
        },
        {
            "product_id": "P005",
            "name": "Winter Insulated Jacket",
            "type": "Clothing",
            "description": (
                "Waterproof insulated jacket with adjustable hood, multiple "
                "pockets, and thermal lining. Rated to -20F."
            ),
            "variants": [
                {"item_id": "I015", "variant_name": "Black / M", "price": 189.99, "available": True},
                {"item_id": "I016", "variant_name": "Black / L", "price": 189.99, "available": True},
                {"item_id": "I017", "variant_name": "Navy / M", "price": 189.99, "available": True},
                {"item_id": "I018", "variant_name": "Olive / L", "price": 199.99, "available": True},
            ],
        },
        {
            "product_id": "P006",
            "name": "Premium Cotton T-Shirt",
            "type": "Clothing",
            "description": (
                "100% organic cotton crew-neck t-shirt. Pre-shrunk, "
                "tagless, and available in multiple colors."
            ),
            "variants": [
                {"item_id": "I019", "variant_name": "White / M", "price": 24.99, "available": True},
                {"item_id": "I020", "variant_name": "Black / M", "price": 24.99, "available": True},
                {"item_id": "I021", "variant_name": "Gray / L", "price": 24.99, "available": True},
            ],
        },
        {
            "product_id": "P007",
            "name": "Programmable Coffee Maker",
            "type": "Home",
            "description": (
                "12-cup coffee maker with programmable timer, auto-shutoff, "
                "and brew-strength control."
            ),
            "variants": [
                {"item_id": "I022", "variant_name": "Stainless Steel", "price": 79.99, "available": True},
                {"item_id": "I023", "variant_name": "Black", "price": 69.99, "available": True},
            ],
        },
        {
            "product_id": "P008",
            "name": "Stainless Steel Toaster",
            "type": "Home",
            "description": (
                "4-slice toaster with bagel/defrost/reheat settings, "
                "removable crumb tray, and extra-wide slots."
            ),
            "variants": [
                {"item_id": "I024", "variant_name": "Silver", "price": 44.99, "available": True},
                {"item_id": "I025", "variant_name": "Black", "price": 44.99, "available": True},
            ],
        },
        {
            "product_id": "P009",
            "name": "Cordless Stick Vacuum",
            "type": "Home",
            "description": (
                "Lightweight cordless vacuum with 40-minute runtime, "
                "HEPA filtration, and detachable hand vac."
            ),
            "variants": [
                {"item_id": "I026", "variant_name": "Standard", "price": 249.99, "available": True},
                {"item_id": "I027", "variant_name": "Pet Edition", "price": 279.99, "available": True},
            ],
        },
        {
            "product_id": "P010",
            "name": "Python Cookbook (3rd Edition)",
            "type": "Books",
            "description": (
                "Comprehensive recipes for Python 3.12 covering data structures, "
                "algorithms, concurrency, and web development."
            ),
            "variants": [
                {"item_id": "I028", "variant_name": "Paperback", "price": 49.99, "available": True},
                {"item_id": "I029", "variant_name": "Hardcover", "price": 69.99, "available": True},
            ],
        },
        {
            "product_id": "P011",
            "name": "Clean Code (Robert C. Martin)",
            "type": "Books",
            "description": (
                "A handbook of agile software craftsmanship covering principles, "
                "patterns, and practices of writing clean code."
            ),
            "variants": [
                {"item_id": "I030", "variant_name": "Paperback", "price": 39.99, "available": True},
                {"item_id": "I031", "variant_name": "Hardcover", "price": 59.99, "available": False},
            ],
        },
        {
            "product_id": "P012",
            "name": "Ergonomic Office Chair",
            "type": "Furniture",
            "description": (
                "Adjustable lumbar support, mesh back, 3D armrests, and "
                "tilt-lock mechanism. Supports up to 300 lbs."
            ),
            "variants": [
                {"item_id": "I032", "variant_name": "Black / Standard", "price": 349.99, "available": True},
                {"item_id": "I033", "variant_name": "Gray / Standard", "price": 349.99, "available": True},
                {"item_id": "I034", "variant_name": "Black / High-Back", "price": 429.99, "available": True},
            ],
        },
    ]


def _seed_retail_orders() -> list[dict[str, Any]]:
    return [
        {
            "order_id": "O001",
            "user_id": "U001",
            "status": "delivered",
            "items": [
                {"item_id": "I001", "product_id": "P001", "name": "Wireless Noise-Cancelling Headphones (Black)", "quantity": 1, "unit_price": 129.99},
            ],
            "shipping_address": {
                "address1": "123 Main St", "address2": "Apt 4B",
                "city": "San Francisco", "state": "CA", "zip": "94102", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "1234", "amount": 142.99},
            "created_at": "2026-05-20T10:15:00",
            "delivered_at": "2026-05-26T14:30:00",
        },
        {
            "order_id": "O002",
            "user_id": "U001",
            "status": "pending",
            "items": [
                {"item_id": "I010", "product_id": "P004", "name": "Professional Running Shoes (Men's 8)", "quantity": 1, "unit_price": 119.99},
            ],
            "shipping_address": {
                "address1": "123 Main St", "address2": "Apt 4B",
                "city": "San Francisco", "state": "CA", "zip": "94102", "country": "USA",
            },
            "payment": {"method": "paypal", "amount": 131.99},
            "created_at": "2026-06-08T09:45:00",
        },
        {
            "order_id": "O003",
            "user_id": "U002",
            "status": "delivered",
            "items": [
                {"item_id": "I004", "product_id": "P002", "name": "Smart Fitness Watch (Black/Small)", "quantity": 1, "unit_price": 199.99},
                {"item_id": "I020", "product_id": "P006", "name": "Premium Cotton T-Shirt (Black / M)", "quantity": 2, "unit_price": 24.99},
            ],
            "shipping_address": {
                "address1": "456 Park Ave", "address2": "",
                "city": "New York", "state": "NY", "zip": "10001", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "5678", "amount": 274.97},
            "created_at": "2026-05-15T14:20:00",
            "delivered_at": "2026-05-22T11:00:00",
        },
        {
            "order_id": "O004",
            "user_id": "U002",
            "status": "shipped",
            "items": [
                {"item_id": "I030", "product_id": "P011", "name": "Clean Code (Paperback)", "quantity": 1, "unit_price": 39.99},
            ],
            "shipping_address": {
                "address1": "456 Park Ave", "address2": "",
                "city": "New York", "state": "NY", "zip": "10001", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "5678", "amount": 44.99},
            "created_at": "2026-06-05T07:30:00",
            "shipped_at": "2026-06-06T16:00:00",
        },
        {
            "order_id": "O005",
            "user_id": "U003",
            "status": "delivered",
            "items": [
                {"item_id": "I022", "product_id": "P007", "name": "Programmable Coffee Maker (Stainless Steel)", "quantity": 1, "unit_price": 79.99},
                {"item_id": "I024", "product_id": "P008", "name": "Stainless Steel Toaster (Silver)", "quantity": 1, "unit_price": 44.99},
            ],
            "shipping_address": {
                "address1": "789 Ocean Dr", "address2": "",
                "city": "Miami", "state": "FL", "zip": "33101", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "9012", "amount": 136.98},
            "created_at": "2026-05-12T11:00:00",
            "delivered_at": "2026-05-19T15:45:00",
        },
        {
            "order_id": "O006",
            "user_id": "U003",
            "status": "pending",
            "items": [
                {"item_id": "I007", "product_id": "P003", "name": "Portable Bluetooth Speaker (Black)", "quantity": 1, "unit_price": 49.99},
            ],
            "shipping_address": {
                "address1": "789 Ocean Dr", "address2": "",
                "city": "Miami", "state": "FL", "zip": "33101", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "9012", "amount": 55.99},
            "created_at": "2026-06-09T16:20:00",
        },
        {
            "order_id": "O007",
            "user_id": "U004",
            "status": "delivered",
            "items": [
                {"item_id": "I015", "product_id": "P005", "name": "Winter Insulated Jacket (Black / M)", "quantity": 1, "unit_price": 189.99},
            ],
            "shipping_address": {
                "address1": "321 Michigan Ave", "address2": "Unit 5",
                "city": "Chicago", "state": "IL", "zip": "60601", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "3456", "amount": 207.99},
            "created_at": "2026-05-08T08:00:00",
            "delivered_at": "2026-05-14T12:30:00",
        },
        {
            "order_id": "O008",
            "user_id": "U004",
            "status": "pending",
            "items": [
                {"item_id": "I028", "product_id": "P010", "name": "Python Cookbook (3rd Ed., Paperback)", "quantity": 1, "unit_price": 49.99},
                {"item_id": "I008", "product_id": "P003", "name": "Portable Bluetooth Speaker (Blue)", "quantity": 1, "unit_price": 49.99},
            ],
            "shipping_address": {
                "address1": "321 Michigan Ave", "address2": "Unit 5",
                "city": "Chicago", "state": "IL", "zip": "60601", "country": "USA",
            },
            "payment": {"method": "gift_card", "last_four": "9999", "amount": 111.98},
            "created_at": "2026-06-07T13:10:00",
        },
        {
            "order_id": "O009",
            "user_id": "U005",
            "status": "delivered",
            "items": [
                {"item_id": "I026", "product_id": "P009", "name": "Cordless Stick Vacuum (Standard)", "quantity": 1, "unit_price": 249.99},
            ],
            "shipping_address": {
                "address1": "654 Pine St", "address2": "",
                "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "7890", "amount": 272.99},
            "created_at": "2026-05-18T17:40:00",
            "delivered_at": "2026-05-25T10:15:00",
        },
        {
            "order_id": "O010",
            "user_id": "U005",
            "status": "cancelled",
            "items": [
                {"item_id": "I032", "product_id": "P012", "name": "Ergonomic Office Chair (Black / Standard)", "quantity": 1, "unit_price": 349.99},
            ],
            "shipping_address": {
                "address1": "654 Pine St", "address2": "",
                "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "7890", "amount": 379.99},
            "created_at": "2026-06-01T10:30:00",
            "cancelled_at": "2026-06-02T09:15:00",
        },
        {
            "order_id": "O011",
            "user_id": "U006",
            "status": "shipped",
            "items": [
                {"item_id": "I006", "product_id": "P002", "name": "Smart Fitness Watch (Rose Gold/Small)", "quantity": 1, "unit_price": 219.99},
            ],
            "shipping_address": {
                "address1": "87 Beacon St", "address2": "",
                "city": "Boston", "state": "MA", "zip": "02108", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "2468", "amount": 240.99},
            "created_at": "2026-06-04T12:00:00",
            "shipped_at": "2026-06-05T14:30:00",
        },
        {
            "order_id": "O012",
            "user_id": "U007",
            "status": "delivered",
            "items": [
                {"item_id": "I019", "product_id": "P006", "name": "Premium Cotton T-Shirt (White / M)", "quantity": 3, "unit_price": 24.99},
            ],
            "shipping_address": {
                "address1": "1590 Broadway", "address2": "Suite 200",
                "city": "Denver", "state": "CO", "zip": "80202", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "1357", "amount": 82.97},
            "created_at": "2026-05-28T09:00:00",
            "delivered_at": "2026-06-03T16:20:00",
        },
        {
            "order_id": "O013",
            "user_id": "U008",
            "status": "pending",
            "items": [
                {"item_id": "I009", "product_id": "P003", "name": "Portable Bluetooth Speaker (Red)", "quantity": 2, "unit_price": 54.99},
                {"item_id": "I021", "product_id": "P006", "name": "Premium Cotton T-Shirt (Gray / L)", "quantity": 1, "unit_price": 24.99},
            ],
            "shipping_address": {
                "address1": "2000 Sunset Blvd", "address2": "Apt 12A",
                "city": "Los Angeles", "state": "CA", "zip": "90028", "country": "USA",
            },
            "payment": {"method": "paypal", "amount": 146.97},
            "created_at": "2026-06-10T08:45:00",
        },
        {
            "order_id": "O014",
            "user_id": "U010",
            "status": "delivered",
            "items": [
                {"item_id": "I002", "product_id": "P001", "name": "Wireless Noise-Cancelling Headphones (White)", "quantity": 1, "unit_price": 129.99},
                {"item_id": "I025", "product_id": "P008", "name": "Stainless Steel Toaster (Black)", "quantity": 1, "unit_price": 44.99},
            ],
            "shipping_address": {
                "address1": "1 E Washington St", "address2": "Floor 14",
                "city": "Phoenix", "state": "AZ", "zip": "85004", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "1122", "amount": 192.98},
            "created_at": "2026-05-25T15:10:00",
            "delivered_at": "2026-06-01T09:30:00",
        },
        {
            "order_id": "O015",
            "user_id": "U012",
            "status": "pending",
            "items": [
                {"item_id": "I004", "product_id": "P002", "name": "Smart Fitness Watch (Black/Small)", "quantity": 1, "unit_price": 199.99},
                {"item_id": "I019", "product_id": "P006", "name": "Premium Cotton T-Shirt (White / M)", "quantity": 1, "unit_price": 24.99},
            ],
            "shipping_address": {
                "address1": "1450 Ala Moana Blvd", "address2": "Apt 2305",
                "city": "Honolulu", "state": "HI", "zip": "96814", "country": "USA",
            },
            "payment": {"method": "credit_card", "last_four": "5566", "amount": 246.98},
            "created_at": "2026-06-09T20:00:00",
        },
    ]


def _seed_airline_users() -> list[dict[str, Any]]:
    return [
        {
            "user_id": "UA001", "name": "Alice Chen", "first_name": "Alice", "last_name": "Chen",
            "email": "alice.chen@email.com", "phone": "415-555-1001",
            "address": {"address1": "123 Main St", "city": "San Francisco", "state": "CA", "zip": "94102"},
            "loyalty_tier": "gold", "loyalty_points": 45200,
            "known_traveler_number": "KTN123456", "passport": "US-A1234567",
        },
        {
            "user_id": "UA002", "name": "Bob Williams", "first_name": "Bob", "last_name": "Williams",
            "email": "bob.w@email.com", "phone": "212-555-1002",
            "address": {"address1": "456 Park Ave", "city": "New York", "state": "NY", "zip": "10001"},
            "loyalty_tier": "silver", "loyalty_points": 18500,
            "known_traveler_number": "KTN789012", "passport": "US-B2345678",
        },
        {
            "user_id": "UA003", "name": "Carlos Rodriguez", "first_name": "Carlos", "last_name": "Rodriguez",
            "email": "carlos.r@email.com", "phone": "305-555-1003",
            "address": {"address1": "789 Ocean Dr", "city": "Miami", "state": "FL", "zip": "33101"},
            "loyalty_tier": "platinum", "loyalty_points": 89300,
            "known_traveler_number": "KTN345678", "passport": "US-C3456789",
        },
        {
            "user_id": "UA004", "name": "Diana Park", "first_name": "Diana", "last_name": "Park",
            "email": "diana.park@email.com", "phone": "312-555-1004",
            "address": {"address1": "321 Michigan Ave", "city": "Chicago", "state": "IL", "zip": "60601"},
            "loyalty_tier": "basic", "loyalty_points": 3200,
            "passport": "US-D4567890",
        },
        {
            "user_id": "UA005", "name": "Edward Kim", "first_name": "Edward", "last_name": "Kim",
            "email": "ed.kim@email.com", "phone": "206-555-1005",
            "address": {"address1": "654 Pine St", "city": "Seattle", "state": "WA", "zip": "98101"},
            "loyalty_tier": "gold", "loyalty_points": 52100,
            "known_traveler_number": "KTN901234", "passport": "US-E5678901",
        },
        {
            "user_id": "UA006", "name": "Fatima Hassan", "first_name": "Fatima", "last_name": "Hassan",
            "email": "fatima.h@email.com", "phone": "617-555-1006",
            "address": {"address1": "87 Beacon St", "city": "Boston", "state": "MA", "zip": "02108"},
            "loyalty_tier": "basic", "loyalty_points": 8900,
            "passport": "US-F6789012",
        },
        {
            "user_id": "UA007", "name": "George Thompson", "first_name": "George", "last_name": "Thompson",
            "email": "george.t@email.com", "phone": "303-555-1007",
            "address": {"address1": "1590 Broadway", "city": "Denver", "state": "CO", "zip": "80202"},
            "loyalty_tier": "silver", "loyalty_points": 22400,
            "known_traveler_number": "KTN567890", "passport": "US-G7890123",
        },
        {
            "user_id": "UA008", "name": "Hannah Lee", "first_name": "Hannah", "last_name": "Lee",
            "email": "hannah.lee@email.com", "phone": "213-555-1008",
            "address": {"address1": "2000 Sunset Blvd", "city": "Los Angeles", "state": "CA", "zip": "90028"},
            "loyalty_tier": "platinum", "loyalty_points": 97800,
            "known_traveler_number": "KTN112233", "passport": "US-H8901234",
        },
        {
            "user_id": "UA009", "name": "Ivan Petrov", "first_name": "Ivan", "last_name": "Petrov",
            "email": "ivan.p@email.com", "phone": "832-555-1009",
            "address": {"address1": "4100 Westheimer Rd", "city": "Houston", "state": "TX", "zip": "77027"},
            "loyalty_tier": "basic", "loyalty_points": 1200,
            "passport": "US-I9012345",
        },
        {
            "user_id": "UA010", "name": "Julia Martinez", "first_name": "Julia", "last_name": "Martinez",
            "email": "julia.m@email.com", "phone": "602-555-1010",
            "address": {"address1": "1 E Washington St", "city": "Phoenix", "state": "AZ", "zip": "85004"},
            "loyalty_tier": "silver", "loyalty_points": 19800,
            "known_traveler_number": "KTN445566", "passport": "US-J0123456",
        },
        {
            "user_id": "UA011", "name": "Kevin O'Brien", "first_name": "Kevin", "last_name": "O'Brien",
            "email": "kevin.ob@email.com", "phone": "503-555-1011",
            "address": {"address1": "720 SW Broadway", "city": "Portland", "state": "OR", "zip": "97205"},
            "loyalty_tier": "basic", "loyalty_points": 5600,
            "passport": "US-K1234567",
        },
        {
            "user_id": "UA012", "name": "Lisa Nakamura", "first_name": "Lisa", "last_name": "Nakamura",
            "email": "lisa.n@email.com", "phone": "808-555-1012",
            "address": {"address1": "1450 Ala Moana Blvd", "city": "Honolulu", "state": "HI", "zip": "96814"},
            "loyalty_tier": "gold", "loyalty_points": 63400,
            "known_traveler_number": "KTN778899", "passport": "US-L2345678",
        },
        {
            "user_id": "UA013", "name": "Michael Brown", "first_name": "Michael", "last_name": "Brown",
            "email": "michael.b@email.com", "phone": "404-555-1013",
            "address": {"address1": "1200 Peachtree St", "city": "Atlanta", "state": "GA", "zip": "30309"},
            "loyalty_tier": "basic", "loyalty_points": 4100,
            "passport": "US-M3456789",
        },
        {
            "user_id": "UA014", "name": "Nancy Davis", "first_name": "Nancy", "last_name": "Davis",
            "email": "nancy.d@email.com", "phone": "214-555-1014",
            "address": {"address1": "300 Reunion Blvd", "city": "Dallas", "state": "TX", "zip": "75207"},
            "loyalty_tier": "silver", "loyalty_points": 27100,
            "known_traveler_number": "KTN990011", "passport": "US-N4567890",
        },
        {
            "user_id": "UA015", "name": "Oscar Garcia", "first_name": "Oscar", "last_name": "Garcia",
            "email": "oscar.g@email.com", "phone": "210-555-1015",
            "address": {"address1": "111 Alamo Plaza", "city": "San Antonio", "state": "TX", "zip": "78205"},
            "loyalty_tier": "gold", "loyalty_points": 56700,
            "known_traveler_number": "KTN223344", "passport": "US-O5678901",
        },
    ]


def _seed_airline_airports() -> dict[str, dict[str, str]]:
    return {
        "SFO": {"code": "SFO", "name": "San Francisco International Airport", "city": "San Francisco", "state": "CA", "country": "USA"},
        "LAX": {"code": "LAX", "name": "Los Angeles International Airport", "city": "Los Angeles", "state": "CA", "country": "USA"},
        "JFK": {"code": "JFK", "name": "John F. Kennedy International Airport", "city": "New York", "state": "NY", "country": "USA"},
        "ORD": {"code": "ORD", "name": "O'Hare International Airport", "city": "Chicago", "state": "IL", "country": "USA"},
        "MIA": {"code": "MIA", "name": "Miami International Airport", "city": "Miami", "state": "FL", "country": "USA"},
        "SEA": {"code": "SEA", "name": "Seattle-Tacoma International Airport", "city": "Seattle", "state": "WA", "country": "USA"},
        "DFW": {"code": "DFW", "name": "Dallas/Fort Worth International Airport", "city": "Dallas", "state": "TX", "country": "USA"},
        "ATL": {"code": "ATL", "name": "Hartsfield-Jackson Atlanta International Airport", "city": "Atlanta", "state": "GA", "country": "USA"},
        "BOS": {"code": "BOS", "name": "Logan International Airport", "city": "Boston", "state": "MA", "country": "USA"},
        "DEN": {"code": "DEN", "name": "Denver International Airport", "city": "Denver", "state": "CO", "country": "USA"},
        "IAD": {"code": "IAD", "name": "Washington Dulles International Airport", "city": "Washington", "state": "DC", "country": "USA"},
        "PHX": {"code": "PHX", "name": "Phoenix Sky Harbor International Airport", "city": "Phoenix", "state": "AZ", "country": "USA"},
        "LAS": {"code": "LAS", "name": "Harry Reid International Airport", "city": "Las Vegas", "state": "NV", "country": "USA"},
        "PDX": {"code": "PDX", "name": "Portland International Airport", "city": "Portland", "state": "OR", "country": "USA"},
        "SAN": {"code": "SAN", "name": "San Diego International Airport", "city": "San Diego", "state": "CA", "country": "USA"},
        "HNL": {"code": "HNL", "name": "Daniel K. Inouye International Airport", "city": "Honolulu", "state": "HI", "country": "USA"},
        "LHR": {"code": "LHR", "name": "London Heathrow Airport", "city": "London", "state": "", "country": "UK"},
        "CDG": {"code": "CDG", "name": "Charles de Gaulle Airport", "city": "Paris", "state": "", "country": "France"},
        "NRT": {"code": "NRT", "name": "Narita International Airport", "city": "Tokyo", "state": "", "country": "Japan"},
        "SYD": {"code": "SYD", "name": "Sydney Kingsford Smith Airport", "city": "Sydney", "state": "NSW", "country": "Australia"},
    }


def _seed_airline_flights() -> list[dict[str, Any]]:
    """Generate flights with available seats for June 2026 dates."""
    dates = [f"2026-06-{d:02d}" for d in range(10, 25)]
    return [
        # --- Domestic ---
        {"flight_number": "AA101", "origin": "SFO", "destination": "JFK", "departure_time": "08:00", "arrival_time": "16:30",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 6, "economy": 42, "basic_economy": 20} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 850, "economy": 320, "basic_economy": 200}},
        {"flight_number": "DL202", "origin": "SFO", "destination": "JFK", "departure_time": "13:30", "arrival_time": "21:55",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 4, "economy": 55, "basic_economy": 25} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 800, "economy": 300, "basic_economy": 190}},
        {"flight_number": "UA303", "origin": "SFO", "destination": "LAX", "departure_time": "07:15", "arrival_time": "08:45",
         "airline": "United", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 8, "economy": 60, "basic_economy": 30} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 250, "economy": 120, "basic_economy": 80}},
        {"flight_number": "WN404", "origin": "SFO", "destination": "LAX", "departure_time": "16:00", "arrival_time": "17:30",
         "airline": "Southwest", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 0, "economy": 48, "basic_economy": 0} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"economy": 110}},
        {"flight_number": "AA505", "origin": "LAX", "destination": "JFK", "departure_time": "09:00", "arrival_time": "17:30",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 5, "economy": 50, "basic_economy": 22} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 820, "economy": 350, "basic_economy": 220}},
        {"flight_number": "DL606", "origin": "LAX", "destination": "JFK", "departure_time": "22:30", "arrival_time": "06:45+1",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 6, "economy": 45, "basic_economy": 18} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 780, "economy": 310, "basic_economy": 190}},
        {"flight_number": "UA909", "origin": "SFO", "destination": "ORD", "departure_time": "10:45", "arrival_time": "17:00",
         "airline": "United", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 4, "economy": 52, "basic_economy": 24} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 620, "economy": 260, "basic_economy": 160}},
        {"flight_number": "AA010", "origin": "SFO", "destination": "ORD", "departure_time": "14:15", "arrival_time": "20:30",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 3, "economy": 48, "basic_economy": 20} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 600, "economy": 250, "basic_economy": 155}},
        {"flight_number": "AA111", "origin": "ORD", "destination": "MIA", "departure_time": "07:00", "arrival_time": "11:00",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 6, "economy": 55, "basic_economy": 25} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 480, "economy": 200, "basic_economy": 130}},
        {"flight_number": "DL212", "origin": "ORD", "destination": "MIA", "departure_time": "15:30", "arrival_time": "19:30",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 5, "economy": 50, "basic_economy": 20} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 460, "economy": 190, "basic_economy": 120}},
        {"flight_number": "AS313", "origin": "SEA", "destination": "SFO", "departure_time": "08:30", "arrival_time": "10:45",
         "airline": "Alaska Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 6, "economy": 58, "basic_economy": 26} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 350, "economy": 150, "basic_economy": 95}},
        {"flight_number": "UA414", "origin": "SEA", "destination": "SFO", "departure_time": "17:00", "arrival_time": "19:15",
         "airline": "United", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 4, "economy": 52, "basic_economy": 22} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 340, "economy": 145, "basic_economy": 90}},
        {"flight_number": "AA515", "origin": "DFW", "destination": "ATL", "departure_time": "09:30", "arrival_time": "12:45",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 5, "economy": 60, "basic_economy": 28} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 420, "economy": 180, "basic_economy": 110}},
        {"flight_number": "DL616", "origin": "DFW", "destination": "ATL", "departure_time": "16:00", "arrival_time": "19:15",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 6, "economy": 55, "basic_economy": 25} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 400, "economy": 170, "basic_economy": 105}},
        {"flight_number": "UA717", "origin": "BOS", "destination": "DEN", "departure_time": "11:00", "arrival_time": "13:30",
         "airline": "United", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 4, "economy": 48, "basic_economy": 22} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 520, "economy": 230, "basic_economy": 140}},
        {"flight_number": "WN818", "origin": "BOS", "destination": "DEN", "departure_time": "17:30", "arrival_time": "20:00",
         "airline": "Southwest", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 0, "economy": 44, "basic_economy": 0} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"economy": 210}},
        {"flight_number": "AA919", "origin": "MIA", "destination": "JFK", "departure_time": "06:30", "arrival_time": "09:30",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 5, "economy": 50, "basic_economy": 20} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 500, "economy": 220, "basic_economy": 140}},
        {"flight_number": "B6020", "origin": "MIA", "destination": "JFK", "departure_time": "14:00", "arrival_time": "17:00",
         "airline": "JetBlue", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 0, "economy": 48, "basic_economy": 0} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"economy": 200}},
        {"flight_number": "AA212", "origin": "JFK", "destination": "ORD", "departure_time": "08:00", "arrival_time": "09:45",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 5, "economy": 55, "basic_economy": 25} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 450, "economy": 190, "basic_economy": 115}},
        {"flight_number": "DL303", "origin": "JFK", "destination": "ORD", "departure_time": "15:00", "arrival_time": "16:45",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 4, "economy": 48, "basic_economy": 22} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 430, "economy": 180, "basic_economy": 110}},
        {"flight_number": "UA555", "origin": "DEN", "destination": "SFO", "departure_time": "09:00", "arrival_time": "10:45",
         "airline": "United", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 4, "economy": 50, "basic_economy": 22} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 400, "economy": 170, "basic_economy": 105}},
        {"flight_number": "WN123", "origin": "DEN", "destination": "SFO", "departure_time": "18:30", "arrival_time": "20:15",
         "airline": "Southwest", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 0, "economy": 42, "basic_economy": 0} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"economy": 160}},
        # --- International ---
        {"flight_number": "BA707", "origin": "JFK", "destination": "LHR", "departure_time": "19:00", "arrival_time": "07:15+1",
         "airline": "British Airways", "dates": dates[::2], "cabin": "economy",
         "available_seats": {d: {"business": 10, "economy": 80, "basic_economy": 30} for d in dates[::2]},
         "status": {d: "on_time" for d in dates[::2]}, "price": {"business": 3200, "economy": 650, "basic_economy": 420}},
        {"flight_number": "AA808", "origin": "JFK", "destination": "LHR", "departure_time": "21:00", "arrival_time": "09:15+1",
         "airline": "American Airlines", "dates": dates[::2], "cabin": "economy",
         "available_seats": {d: {"business": 8, "economy": 75, "basic_economy": 25} for d in dates[::2]},
         "status": {d: "on_time" for d in dates[::2]}, "price": {"business": 3000, "economy": 600, "basic_economy": 390}},
        {"flight_number": "AF111", "origin": "JFK", "destination": "CDG", "departure_time": "17:30", "arrival_time": "06:45+1",
         "airline": "Air France", "dates": dates[::3], "cabin": "economy",
         "available_seats": {d: {"business": 12, "economy": 90, "basic_economy": 35} for d in dates[::3]},
         "status": {d: "on_time" for d in dates[::3]}, "price": {"business": 3500, "economy": 700, "basic_economy": 450}},
        {"flight_number": "DL404", "origin": "LAX", "destination": "HNL", "departure_time": "10:00", "arrival_time": "13:30",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 8, "economy": 65, "basic_economy": 28} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 750, "economy": 350, "basic_economy": 220}},
        {"flight_number": "UA888", "origin": "SFO", "destination": "NRT", "departure_time": "11:30", "arrival_time": "15:00+1",
         "airline": "United", "dates": dates[::4], "cabin": "economy",
         "available_seats": {d: {"business": 14, "economy": 95, "basic_economy": 40} for d in dates[::4]},
         "status": {d: "on_time" for d in dates[::4]}, "price": {"business": 4200, "economy": 850, "basic_economy": 550}},
        {"flight_number": "QF101", "origin": "LAX", "destination": "SYD", "departure_time": "23:00", "arrival_time": "07:30+2",
         "airline": "Qantas", "dates": dates[::5], "cabin": "economy",
         "available_seats": {d: {"business": 16, "economy": 100, "basic_economy": 45} for d in dates[::5]},
         "status": {d: "on_time" for d in dates[::5]}, "price": {"business": 5000, "economy": 950, "basic_economy": 620}},
        {"flight_number": "DL505", "origin": "ATL", "destination": "LAX", "departure_time": "08:00", "arrival_time": "10:00",
         "airline": "Delta", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 6, "economy": 58, "basic_economy": 26} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 550, "economy": 240, "basic_economy": 150}},
        {"flight_number": "AA303", "origin": "ATL", "destination": "LAX", "departure_time": "14:30", "arrival_time": "16:30",
         "airline": "American Airlines", "dates": dates[:], "cabin": "economy",
         "available_seats": {d: {"business": 7, "economy": 62, "basic_economy": 28} for d in dates},
         "status": {d: "on_time" for d in dates}, "price": {"business": 530, "economy": 230, "basic_economy": 145}},
    ]


def _seed_airline_reservations() -> list[dict[str, Any]]:
    return [
        {
            "reservation_id": "RA001", "user_id": "UA001",
            "flights": [
                {"flight_number": "AA101", "date": "2026-06-15", "origin": "SFO", "destination": "JFK",
                 "cabin": "business", "fare": 850.00},
                {"flight_number": "AA101", "date": "2026-06-20", "origin": "JFK", "destination": "SFO",
                 "cabin": "business", "fare": 850.00},
            ],
            "passengers": [
                {"name": "Alice Chen", "dob": "1990-03-15"},
            ],
            "bags": {"total": 2, "nonfree": 1},
            "payment": {"method": "credit_card", "last_four": "1234", "amount": 1700.00},
            "status": "confirmed", "created_at": "2026-06-01T10:30:00",
        },
        {
            "reservation_id": "RA002", "user_id": "UA002",
            "flights": [
                {"flight_number": "DL202", "date": "2026-06-12", "origin": "JFK", "destination": "SFO",
                 "cabin": "economy", "fare": 300.00},
            ],
            "passengers": [
                {"name": "Bob Williams", "dob": "1988-07-22"},
                {"name": "Sarah Williams", "dob": "1990-11-08"},
            ],
            "bags": {"total": 3, "nonfree": 1},
            "payment": {"method": "credit_card", "last_four": "5678", "amount": 640.00},
            "status": "confirmed", "created_at": "2026-06-05T14:20:00",
        },
        {
            "reservation_id": "RA003", "user_id": "UA003",
            "flights": [
                {"flight_number": "BA707", "date": "2026-06-18", "origin": "JFK", "destination": "LHR",
                 "cabin": "business", "fare": 3200.00},
            ],
            "passengers": [
                {"name": "Carlos Rodriguez", "dob": "1985-05-10"},
            ],
            "bags": {"total": 2, "nonfree": 0},
            "payment": {"method": "credit_card", "last_four": "9012", "amount": 3200.00},
            "status": "confirmed", "created_at": "2026-05-28T09:00:00",
        },
        {
            "reservation_id": "RA004", "user_id": "UA005",
            "flights": [
                {"flight_number": "UA909", "date": "2026-06-11", "origin": "SFO", "destination": "ORD",
                 "cabin": "economy", "fare": 260.00},
                {"flight_number": "AA111", "date": "2026-06-11", "origin": "ORD", "destination": "MIA",
                 "cabin": "economy", "fare": 200.00},
            ],
            "passengers": [
                {"name": "Edward Kim", "dob": "1992-01-30"},
            ],
            "bags": {"total": 1, "nonfree": 0},
            "payment": {"method": "credit_card", "last_four": "7890", "amount": 480.00},
            "status": "confirmed", "created_at": "2026-06-02T16:45:00",
        },
        {
            "reservation_id": "RA005", "user_id": "UA008",
            "flights": [
                {"flight_number": "DL606", "date": "2026-06-16", "origin": "LAX", "destination": "JFK",
                 "cabin": "business", "fare": 780.00},
            ],
            "passengers": [
                {"name": "Hannah Lee", "dob": "1991-09-12"},
                {"name": "David Lee", "dob": "1990-04-25"},
                {"name": "Emma Lee", "dob": "2020-02-18"},
            ],
            "bags": {"total": 4, "nonfree": 2},
            "payment": {"method": "credit_card", "last_four": "8642", "amount": 2370.00},
            "status": "confirmed", "created_at": "2026-06-03T11:30:00",
        },
        {
            "reservation_id": "RA006", "user_id": "UA012",
            "flights": [
                {"flight_number": "DL404", "date": "2026-06-14", "origin": "LAX", "destination": "HNL",
                 "cabin": "economy", "fare": 350.00},
                {"flight_number": "DL404", "date": "2026-06-21", "origin": "HNL", "destination": "LAX",
                 "cabin": "economy", "fare": 350.00},
            ],
            "passengers": [
                {"name": "Lisa Nakamura", "dob": "1987-06-03"},
            ],
            "bags": {"total": 1, "nonfree": 0},
            "payment": {"method": "credit_card", "last_four": "5566", "amount": 700.00},
            "status": "confirmed", "created_at": "2026-05-30T08:15:00",
        },
        {
            "reservation_id": "RA007", "user_id": "UA003",
            "flights": [
                {"flight_number": "UA303", "date": "2026-06-10", "origin": "SFO", "destination": "LAX",
                 "cabin": "basic_economy", "fare": 80.00},
            ],
            "passengers": [
                {"name": "Carlos Rodriguez", "dob": "1985-05-10"},
            ],
            "bags": {"total": 1, "nonfree": 1},
            "payment": {"method": "credit_card", "last_four": "9012", "amount": 115.00},
            "status": "cancelled", "created_at": "2026-06-04T07:00:00",
            "cancelled_at": "2026-06-05T12:00:00",
        },
        {
            "reservation_id": "RA008", "user_id": "UA010",
            "flights": [
                {"flight_number": "AA515", "date": "2026-06-13", "origin": "DFW", "destination": "ATL",
                 "cabin": "economy", "fare": 180.00},
            ],
            "passengers": [
                {"name": "Julia Martinez", "dob": "1993-12-20"},
            ],
            "bags": {"total": 2, "nonfree": 1},
            "payment": {"method": "credit_card", "last_four": "1122", "amount": 215.00},
            "status": "confirmed", "created_at": "2026-06-06T13:45:00",
        },
        {
            "reservation_id": "RA009", "user_id": "UA005",
            "flights": [
                {"flight_number": "AS313", "date": "2026-06-17", "origin": "SEA", "destination": "SFO",
                 "cabin": "economy", "fare": 150.00},
            ],
            "passengers": [
                {"name": "Edward Kim", "dob": "1992-01-30"},
            ],
            "bags": {"total": 1, "nonfree": 0},
            "payment": {"method": "credit_card", "last_four": "7890", "amount": 150.00},
            "status": "confirmed", "created_at": "2026-06-08T10:00:00",
        },
        {
            "reservation_id": "RA010", "user_id": "UA007",
            "flights": [
                {"flight_number": "UA717", "date": "2026-06-15", "origin": "BOS", "destination": "DEN",
                 "cabin": "economy", "fare": 230.00},
            ],
            "passengers": [
                {"name": "George Thompson", "dob": "1983-08-14"},
                {"name": "Maria Thompson", "dob": "1985-02-28"},
            ],
            "bags": {"total": 2, "nonfree": 0},
            "payment": {"method": "credit_card", "last_four": "1357", "amount": 460.00},
            "status": "confirmed", "created_at": "2026-06-01T15:30:00",
        },
        {
            "reservation_id": "RA011", "user_id": "UA004",
            "flights": [
                {"flight_number": "AA111", "date": "2026-06-12", "origin": "ORD", "destination": "MIA",
                 "cabin": "basic_economy", "fare": 130.00},
            ],
            "passengers": [
                {"name": "Diana Park", "dob": "1995-04-05"},
            ],
            "bags": {"total": 1, "nonfree": 1},
            "payment": {"method": "credit_card", "last_four": "3456", "amount": 165.00},
            "status": "confirmed", "created_at": "2026-06-07T09:20:00",
        },
        {
            "reservation_id": "RA012", "user_id": "UA014",
            "flights": [
                {"flight_number": "AA101", "date": "2026-06-20", "origin": "SFO", "destination": "JFK",
                 "cabin": "economy", "fare": 320.00},
            ],
            "passengers": [
                {"name": "Nancy Davis", "dob": "1989-10-12"},
            ],
            "bags": {"total": 2, "nonfree": 0},
            "payment": {"method": "credit_card", "last_four": "2233", "amount": 320.00},
            "status": "confirmed", "created_at": "2026-06-09T11:00:00"},
    ]


def _seed_airline_db() -> dict[str, Any]:
    """Build a fresh airline DB state."""
    airports = _seed_airline_airports()
    flights_list = _seed_airline_flights()
    return {
        "users": _seed_airline_users(),
        "airports": airports,
        "flights": {f["flight_number"]: f for f in flights_list},
        "reservations": _seed_airline_reservations(),
    }


def _seed_retail_db() -> dict[str, Any]:
    """Build a fresh retail DB state."""
    return {
        "users": _seed_retail_users(),
        "products": _seed_retail_products(),
        "orders": _seed_retail_orders(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_seed_lock = threading.Lock()


def seed(scenario: str) -> dict[str, Any]:
    """Return a **freshly-constructed** DB state dict for the given scenario.

    Args:
        scenario: ``"retail"`` or ``"airline"``.

    Returns:
        A new dict with users, orders/products/flights/reservations tables.

    Raises:
        ValueError: If scenario is not recognised.
    """
    scenario = scenario.strip().lower()
    if scenario == "retail":
        return _seed_retail_db()
    elif scenario == "airline":
        return _seed_airline_db()
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Expected 'retail' or 'airline'.")


class MockDatabase:
    """Thread-safe in-memory database for tau-bench mock tools.

    Wraps a mutable dict state with a ``threading.Lock``.  Tools execute
    within the lock so concurrent access is safe.

    Usage::

        db = MockDatabase("retail")
        result = db.execute(find_user, {"first_name": "Alice", "last_name": "Chen", "zip": "94102"})
        db.save("retail_snapshot.json")
        db2 = MockDatabase.load("retail_snapshot.json")
    """

    def __init__(self, scenario: str = "retail"):
        self._state = seed(scenario)
        self._lock = threading.Lock()

    @property
    def state(self) -> dict[str, Any]:
        """Return a reference to the internal state dict.

        *Caution*: the reference is not guarded — do not mutate it outside
        of ``execute()``.
        """
        return self._state

    def execute(self, tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute *tool* against the database state under the lock.

        Args:
            tool: A ``MockTool`` instance (must have an ``execute`` method).
            arguments: Dict of parameter values matching the tool's JSON Schema.

        Returns:
            The tool's result dict.
        """
        with self._lock:
            return tool.execute(arguments, self._state)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full database state to a JSON string."""
        with self._lock:
            return json.dumps(self._state, indent=indent, default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str, scenario: str = "retail") -> "MockDatabase":
        """Create a MockDatabase from a previously-serialised state."""
        db = cls.__new__(cls)
        db._state = json.loads(data)
        db._lock = threading.Lock()
        # We ignore the scenario param — the loaded data determines the tables.
        return db

    def save(self, path: str | Path) -> None:
        """Write the database state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MockDatabase":
        """Create a MockDatabase by loading state from a JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
