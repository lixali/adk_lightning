# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Dict, List
import os

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm


def load_retail_policy():
    """Load the retail policy from the wiki.md file."""
    wiki_path = "/home/lixali_google_com/data/adk-lightning/tau-bench-lightning/tau_bench/envs/retail/wiki.md"
    try:
        with open(wiki_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Retail policy file not found. Please contact support."
    except Exception as e:
        return f"Error loading retail policy: {str(e)}"

def load_retail_data():
    """Load the retail database from JSON files."""
    data_path = "/home/lixali_google_com/data/adk-lightning/tau-bench-lightning/tau_bench/envs/retail/data"
    
    try:
        with open(f"{data_path}/users.json", 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        with open(f"{data_path}/products.json", 'r', encoding='utf-8') as f:
            products = json.load(f)
            
        with open(f"{data_path}/orders.json", 'r', encoding='utf-8') as f:
            orders = json.load(f)
            
        return {
            "users": users,
            "products": products,
            "orders": orders
        }
    except Exception as e:
        print(f"Error loading retail data: {e}")
        return {
            "users": {},
            "products": {},
            "orders": {}
        }

retail_wiki = load_retail_policy()
# Load the actual retail database
retail_data = load_retail_data()


# Retail Tool Functions (converted from tau-bench)

def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression.
    
    Args:
        expression: The mathematical expression to calculate, such as '2 + 2'. 
                   The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.
    
    Returns:
        The calculated result as a string.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return "Error: invalid characters in expression"
    try:
        # Evaluate the mathematical expression safely
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
    except Exception as e:
        return f"Error: {e}"


def cancel_pending_order(order_id: str, reason: str) -> str:
    """Cancel a pending order. If the order is already processed or delivered, 
    it cannot be cancelled. The agent needs to explain the cancellation detail 
    and ask for explicit user confirmation (yes/no) to proceed.
    
    Args:
        order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
        reason: The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.
    
    Returns:
        The order details after the cancellation.
    """
    # check order exists and is pending
    orders = retail_data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be cancelled"

    # check reason
    if reason not in ["no longer needed", "ordered by mistake"]:
        return "Error: invalid reason"

    # handle refund
    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment_id,
        }
        refunds.append(refund)
        if "gift_card" in payment_id:  # refund to gift card immediately
            payment_method = retail_data["users"][order["user_id"]]["payment_methods"][
                payment_id
            ]
            payment_method["balance"] += payment["amount"]
            payment_method["balance"] = round(payment_method["balance"], 2)

    # update order status
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)

    return json.dumps(order)


def exchange_delivered_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str) -> str:
    """Exchange items in a delivered order to new items of the same product type.
    
    Args:
        order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
        item_ids: The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.
        new_item_ids: The item ids to be exchanged for, each such as '1008292230'. 
        payment_method_id: The payment method id to pay or receive refund for the item price difference.
    
    Returns:
        The order details after the exchange.
    """
    # Validate inputs
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"
    
    if not item_ids:
        return "Error: no items specified for exchange"
    
    exchange_summary = {
        "order_id": order_id,
        "items_to_exchange": item_ids,
        "new_items": new_item_ids,
        "payment_method": payment_method_id,
        "status": "exchange requested"
    }
    
    return json.dumps(exchange_summary)


def find_user_id_by_email(email: str) -> str:
    """Find user id by email. If the user is not found, the function will return an error message.
    
    Args:
        email: The email of the user, such as 'something@example.com'.
    
    Returns:
        The user id or error message.
    """
    users = retail_data["users"]
    for user_id, profile in users.items():
        if profile["email"].lower() == email.lower():
            return user_id
    return "Error: user not found"


def find_user_id_by_name_zip(first_name: str, last_name: str, zip_code: str) -> str:
    """Find user id by first name, last name, and zip code.
    
    Args:
        first_name: The first name of the customer, such as 'John'.
        last_name: The last name of the customer, such as 'Doe'.
        zip_code: The zip code of the customer, such as '12345'.
    
    Returns:
        The user id or error message.
    """
    users = retail_data["users"]
    for user_id, profile in users.items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip_code
        ):
            return user_id
    return "Error: user not found"


def get_order_details(order_id: str) -> str:
    """Get the status and details of an order.
    
    Args:
        order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
    
    Returns:
        The order details as JSON string.
    """
    orders = retail_data["orders"]
    if order_id in orders:
        return json.dumps(orders[order_id])
    return "Error: order not found"


def get_product_details(product_id: str) -> str:
    """Get the inventory details of a product.
    
    Args:
        product_id: The product id, such as '6086499569'. Be careful the product id is different from the item id.
    
    Returns:
        The product details as JSON string.
    """
    products = retail_data["products"]
    if product_id in products:
        return json.dumps(products[product_id])
    return "Error: product not found"


def get_user_details(user_id: str) -> str:
    """Get the details of a user, including their orders.
    
    Args:
        user_id: The user id, such as 'sara_doe_496'.
    
    Returns:
        The user details as JSON string.
    """
    users = retail_data["users"]
    if user_id in users:
        return json.dumps(users[user_id])
    return "Error: user not found"


def list_all_product_types() -> str:
    """List the name and product id of all product types. Each product type has 
    a variety of different items with unique item ids and options.
    
    Returns:
        A JSON string with product names and their ids.
    """
    products = retail_data["products"]
    product_dict = {
        product["name"]: product["product_id"] for product in products.values()
    }
    product_dict = dict(sorted(product_dict.items()))
    return json.dumps(product_dict)


def modify_pending_order_address(order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip_code: str) -> str:
    """Modify the shipping address of a pending order.
    
    Args:
        order_id: The order id, such as '#W0000000'.
        address1: The first line of the address, such as '123 Main St'.
        address2: The second line of the address, such as 'Apt 1' or ''.
        city: The city, such as 'San Francisco'.
        state: The state, such as 'CA'.
        country: The country, such as 'USA'.
        zip_code: The zip code, such as '12345'.
    
    Returns:
        The updated order details.
    """
    # Check if the order exists and is pending
    orders = retail_data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Modify the address
    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip_code,
    }
    return json.dumps(order)


def modify_pending_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str) -> str:
    """Modify items in a pending order to new items of the same product type.
    
    Args:
        order_id: The order id, such as '#W0000000'.
        item_ids: The item ids to be modified.
        new_item_ids: The item ids to be modified for.
        payment_method_id: The payment method id to pay or receive refund.
    
    Returns:
        The updated order details.
    """
    products, orders, users = retail_data["products"], retail_data["orders"], retail_data["users"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the items to be modified exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # Check new items exist, match old items, and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    # Check if the payment method exists
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # If the new item is more expensive, check if the gift card has enough balance
    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return "Error: insufficient gift card balance to pay for the new item"

    # Handle the payment or refund
    order["payment_history"].append(
        {
            "transaction_type": "payment" if diff_price > 0 else "refund",
            "amount": abs(diff_price),
            "payment_method_id": payment_method_id,
        }
    )
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= diff_price
        payment_method["balance"] = round(payment_method["balance"], 2)

    # Modify the order
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        item["item_id"] = new_item_id
        item["price"] = products[item["product_id"]]["variants"][new_item_id]["price"]
        item["options"] = products[item["product_id"]]["variants"][new_item_id]["options"]
    order["status"] = "pending (item modified)"

    return json.dumps(order)


def modify_pending_order_payment(order_id: str, payment_method_id: str) -> str:
    """Modify the payment method of a pending order.
    
    Args:
        order_id: The order id, such as '#W0000000'.
        payment_method_id: The payment method id.
    
    Returns:
        The updated order details.
    """
    orders = retail_data["orders"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the payment method exists
    if payment_method_id not in retail_data["users"][order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # Check that the payment history should only have one payment
    if (
        len(order["payment_history"]) > 1
        or order["payment_history"][0]["transaction_type"] != "payment"
    ):
        return "Error: there should be exactly one payment for a pending order"

    # Check that the payment method is different
    if order["payment_history"][0]["payment_method_id"] == payment_method_id:
        return "Error: the new payment method should be different from the current one"

    amount = order["payment_history"][0]["amount"]
    payment_method = retail_data["users"][order["user_id"]]["payment_methods"][payment_method_id]

    # Check if the new payment method has enough balance if it is a gift card
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < amount
    ):
        return "Error: insufficient gift card balance to pay for the order"

    # Modify the payment method
    order["payment_history"].extend(
        [
            {
                "transaction_type": "payment",
                "amount": amount,
                "payment_method_id": payment_method_id,
            },
            {
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": order["payment_history"][0]["payment_method_id"],
            },
        ]
    )

    # If payment is made by gift card, update the balance
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= amount
        payment_method["balance"] = round(payment_method["balance"], 2)

    # If refund is made to a gift card, update the balance
    if "gift_card" in order["payment_history"][0]["payment_method_id"]:
        old_payment_method = retail_data["users"][order["user_id"]]["payment_methods"][
            order["payment_history"][0]["payment_method_id"]
        ]
        old_payment_method["balance"] += amount
        old_payment_method["balance"] = round(old_payment_method["balance"], 2)

    return json.dumps(order)


def modify_user_address(user_id: str, address1: str, address2: str, city: str, state: str, country: str, zip_code: str) -> str:
    """Modify the default address of a user.
    
    Args:
        user_id: The user id, such as 'sara_doe_496'.
        address1: The first line of the address.
        address2: The second line of the address.
        city: The city.
        state: The state.
        country: The country.
        zip_code: The zip code.
    
    Returns:
        The updated user details.
    """
    users = retail_data["users"]
    if user_id not in users:
        return "Error: user not found"
    user = users[user_id]
    user["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip_code,
    }
    return json.dumps(user)


def return_delivered_order_items(order_id: str, item_ids: List[str], payment_method_id: str) -> str:
    """Return some items of a delivered order.
    
    Args:
        order_id: The order id, such as '#W0000000'.
        item_ids: The item ids to be returned.
        payment_method_id: The payment method id for refund.
    
    Returns:
        The updated order details.
    """
    # Validate inputs
    if not order_id.startswith('#'):
        return "Error: order_id must start with '#' symbol"
    
    if not item_ids:
        return "Error: no items specified for return"
    
    if not payment_method_id:
        return "Error: payment_method_id is required for refund"
    
    
    return_details = {
        "order_id": order_id,
        "status": "return requested",
        "return_items": item_ids,
        "refund_method": payment_method_id,
        "message": "Return request has been processed. You will receive an email with return instructions."
    }
    
    return json.dumps(return_details)


def think(thought: str) -> str:
    """Use the tool to think about something. It will not obtain new information or change 
    the database, but just append the thought to the log.
    
    Args:
        thought: A thought to think about.
    
    Returns:
        Empty string.
    """
    return ""


def transfer_to_human_agents(summary: str) -> str:
    """Transfer the user to a human agent, with a summary of the user's issue.
    
    Args:
        summary: A summary of the user's issue.
    
    Returns:
        Transfer confirmation message.
    """
    return "Transfer successful"


root_agent = Agent(
    model=LiteLlm(model="ollama_chat/gemma3:12b"),
    name="retail_support_agent",
    description=(
        "Helpful customer service agent in retail domain with access to order management, user lookup, and product information tools."
    ),
    instruction=retail_wiki,
    tools=[
        calculate,
        cancel_pending_order,
        exchange_delivered_order_items,
        find_user_id_by_email,
        find_user_id_by_name_zip,
        get_order_details,
        get_product_details,
        get_user_details,
        list_all_product_types,
        modify_pending_order_address,
        modify_pending_order_items,
        modify_pending_order_payment,
        modify_user_address,
        return_delivered_order_items,
        think,
        transfer_to_human_agents,
    ],
)
