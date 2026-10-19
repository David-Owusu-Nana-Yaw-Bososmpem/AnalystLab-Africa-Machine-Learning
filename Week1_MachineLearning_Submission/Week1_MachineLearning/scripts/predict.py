"""
ShopSmart Week 1 - Simple Prediction Script

Loads the saved regression and classification pipelines and accepts user input
to predict Order_Total and Return Probability.

Run from the project root:
    python scripts/predict.py

Important: use scikit-learn==1.6.1, matching the environment used to save the models.
"""

from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

REGRESSION_FEATURES = [
    "Age", "Customer_Segment", "Category", "Brand", "Store_ID", "Region",
    "Payment_Method", "Sales_Channel", "Quantity", "Unit_Price", "Discount",
    "Shipping_Cost", "Order_Month", "Order_DayOfWeek"
]

CLASSIFICATION_FEATURES = [
    "Age", "Customer_Segment", "Category", "Brand", "Store_ID", "Region",
    "Payment_Method", "Sales_Channel", "Order_Status", "Quantity", "Unit_Price",
    "Discount", "Shipping_Cost", "Order_Total", "Order_Month", "Order_DayOfWeek"
]


def ask(prompt, cast=str):
    while True:
        value = input(prompt).strip()
        try:
            return cast(value)
        except ValueError:
            print("Invalid value. Please try again.")


def collect_inputs():
    print("\nEnter order/customer information.")
    values = {
        "Age": ask("Age: ", int),
        "Customer_Segment": ask("Customer segment (Bronze/Gold/Silver): "),
        "Category": ask("Product category: "),
        "Brand": ask("Brand: "),
        "Store_ID": ask("Store ID (e.g., STR001): "),
        "Region": ask("Region: "),
        "Payment_Method": ask("Payment method: "),
        "Sales_Channel": ask("Sales channel: "),
        "Order_Status": ask("Order status: "),
        "Quantity": ask("Quantity: ", int),
        "Unit_Price": ask("Unit price: ", float),
        "Discount": ask("Discount: ", float),
        "Shipping_Cost": ask("Shipping cost: ", float),
        "Order_Month": ask("Order month (1-12): ", int),
        "Order_DayOfWeek": ask("Order day of week (0-6): ", int),
    }
    return values


def main():
    linear_model = joblib.load(MODEL_DIR / "linear_regression.pkl")
    logistic_model = joblib.load(MODEL_DIR / "logistic_regression.pkl")

    values = collect_inputs()

    reg_row = pd.DataFrame([{k: values[k] for k in REGRESSION_FEATURES}])
    predicted_order_value = float(linear_model.predict(reg_row)[0])

    class_values = dict(values)
    class_values["Order_Total"] = predicted_order_value
    class_row = pd.DataFrame([{k: class_values[k] for k in CLASSIFICATION_FEATURES}])

    return_probability = float(logistic_model.predict_proba(class_row)[0, 1])
    return_prediction = int(logistic_model.predict(class_row)[0])

    print("\n" + "=" * 55)
    print("SHOPSMART PREDICTION")
    print("=" * 55)
    print(f"Predicted Order Value : {predicted_order_value:,.2f}")
    print(f"Return Probability    : {return_probability:.2%}")
    print(f"Predicted Return Flag : {return_prediction} "
          f"({'Returned' if return_prediction == 1 else 'Not Returned'})")
    print("=" * 55)


if __name__ == "__main__":
    main()
