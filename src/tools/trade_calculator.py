import os
import time
import math
from typing import Literal
from src.tools.base import ToolResult


GBP_TO_CNY = 9.15
GBP_TO_EUR = 1.18
GBP_TO_USD = 1.27


def trade_calculator(
    operation: Literal["landed_cost", "currency_convert", "duty_calculation", "roi_projection", "margin_analysis"],
    params: dict,
) -> ToolResult:
    start = time.time()

    try:
        if operation == "currency_convert":
            amount = float(params.get("amount", 0))
            from_currency = params.get("from_currency", "CNY").upper()
            to_currency = params.get("to_currency", "GBP").upper()

            rates = {"GBP": 1.0, "CNY": 1 / GBP_TO_CNY, "EUR": 1 / GBP_TO_EUR, "USD": 1 / GBP_TO_USD}

            if from_currency not in rates or to_currency not in rates:
                return ToolResult(
                    call_id="local",
                    tool_name="trade_calculator",
                    status="error",
                    content={},
                    latency_ms=int((time.time() - start) * 1000),
                    error_message=f"Unsupported currency: {from_currency} or {to_currency}",
                )

            result = amount * rates[from_currency] / rates[to_currency]
            return ToolResult(
                call_id="local",
                tool_name="trade_calculator",
                status="success",
                content={
                    "amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "result": round(result, 2),
                    "rate_used": f"1 {from_currency} = {round(rates[from_currency]/rates[to_currency], 4)} {to_currency}",
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif operation == "landed_cost":
            fob_price = float(params.get("fob_price", 0) or params.get("fob_per_unit", 0))
            units = int(params.get("units", 0) or params.get("quantity", 0))
            freight = float(params.get("freight", 0))
            insurance_rate = float(params.get("insurance_rate", 0.005))
            duty_rate = float(params.get("duty_rate", 0))
            handling = float(params.get("handling", 0))
            customs_clearance = float(params.get("customs_clearance", 0))

            cif = (fob_price * units) + freight
            insurance = cif * (insurance_rate / 100)
            duty = cif * (duty_rate / 100)
            total = cif + insurance + duty + handling + customs_clearance

            return ToolResult(
                call_id="local",
                tool_name="trade_calculator",
                status="success",
                content={
                    "fob_price_per_unit": fob_price,
                    "units": units,
                    "cif_value": round(cif, 2),
                    "insurance": round(insurance, 2),
                    "uk_import_duty": round(duty, 2),
                    "handling": handling,
                    "customs_clearance": customs_clearance,
                    "total_landed_cost": round(total, 2),
                    "cost_per_unit": round(total / units, 2) if units else 0,
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif operation == "duty_calculation":
            commodity_value = float(params.get("commodity_value", 0))
            duty_rate = float(params.get("duty_rate", 0))
            duty = commodity_value * (duty_rate / 100)
            return ToolResult(
                call_id="local",
                tool_name="trade_calculator",
                status="success",
                content={
                    "commodity_value": commodity_value,
                    "duty_rate": duty_rate,
                    "uk_import_duty": round(duty, 2),
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif operation == "roi_projection":
            principal = float(params.get("principal", 0) or params.get("initial_investment", 0) or params.get("amount", 0))
            rate = float(params.get("annual_rate", 0) or params.get("annual_return_rate", 0) or params.get("rate", 0))
            years = int(params.get("years", 0) or params.get("investment_period_years", 1))

            future_value = principal * ((1 + rate / 100) ** years)
            total_return = future_value - principal

            return ToolResult(
                call_id="local",
                tool_name="trade_calculator",
                status="success",
                content={
                    "principal": principal,
                    "annual_rate": rate,
                    "years": years,
                    "future_value": round(future_value, 2),
                    "total_return": round(total_return, 2),
                    "total_return_pct": round((total_return / principal) * 100, 2) if principal else 0,
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif operation == "margin_analysis":
            selling_price = float(params.get("selling_price", 0))
            cogs = float(params.get("cogs", 0))
            duty_cost = float(params.get("duty_cost", 0))
            distribution = float(params.get("distribution", 0))

            total_cost = cogs + duty_cost + distribution
            margin = selling_price - total_cost
            margin_pct = (margin / selling_price * 100) if selling_price else 0

            return ToolResult(
                call_id="local",
                tool_name="trade_calculator",
                status="success",
                content={
                    "selling_price": selling_price,
                    "cogs": cogs,
                    "duty_cost": duty_cost,
                    "distribution": distribution,
                    "total_cost": round(total_cost, 2),
                    "gross_margin": round(margin, 2),
                    "margin_pct": round(margin_pct, 2),
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        else:
            return ToolResult(
                call_id="local",
                tool_name="trade_calculator",
                status="error",
                content={},
                latency_ms=int((time.time() - start) * 1000),
                error_message=f"Unknown operation: {operation}",
            )

    except Exception as e:
        return ToolResult(
            call_id="local",
            tool_name="trade_calculator",
            status="error",
            content={},
            latency_ms=int((time.time() - start) * 1000),
            error_message=str(e),
        )
