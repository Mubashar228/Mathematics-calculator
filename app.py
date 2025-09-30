# app.py
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.calculus.util import continuous_domain
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)

# ---------------- Config ----------------
st.set_page_config(page_title="Math Master Pro", layout="wide")
st.title("🧮 Math Master Pro – Advanced Calculator")

transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))

def safe_parse(s):
    try:
        return parse_expr(s, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Parse error: {e}")

# ---------------- Sidebar Symbols ----------------
st.sidebar.title("Math Symbols")
symbols = {
    "+": "+",
    "−": "-",
    "×": "*",
    "÷": "/",
    "√": "sqrt()",
    "∛": "cbrt()",
    "x²": "**2",
    "x³": "**3",
    "π": "pi",
    "sin": "sin()",
    "cos": "cos()",
    "tan": "tan()",
    "log": "log()",
    "ln": "ln()",
    "e": "E"
}
for label, symbol in symbols.items():
    st.sidebar.write(f"**{label}** → `{symbol}`")

st.sidebar.info("📌 Copy symbols from above and paste in input boxes")

# ---------------- Functions ----------------
def arithmetic_solver(expr_str):
    try:
        steps = []
        expr = safe_parse(expr_str)
        steps.append(f"Expression entered: {expr}")
        simplified = sp.simplify(expr)
        steps.append(f"Simplified form: {simplified}")
        numeric_val = sp.N(expr)
        steps.append(f"Numeric value: {numeric_val}")
        return steps
    except Exception as e:
        return [f"Error: {e}"]

def solve_and_plot_equation(eq_str, var="x", x_min=-10, x_max=10):
    x = sp.Symbol(var)
    if "=" in eq_str:
        left_str, right_str = eq_str.split("=")
        left = safe_parse(left_str)
        right = safe_parse(right_str)
    else:
        left = safe_parse(eq_str)
        right = 0
    expr = sp.simplify(left - right)
    solutions = sp.solve(sp.Eq(expr, 0), x)

    # Plot
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(x_min, x_max, 500)
    ys = f(xs)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(xs, ys, label=f"{sp.latex(expr)} = 0")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    # Mark solutions
    real_solutions = []
    for sol in solutions:
        try:
            sol_val = float(sp.N(sol))
            if x_min <= sol_val <= x_max:
                real_solutions.append(sol_val)
                ax.scatter(sol_val, 0, color="red", zorder=5)
        except:
            pass

    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    return solutions, real_solutions

def plot_trig(expr_str, var="x", xmin=-2*np.pi, xmax=2*np.pi):
    x = sp.Symbol(var)
    expr = safe_parse(expr_str)
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(xmin, xmax, 500)
    ys = f(xs)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(xs, ys, label=str(expr))
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

def analyze_function(expr_str, var="x"):
    x = sp.Symbol(var)
    expr = safe_parse(expr_str)

    dom = continuous_domain(expr, x, sp.S.Reals)

    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(-10, 10, 500)
    ys = f(xs)
    rng = (np.nanmin(ys), np.nanmax(ys))

    deriv = sp.diff(expr, x)
    crit_points = sp.solve(sp.Eq(deriv, 0), x)
    increasing, decreasing = [], []
    for i in range(len(crit_points)-1):
        test_val = (crit_points[i] + crit_points[i+1]) / 2
        slope = deriv.subs(x, test_val)
        if slope > 0:
            increasing.append((crit_points[i], crit_points[i+1]))
        else:
            decreasing.append((crit_points[i], crit_points[i+1]))

    return dom, rng, increasing, decreasing, deriv

# ---------------- Menu ----------------
menu = st.sidebar.radio("📂 Select Tool", [
    "Arithmetic Calculator",
    "Equation Solver (with Graph)",
    "Trigonometry & Graphs",
    "Function Analysis",
    "Derivative",
    "Integral",
    "Plot Function"
])

# ---------------- Pages ----------------
if menu == "Arithmetic Calculator":
    st.header("➗ Arithmetic Calculator (Step-by-Step)")
    expr = st.text_input("Enter arithmetic expression", "2+3*sqrt(16)/2")
    if st.button("Solve Arithmetic"):
        steps = arithmetic_solver(expr)
        for step in steps:
            st.write(step)

elif menu == "Equation Solver (with Graph)":
    st.header("📝 Equation Solver with Graph")
    eq = st.text_input("Enter equation", "x**2 - 4 = 0")
    if st.button("Solve Equation"):
        try:
            sols, real_sols = solve_and_plot_equation(eq)
            st.write("Solutions:", sols)
            if real_sols:
                st.success(f"Real roots in range: {real_sols}")
        except Exception as e:
            st.error(e)

elif menu == "Trigonometry & Graphs":
    st.header("📈 Trigonometry Functions & Graphs")
    expr = st.text_input("Enter trig function (e.g., sin(x), cos(x)+1)", "sin(x)")
    xmin = st.number_input("x min", value=-6.28)  # -2π
    xmax = st.number_input("x max", value=6.28)   # 2π
    if st.button("Plot Trigonometry Function"):
        try:
            plot_trig(expr, xmin=xmin, xmax=xmax)
        except Exception as e:
            st.error(e)

elif menu == "Function Analysis":
    st.header("📊 Function Analysis")
    expr = st.text_input("Enter function", "x**3 - 3*x")
    if st.button("Analyze Function"):
        try:
            dom, rng, inc, dec, deriv = analyze_function(expr)
            st.write("Domain:", dom)
            st.write("Approx Range:", rng)
            st.write("Derivative:", deriv)
            st.write("Increasing intervals:", inc)
            st.write("Decreasing intervals:", dec)
        except Exception as e:
            st.error(e)

elif menu == "Derivative":
    st.header("📐 Derivative Calculator")
    expr = st.text_input("Enter function for derivative", "x**2 + 3*x")
    if st.button("Find Derivative"):
        try:
            x = sp.Symbol("x")
            derivative = sp.diff(safe_parse(expr), x)
            st.write("Derivative:", derivative)
        except Exception as e:
            st.error(e)

elif menu == "Integral":
    st.header("∫ Integral Calculator")
    expr = st.text_input("Enter function for integral", "x**2")
    if st.button("Find Integral"):
        try:
            x = sp.Symbol("x")
            integral = sp.integrate(safe_parse(expr), x)
            st.write("Integral:", integral, "+ C")
        except Exception as e:
            st.error(e)

elif menu == "Plot Function":
    st.header("📉 Plot a Function")
    expr = st.text_input("Enter function", "x**2 - 4")
    xmin = st.number_input("x min", value=-10)
    xmax = st.number_input("x max", value=10)
    if st.button("Plot Function"):
        try:
            x = sp.Symbol("x")
            f = sp.lambdify(x, safe_parse(expr), "numpy")
            xs = np.linspace(xmin, xmax, 500)
            ys = f(xs)

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(xs, ys, label=expr)
            ax.axhline(0, color="black", lw=0.5)
            ax.axvline(0, color="black", lw=0.5)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(e)
