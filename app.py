# app.py
import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import (parse_expr,
                                        standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor)
transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))

st.set_page_config(page_title="Math Master — Algebra, Calculus, Trig, Sets & Graphs", layout="wide")

st.title("🔢 Math Master — Algebra, Calculus, Trig, Sets & Graphs")

st.markdown("""
This app solves arithmetic, algebra, trigonometry, calculus (derivative/integral), set operations,
profit & loss problems, and plots equations. Enter expressions in usual math form, e.g. `2*x+3`, `sin(x)`, `x**2 - 4 = 0`.
""")

# -----------------------
# Helper functions
# -----------------------
def safe_parse(s):
    """Parse a user expression into sympy expression safely with some transformations."""
    try:
        return parse_expr(s, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Could not parse expression: {e}")

def solve_equation(equation_str, symbol_list=None):
    """Solve equation_str (like 'x^2-1=0' or 'sin(x)-0.5=0')"""
    if "=" in equation_str:
        left, right = equation_str.split("=" ,1)
        left_e = safe_parse(left)
        right_e = safe_parse(right)
        eq = sp.Eq(left_e, right_e)
    else:
        eq = sp.Eq(safe_parse(equation_str), 0)
    # determine symbols
    syms = symbol_list or list(eq.free_symbols)
    if len(syms) == 0:
        solutions = sp.solve(eq, dict=True)
    elif len(syms) == 1:
        solutions = sp.solve(eq, syms[0], dict=True)
    else:
        # system? if multiple eqns (comma separated?) user should use system solver
        solutions = sp.solve(eq, syms)
    return solutions

def solve_system(eqs_text):
    """Solve system: input lines of equations separated by newline or ';'"""
    parts = [p.strip() for p in eqs_text.replace(";", "\n").splitlines() if p.strip()]
    eqs = []
    syms = set()
    for p in parts:
        if "=" in p:
            l, r = p.split("=",1)
            le = safe_parse(l); re = safe_parse(r)
            eqs.append(sp.Eq(le, re))
            syms |= set(le.free_symbols) | set(re.free_symbols)
        else:
            ex = safe_parse(p)
            eqs.append(sp.Eq(ex, 0))
            syms |= set(ex.free_symbols)
    syms = sorted(list(syms), key=lambda s: s.name)
    if len(eqs) == 0:
        return "No equations found"
    sol = sp.solve(eqs, syms, dict=True)
    return sol

def differentiate(expr_str, var_str="x", order=1):
    var = sp.Symbol(var_str)
    expr = safe_parse(expr_str)
    return sp.diff(expr, var, order)

def integrate(expr_str, var_str="x", lower=None, upper=None):
    var = sp.Symbol(var_str)
    expr = safe_parse(expr_str)
    if lower is None and upper is None:
        return sp.integrate(expr, var)
    else:
        low = safe_parse(str(lower))
        up = safe_parse(str(upper))
        return sp.integrate(expr, (var, low, up))

def numeric_eval(expr_str, subs=None):
    expr = safe_parse(expr_str)
    if subs:
        return float(expr.evalf(subs=subs))
    return float(expr.evalf())

def plot_functions(exprs, var='x', x_min=-10, x_max=10, points=500):
    x = sp.Symbol(var)
    xs = np.linspace(x_min, x_max, points)
    plt.figure(figsize=(8,4))
    for expr_text in exprs:
        try:
            expr = safe_parse(expr_text)
            f = sp.lambdify(x, expr, modules=["numpy", "math"])
            ys = f(xs)
            plt.plot(xs, ys, label=expr_text)
        except Exception as e:
            st.warning(f"Could not plot {expr_text}: {e}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(plt.gcf())
    plt.clf()

def parse_set(s):
    # expect input like {1,2,3} or 1,2,3
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if not s:
        return set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # Try to parse ints/floats, otherwise leave as string
    parsed = set()
    for p in parts:
        try:
            val = int(p)
        except:
            try:
                val = float(p)
            except:
                val = p
        parsed.add(val)
    return parsed

# -----------------------
# UI: Left column — choose operation
# -----------------------
ops = [
    "Arithmetic / Percentage / Profit-Loss",
    "Algebra: simplify / expand / factor / substitute",
    "Solve equation (single)",
    "Solve system of equations",
    "Trigonometry simplify / eval",
    "Derivative (symbolic)",
    "Integral (indefinite / definite)",
    "Plot function(s) y = f(x)",
    "Sets: union/intersection/difference/membership",
    "Expression evaluation / numeric"
]
st.sidebar.header("Choose a tool")
choice = st.sidebar.selectbox("Operation", ops)

# -----------------------
# Tool: Arithmetic / Percent / Profit-Loss
# -----------------------
if choice == ops[0]:
    st.header("Arithmetic / Percentage / Profit & Loss")
    st.markdown("Use this to solve simple business math: profit, loss, discount, percentage change, simple interest.")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Value A", value=100.0)
        b = st.number_input("Value B", value=120.0)
        op = st.selectbox("Operation", ["Difference (B-A)", "Percentage change (B vs A)", "Profit/Loss (given cost & sell)", "Discount", "Simple Interest"])
    with col2:
        if op == "Profit/Loss (given cost & sell)":
            cost = st.number_input("Cost price", value=100.0)
            sell = st.number_input("Selling price", value=120.0)
            if st.button("Compute Profit/Loss"):
                diff = sell - cost
                pct = (diff / cost) * 100 if cost != 0 else float('inf')
                if diff > 0:
                    st.success(f"Profit = {diff:.2f} ({pct:.2f}%)")
                elif diff < 0:
                    st.error(f"Loss = {abs(diff):.2f} ({abs(pct):.2f}%)")
                else:
                    st.info("No profit, no loss")
        elif op == "Difference (B-A)":
            if st.button("Compute Difference"):
                st.write(f"Difference = {b - a:.4f}")
        elif op == "Percentage change (B vs A)":
            if st.button("Compute % change"):
                if a == 0:
                    st.warning("Base A is zero — percentage change undefined")
                else:
                    pct = ((b - a) / a) * 100
                    st.write(f"Percentage change = {pct:.2f}%")
        elif op == "Discount":
            price = st.number_input("Original price", value=100.0)
            disc = st.number_input("Discount percent", value=10.0)
            if st.button("Compute Discount"):
                newp = price * (1 - disc/100)
                st.write(f"Discounted price = {newp:.2f}")
        elif op == "Simple Interest":
            p = st.number_input("Principal", value=1000.0)
            r = st.number_input("Annual rate (%)", value=5.0)
            t = st.number_input("Time (years)", value=1.0)
            if st.button("Compute SI"):
                si = p * r * t / 100
                st.write(f"Simple Interest = {si:.2f}, Amount = {p+si:.2f}")

# -----------------------
# Algebra: simplify/expand/factor/substitute
# -----------------------
elif choice == ops[1]:
    st.header("Algebra: simplify / expand / factor / substitute")
    expr_in = st.text_input("Enter expression (use x,y,z):", value="(x+1)**2")
    action = st.selectbox("Action", ["Simplify", "Expand", "Factor", "Substitute (give x=2)"])
    if st.button("Run"):
        try:
            e = safe_parse(expr_in)
            if action == "Simplify":
                st.write(sp.simplify(e))
            elif action == "Expand":
                st.write(sp.expand(e))
            elif action == "Factor":
                st.write(sp.factor(e))
            elif action.startswith("Substitute"):
                sub_str = st.text_input("Substitute (syntax: x=2,y=3)", value="x=2")
                subs = {}
                for kv in sub_str.split(","):
                    if "=" in kv:
                        k,v = kv.split("=")
                        subs[sp.Symbol(k.strip())] = safe_parse(v.strip())
                st.write(e.subs(subs))
        except Exception as ex:
            st.error(ex)

# -----------------------
# Solve single equation
# -----------------------
elif choice == ops[2]:
    st.header("Solve an equation (single variable)")
    eq_text = st.text_input("Equation, e.g. x**2 - 4 = 0", value="x**2 - 4 = 0")
    var = st.text_input("Solve for (symbol) — leave blank to auto-detect", value="x")
    if st.button("Solve"):
        try:
            sols = solve_equation(eq_text, [sp.Symbol(var)] if var.strip() else None)
            st.write("Solutions:")
            st.write(sols)
            if hasattr(sols, '__iter__') and len(sols) > 0:
                # show numeric approximate
                try:
                    approx = [sp.N(s) for s in sols]
                    st.write("Numeric approximations:")
                    st.write(approx)
                except:
                    pass
        except Exception as ex:
            st.error(ex)

# -----------------------
# Solve system
# -----------------------
elif choice == ops[3]:
    st.header("Solve a system of equations")
    st.markdown("Enter one equation per line. Use `=`. Example:\n```\nx + y = 5\nx - y = 1\n```")
    sys_text = st.text_area("Equations (one per line)", height=150, value="x + y = 5\nx - y = 1")
    if st.button("Solve system"):
        try:
            sol = solve_system(sys_text)
            st.write("Solution:")
            st.write(sol)
        except Exception as ex:
            st.error(ex)

# -----------------------
# Trig
# -----------------------
elif choice == ops[4]:
    st.header("Trigonometry")
    trig_expr = st.text_input("Trig expression (use sin, cos, tan, asin, acos):", value="sin(x)**2 + cos(x)**2")
    if st.button("Simplify / Evaluate"):
        try:
            e = safe_parse(trig_expr)
            st.write("Simplified:", sp.simplify(e))
            # evaluate at numeric value if wants
            if st.checkbox("Evaluate numerically at x=pi/4"):
                val = float(e.subs({sp.Symbol('x'): sp.pi/4}).evalf())
                st.write("Value at x=π/4:", val)
        except Exception as ex:
            st.error(ex)

# -----------------------
# Derivative
# -----------------------
elif choice == ops[5]:
    st.header("Derivative (symbolic)")
    expr = st.text_input("Enter function f(x):", value="sin(x)*x**2")
    var = st.text_input("Variable (default x):", value="x")
    order = st.number_input("Order of derivative", min_value=1, max_value=10, value=1, step=1)
    if st.button("Differentiate"):
        try:
            res = differentiate(expr, var, int(order))
            st.write("Result:")
            st.write(sp.simplify(res))
            st.write("LaTeX:")
            st.latex(sp.latex(res))
        except Exception as ex:
            st.error(ex)

# -----------------------
# Integral
# -----------------------
elif choice == ops[6]:
    st.header("Integral (indefinite / definite)")
    expr = st.text_input("Function f(x) to integrate:", value="x**2")
    var = st.text_input("Variable (default x):", value="x")
    mode = st.selectbox("Mode", ["Indefinite", "Definite"])
    if mode == "Definite":
        low = st.text_input("Lower limit (e.g., 0)", value="0")
        high = st.text_input("Upper limit (e.g., 1)", value="1")
    if st.button("Integrate"):
        try:
            if mode == "Indefinite":
                res = integrate(expr, var, None, None)
                st.write("Indefinite integral:")
                st.write(sp.simplify(res))
                st.latex(sp.latex(res))
            else:
                res = integrate(expr, var, low, high)
                st.write(f"Definite integral from {low} to {high}:")
                st.write(res)
        except Exception as ex:
            st.error(ex)

# -----------------------
# Plot functions
# -----------------------
elif choice == ops[7]:
    st.header("Plot functions y = f(x)")
    text = st.text_area("Enter one or more functions (one per line), use 'x' as variable. E.g.:\n x**2\n sin(x)\n x**3 - 4*x + 1", height=160)
    x_min = st.number_input("x min", value=-10.0)
    x_max = st.number_input("x max", value=10.0)
    if st.button("Plot"):
        exprs = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not exprs:
            st.warning("Enter at least one function")
        else:
            try:
                plot_functions(exprs, 'x', float(x_min), float(x_max))
            except Exception as ex:
                st.error(ex)

# -----------------------
# Sets
# -----------------------
elif choice == ops[8]:
    st.header("Set operations")
    a = st.text_input("Set A (comma separated or {1,2,3}):", value="1,2,3,4")
    b = st.text_input("Set B (comma separated):", value="3,4,5")
    op_set = st.selectbox("Operation", ["Union", "Intersection", "Difference A-B", "Difference B-A", "Membership A? x"])
    if st.button("Compute"):
        try:
            A = parse_set(a)
            B = parse_set(b)
            if op_set == "Union":
                st.write("Union:", A | B)
            elif op_set == "Intersection":
                st.write("Intersection:", A & B)
            elif op_set == "Difference A-B":
                st.write("A - B:", A - B)
            elif op_set == "Difference B-A":
                st.write("B - A:", B - A)
            elif op_set.startswith("Membership"):
                x = st.text_input("Check element:", value="3")
                try:
                    xv = int(x)
                except:
                    try:
                        xv = float(x)
                    except:
                        xv = x
                st.write(xv, "in A?", xv in A)
        except Exception as ex:
            st.error(ex)

# -----------------------
# Expression evaluation / numeric
# -----------------------
elif choice == ops[9]:
    st.header("Expression evaluation / numeric")
    expr = st.text_input("Expression to evaluate, e.g., 2+3*4 or x**2 + 2*x where you provide x value", value="2+3*4")
    subs_text = st.text_input("Subs (optional) like x=2,y=3", value="")
    if st.button("Evaluate"):
        try:
            subs = {}
            if subs_text.strip():
                for kv in subs_text.split(","):
                    if "=" in kv:
                        k,v = kv.split("=")
                        try:
                            subs[sp.Symbol(k.strip())] = safe_parse(v.strip())
                        except:
                            subs[sp.Symbol(k.strip())] = float(v.strip())
            val = safe_parse(expr)
            if subs:
                valn = val.evalf(subs=subs)
            else:
                valn = val.evalf()
            st.write(valn)
        except Exception as ex:
            st.error(ex)

# -----------------------
# Footer / tips
# -----------------------
st.markdown("---")
st.markdown("**Tips:** Use operators `**` for powers (x**2). Functions: `sin, cos, tan, exp, log, sqrt`. For system of equations use one per line.")
