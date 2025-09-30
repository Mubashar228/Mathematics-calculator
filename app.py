# app.py
import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.calculus.util import continuous_domain, function_range
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Math Pro — Advanced Calculator", layout="wide")
st.title("🧮 Math Pro — Algebra • Calculus • Trig • Plots • Steps")

# -------------------------
# Parsing helpers
# -------------------------
transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def safe_parse(s):
    if s is None or str(s).strip()=="":
        raise ValueError("Empty expression")
    try:
        # allow ^ as power
        s2 = str(s).replace("^", "**")
        return parse_expr(s2, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Cannot parse expression: {e}")

def latex_print(x):
    try:
        st.latex(sp.latex(x))
    except Exception:
        st.write(x)

# -------------------------
# Symbol buttons (insert into main input)
# -------------------------
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def insert_symbol(sym):
    # insert at end
    st.session_state.input_text = str(st.session_state.input_text) + sym

st.sidebar.header("Symbols / Quick Insert")
symbol_rows = [
    ["+", "−", "*", "/"],
    ["sqrt()", "cbrt()", "**2", "**3"],
    ["pi", "E", "sin()", "cos()"],
    ["tan()", "log()", "ln()", "^"],
    ["(", ")", ",", " "]
]
for row in symbol_rows:
    cols = st.sidebar.columns(len(row))
    for i, sym in enumerate(row):
        if cols[i].button(sym):
            # map display to parser-friendly text
            mapping = {"−":"-", "^":"**", "cbrt()":"root( ,3)"}
            insert_symbol(mapping.get(sym, sym))

st.sidebar.markdown("---")
st.sidebar.write("Tip: Click symbols to insert into the main input box below.")

# -------------------------
# Sidebar: Tools menu
# -------------------------
tool = st.sidebar.radio("Choose Tool", [
    "Arithmetic & Business Math",
    "Algebra (simplify/expand/factor/solve)",
    "Equation Solver (single + graph)",
    "Systems of Equations",
    "Trigonometry & Trig Plot",
    "Calculus (Derivative & Integral with steps)",
    "Function Analyzer (domain/range/monotonicity/extrema)",
    "Plot Function (1D/2D/3D)",
    "Sets & Factors",
    "Help / Examples"
])

st.markdown("## Input / Quick edit")
st.text_area("Main input (click symbols to insert)", value=st.session_state.input_text, key="input_text", height=90)

# -------------------------
# Utility: plotting helpers
# -------------------------
def plot_1d_expr(expr, var=sp.Symbol('x'), xmin=-10, xmax=10, color='tab:blue', mark_roots=True):
    f = sp.lambdify(var, expr, "numpy")
    xs = np.linspace(xmin, xmax, 800)
    ys = np.full_like(xs, np.nan, dtype=float)
    for i, xv in enumerate(xs):
        try:
            ys[i] = float(f(xv))
        except Exception:
            ys[i] = np.nan
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, ys, color=color, label=str(expr))
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_xlim(xmin, xmax)
    ax.grid(alpha=0.3)
    # roots
    if mark_roots:
        try:
            roots = sp.nroots(sp.poly(sp.simplify(sp.together(expr)).as_expr())) if sp.degree(sp.simplify(sp.together(expr)))>0 else []
        except Exception:
            roots = []
        plotted = 0
        for r in roots:
            try:
                rv = complex(r)
                if abs(rv.imag) < 1e-6 and xmin <= rv.real <= xmax:
                    ax.scatter(rv.real, 0, color='red', zorder=5)
                    plotted += 1
            except Exception:
                pass
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def plot_contour(expr, vars_tuple, rng=(-5,5), cmap='viridis'):
    x, y = vars_tuple
    f = sp.lambdify((x,y), expr, "numpy")
    nx = ny = 200
    xs = np.linspace(rng[0], rng[1], nx)
    ys = np.linspace(rng[0], rng[1], ny)
    X, Y = np.meshgrid(xs, ys)
    try:
        Z = f(X, Y)
    except Exception:
        Z = np.full_like(X, np.nan, dtype=float)
        for i in range(nx):
            for j in range(ny):
                try:
                    Z[i,j] = float(f(X[i,j], Y[i,j]))
                except Exception:
                    Z[i,j] = np.nan
    fig, ax = plt.subplots(figsize=(6,5))
    cs = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel(str(x)); ax.set_ylabel(str(y))
    ax.set_title("Contour / heatmap")
    st.pyplot(fig)
    plt.close(fig)

def plot_surface_3d(expr, vars_tuple, rng=(-3,3), cmap='plasma'):
    x,y,z = vars_tuple
    f = sp.lambdify((x,y,z), expr, "numpy")
    # sample 3D scatter (not continuous surface)
    xs = ys = np.linspace(rng[0], rng[1], 20)
    X, Y = np.meshgrid(xs, ys)
    # for visualization pick z computed from z = f(x,y,z) => not generally solvable; instead if expr is function of (x,y) only treat z variable as output.
    try:
        Z = f(X, Y)
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax)
        ax.set_xlabel(str(x)); ax.set_ylabel(str(y)); ax.set_zlabel("f")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error("3D surface plotting for general 3-var implicit expressions is not supported. Try giving a function of two variables.")

# -------------------------
# 1) Arithmetic & Business Math
# -------------------------
if tool == "Arithmetic & Business Math":
    st.header("Arithmetic & Business Math")
    st.markdown("Operations: evaluate expression, percentage change, ratio, profit/loss, simple interest, discount.")
    col1, col2 = st.columns(2)
    with col1:
        expr = st.text_input("Expression to evaluate (e.g., 2+3*sqrt(16)/2)", value=st.session_state.input_text or "2+3*4/2")
        if st.button("Evaluate Expression"):
            try:
                e = safe_parse(expr)
                st.write("Simplified:", sp.simplify(e))
                st.write("Numeric value:", float(sp.N(e)))
            except Exception as ex:
                st.error(ex)
    with col2:
        st.subheader("Quick business calculators")
        # Percentage change
        a = st.number_input("Base value A", value=100.0, key="pct_a")
        b = st.number_input("New value B", value=120.0, key="pct_b")
        if st.button("Percentage change (B vs A)"):
            if a == 0:
                st.warning("Base value A is zero — percentage undefined")
            else:
                pct = ((b-a)/a)*100
                st.write(f"Change = {pct:.2f}%")
        # Profit/Loss
        cp = st.number_input("Cost Price (CP)", value=100.0, key="cp")
        sp_ = st.number_input("Selling Price (SP)", value=120.0, key="sp")
        if st.button("Compute Profit / Loss"):
            diff = sp_ - cp
            pct = (diff/cp)*100 if cp !=0 else float('inf')
            if diff>0:
                st.success(f"Profit = {diff:.2f} ({pct:.2f}%)")
            elif diff<0:
                st.error(f"Loss = {abs(diff):.2f} ({abs(pct):.2f}%)")
            else:
                st.info("No profit, no loss")
        # Simple Interest
        p = st.number_input("Principal (P)", value=1000.0, key="P")
        r = st.number_input("Annual rate (%)", value=5.0, key="R")
        t = st.number_input("Time (years)", value=1.0, key="T")
        if st.button("Compute Simple Interest"):
            si = p*r*t/100.0
            st.write(f"Simple Interest = {si:.2f}; Amount = {p+si:.2f}")
        # Ratio
        st.markdown("**Ratio**")
        num1 = st.number_input("Num1", value=2.0, key="r1")
        num2 = st.number_input("Num2", value=3.0, key="r2")
        if st.button("Show Ratio simplest form"):
            if num2==0:
                st.error("Division by zero")
            else:
                gcd = np.gcd(int(num1), int(num2)) if float(num1).is_integer() and float(num2).is_integer() else 1
                st.write(f"Ratio = {int(num1/gcd)} : {int(num2/gcd)}")

# -------------------------
# 2) Algebra
# -------------------------
elif tool == "Algebra (simplify/expand/factor/solve)":
    st.header("Algebra tools")
    st.markdown("Simplify, Expand, Factor, Polynomial factors, Solve equations.")
    expr = st.text_input("Expression (e.g., (x+1)**2)", value=st.session_state.input_text or "(x+1)**2")
    action = st.selectbox("Action", ["Simplify","Expand","Factor","Polynomial factorization","Prime factors (integer)"])
    if st.button("Run Algebra"):
        try:
            if action == "Simplify":
                res = sp.simplify(safe_parse(expr))
                st.write("Result:"); latex_print(res)
            elif action == "Expand":
                res = sp.expand(safe_parse(expr))
                st.write("Result:"); latex_print(res)
            elif action == "Factor":
                res = sp.factor(safe_parse(expr))
                st.write("Result:"); latex_print(res)
            elif action == "Polynomial factorization":
                p = safe_parse(expr)
                if p.is_polynomial():
                    st.write(sp.factor(p))
                else:
                    st.warning("Expression is not a polynomial in a single variable.")
            elif action == "Prime factors (integer)":
                try:
                    n = int(expr)
                    facs = sp.factorint(n)
                    st.write(f"Prime factorization: {facs}")
                except Exception:
                    st.error("Provide an integer for prime factorization.")
        except Exception as e:
            st.error(e)
    st.markdown("---")
    st.subheader("Solve an equation (symbolic)")
    eq = st.text_input("Equation (e.g., x**2 - 4 = 0)", value="x**2 - 4 = 0")
    if st.button("Solve Equation"):
        try:
            if "=" in eq:
                left, right = eq.split("=",1)
                sol = sp.solve(sp.Eq(safe_parse(left), safe_parse(right)))
            else:
                sol = sp.solve(safe_parse(eq))
            st.write("Solutions:", sol)
        except Exception as e:
            st.error(e)

# -------------------------
# 3) Equation Solver + Graph
# -------------------------
elif tool == "Equation Solver (single + graph)":
    st.header("Equation Solver with Graph (single variable)")
    eq = st.text_input("Equation (e.g., sin(x) - 0.5 = 0)", value=st.session_state.input_text or "x**3 - 3*x + 1 = 0")
    varname = st.text_input("Variable (e.g., x)", value="x")
    xmin = st.number_input("Plot x min", value=-10.0)
    xmax = st.number_input("Plot x max", value=10.0)
    if st.button("Solve & Plot"):
        try:
            var = sp.Symbol(varname)
            if "=" in eq:
                l, r = eq.split("=",1)
                expr = sp.simplify(safe_parse(l) - safe_parse(r))
            else:
                expr = safe_parse(eq)
            sols = sp.solve(sp.Eq(expr,0), var)
            st.write("Symbolic solutions:", sols)
            # numeric approximations
            approx = []
            for s in sols:
                try:
                    approx.append(float(sp.N(s)))
                except:
                    pass
            if approx:
                st.write("Numeric approximations:", approx)
            # plot
            plot_1d_expr(expr, var=var, xmin=xmin, xmax=xmax, color='tab:green', mark_roots=True)
        except Exception as e:
            st.error(e)

# -------------------------
# 4) Systems of Equations
# -------------------------
elif tool == "Systems of Equations":
    st.header("System of linear/nonlinear equations")
    st.markdown("Enter one equation per line. Example:\n x + y = 5\n x - y = 1")
    eqs_text = st.text_area("Equations", height=180, value="x + y = 5\nx - y = 1")
    if st.button("Solve System"):
        try:
            parts = [p.strip() for p in eqs_text.replace(";", "\n").splitlines() if p.strip()]
            eqs = []
            syms = set()
            for p in parts:
                if "=" in p:
                    a,b = p.split("=",1)
                    la, ra = safe_parse(a), safe_parse(b)
                    eqs.append(sp.Eq(la, ra))
                    syms |= set(la.free_symbols) | set(ra.free_symbols)
                else:
                    ex = safe_parse(p)
                    eqs.append(sp.Eq(ex, 0))
                    syms |= set(ex.free_symbols)
            syms = sorted(list(syms), key=lambda s: s.name)
            sol = sp.solve(eqs, syms, dict=True)
            st.write("Solutions:", sol)
            # If two variables and linear, plot lines intersections
            if len(syms) == 2:
                x, y = syms
                fig, ax = plt.subplots(figsize=(6,5))
                # plot each equation implicitly
                xs = np.linspace(-10, 10, 200)
                ys = np.linspace(-10, 10, 200)
                X, Y = np.meshgrid(xs, ys)
                for eqn in eqs:
                    f = sp.lambdify((x,y), sp.simplify(eqn.lhs - eqn.rhs), "numpy")
                    Z = np.vectorize(lambda xx,yy: f(xx,yy))(X, Y)
                    cs = ax.contour(X, Y, Z, levels=[0], linewidths=2)
                ax.set_xlabel(str(x)); ax.set_ylabel(str(y))
                ax.grid(alpha=0.3)
                st.pyplot(fig)
        except Exception as e:
            st.error(e)

# -------------------------
# 5) Trigonometry
# -------------------------
elif tool == "Trigonometry & Trig Plot":
    st.header("Trigonometry & Values")
    expr = st.text_input("Trig expression (e.g., sin(x)+cos(x))", value="sin(x)")
    angle = st.number_input("Evaluate at x = (radians)", value=0.785398)  # ~pi/4
    xmin = st.number_input("Plot xmin (default -2π)", value=-6.28318530718)
    xmax = st.number_input("Plot xmax (default 2π)", value=6.28318530718)
    if st.button("Evaluate & Plot"):
        try:
            e = safe_parse(expr)
            val = float(e.subs(sp.Symbol('x'), angle).evalf())
            st.write(f"Value at x={angle}: {val}")
            plot_1d_expr(e, var=sp.Symbol('x'), xmin=xmin, xmax=xmax, color='tab:orange')
        except Exception as ex:
            st.error(ex)

# -------------------------
# 6) Calculus: Derivative & Integral (with steps)
# -------------------------
elif tool == "Calculus (Derivative & Integral with steps)":
    st.header("Calculus: Derivative & Integral (with steps)")
    calc_mode = st.radio("Choose", ["Derivative", "Integral"])
    expr_in = st.text_input("Function in x (e.g., x**3 - 3*x + 1)", value="x**3 - 3*x + 1")
    if calc_mode == "Derivative":
        order = st.number_input("Order", min_value=1, max_value=5, value=1)
        if st.button("Compute Derivative"):
            try:
                x = sp.Symbol('x')
                expr = safe_parse(expr_in)
                d = sp.diff(expr, x, int(order))
                st.write("Derivative:")
                latex_print(d)
                # show steps: simplistic step display
                st.markdown("**Steps (symbolic):**")
                st.write("1. Parse expression; 2. Apply differentiation rules; 3. Simplify.")
                st.markdown("**Simplified derivative:**")
                latex_print(sp.simplify(d))
            except Exception as e:
                st.error(e)
    else:
        mode = st.selectbox("Integral type", ["Indefinite", "Definite"])
        if mode == "Definite":
            low = st.text_input("Lower limit (e.g., 0)")
            high = st.text_input("Upper limit (e.g., 1)")
        if st.button("Compute Integral"):
            try:
                x = sp.Symbol('x')
                expr = safe_parse(expr_in)
                if mode == "Indefinite":
                    F = sp.integrate(expr, x)
                    st.write("Antiderivative:")
                    latex_print(F)
                    st.markdown("Steps: 1. Recognize rule (power, trig, exp, etc.); 2. Apply antiderivative; 3. Simplify.")
                else:
                    low_e = safe_parse(low); high_e = safe_parse(high)
                    F = sp.integrate(expr, x)
                    val = sp.simplify(F.subs(x, high_e) - F.subs(x, low_e))
                    st.write(f"Definite integral from {low} to {high}:")
                    latex_print(val)
            except Exception as e:
                st.error(e)

# -------------------------
# 7) Function Analyzer
# -------------------------
elif tool == "Function Analyzer (domain/range/monotonicity/extrema)":
    st.header("Function Analyzer")
    ftext = st.text_input("Function f(x) (example: (x**3-3*x+1)/(x-1))", value="(x**3-3*x+1)/(x-1)")
    xmin = st.number_input("Plot xmin", value=-10.0)
    xmax = st.number_input("Plot xmax", value=10.0)
    if st.button("Analyze Function"):
        try:
            x = sp.Symbol('x')
            fexpr = safe_parse(ftext)
            st.write("Function:")
            latex_print(fexpr)
            # Domain
            try:
                dom = continuous_domain(fexpr, x, sp.S.Reals)
                st.write("Domain (symbolic):", dom)
            except Exception:
                st.write("Domain: could not compute symbolically.")
            # Range (attempt)
            try:
                rng = function_range(fexpr, sp.S.Reals, x)
                st.write("Range (symbolic):", rng)
            except Exception:
                # numeric approx
                fnum = sp.lambdify(x, fexpr, "numpy")
                xs = np.linspace(xmin, xmax, 800)
                ys = np.full_like(xs, np.nan, dtype=float)
                for i,v in enumerate(xs):
                    try:
                        ys[i] = float(fnum(v))
                    except:
                        ys[i] = np.nan
                st.write("Approx range on given interval:", np.nanmin(ys), np.nanmax(ys))
            # derivative and critical points
            deriv = sp.simplify(sp.diff(fexpr, x))
            st.write("f'(x):"); latex_print(deriv)
            try:
                crits = sp.solve(sp.Eq(deriv,0), x)
                st.write("Critical points:", crits)
            except Exception:
                st.write("Critical points: could not compute symbolically")
            # plot with critical points marked
            plot_1d_expr(fexpr, var=x, xmin=xmin, xmax=xmax, color='tab:purple', mark_roots=False)
        except Exception as e:
            st.error(e)

# -------------------------
# 8) Plot Function (multi-var)
# -------------------------
elif tool == "Plot Function (1D/2D/3D)":
    st.header("Plot Function (1D, 2D contour, 3D surface where applicable)")
    expr_text = st.text_input("Expression (in variables like x or x,y or x,y,z)", value=st.session_state.input_text or "sin(x) + cos(y)")
    rng_min = st.number_input("Range min (for plotting axes)", value=-5.0)
    rng_max = st.number_input("Range max (for plotting axes)", value=5.0)
    if st.button("Plot Expression"):
        try:
            expr = safe_parse(expr_text)
            vars_ = sorted(list(expr.free_symbols), key=lambda s: s.name)
            if len(vars_) == 0:
                st.write("Constant:", float(sp.N(expr)))
            elif len(vars_) == 1:
                plot_1d_expr(expr, var=vars_[0], xmin=rng_min, xmax=rng_max, color='tab:blue', mark_roots=True)
            elif len(vars_) == 2:
                plot_contour(expr, vars_tuple=(vars_[0], vars_[1]), rng=(rng_min, rng_max))
            elif len(vars_) == 3:
                # attempt surface by treating expression as z = f(x,y)
                # If one variable appears as dependent z, convert accordingly; else error
                # We'll try to detect if expr is function of two variables only (common case)
                allnames = [str(v) for v in vars_]
                # If third variable is not actually used in expr (rare), proceed; else show message
                st.info("For 3D, app tries to display z = f(x,y) surface; if expression is truly 3-var implicit, surface may not be shown.")
                try:
                    # attempt using first two variables as inputs
                    plot_surface_3d(expr, vars_tuple=(vars_[0], vars_[1], vars_[2]), rng=(rng_min, rng_max))
                except Exception as e:
                    st.error(e)
            else:
                st.error("More than 3 variables unsupported for plotting")
        except Exception as e:
            st.error(e)

# -------------------------
# 9) Sets & Factors
# -------------------------
elif tool == "Sets & Factors":
    st.header("Sets operations and factorization")
    st.subheader("Sets")
    A_text = st.text_input("Set A (comma or { } )", value="1,2,3,4")
    B_text = st.text_input("Set B (comma or { } )", value="3,4,5")
    def parse_set(s):
        s = str(s).strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out = set()
        for p in parts:
            try:
                out.add(int(p))
            except:
                try:
                    out.add(float(p))
                except:
                    out.add(p)
        return out
    A = parse_set(A_text)
    B = parse_set(B_text)
    op = st.selectbox("Operation", ["Union","Intersection","A - B","B - A","Membership A contains x?"])
    if st.button("Compute Sets"):
        if op == "Union":
            st.write("A ∪ B =", A | B)
        elif op == "Intersection":
            st.write("A ∩ B =", A & B)
        elif op == "A - B":
            st.write("A \\ B =", A - B)
        elif op == "B - A":
            st.write("B \\ A =", B - A)
        else:
            xq = st.text_input("Element to check (x)", value="3")
            try:
                val = int(xq)
            except:
                try:
                    val = float(xq)
                except:
                    val = xq
            st.write(val, "in A?", val in A)

    st.markdown("---")
    st.subheader("Factorization")
    n_text = st.text_input("Integer to factorize (prime factors)", value="360")
    if st.button("Prime Factorization"):
        try:
            n = int(n_text)
            pf = sp.factorint(n)
            st.write("Prime factors:", pf)
        except Exception as e:
            st.error(e)
    st.markdown("Polynomial factoring:")
    poly_text = st.text_input("Polynomial (e.g., x**3 - 3*x + 1)", value="x**3 - 3*x + 1")
    if st.button("Factor Polynomial"):
        try:
            p = safe_parse(poly_text)
            st.write("Factored:", sp.factor(p))
        except Exception as e:
            st.error(e)

# -------------------------
# Help / Examples
# -------------------------
elif tool == "Help / Examples":
    st.header("Help & Examples")
    st.markdown("""
**Examples you can paste into the input box or try via tools:**

- Arithmetic: `2 + 3*sqrt(16) / 2`
- Percentage change: set A=100, B=125 and click Percentage change
- Algebra simplify/expand: `(x+1)**2`
- Solve equation: `x**2 - 4 = 0`
- System: `x + y = 5` (new line) `x - y = 1`
- Trig: `sin(x) + cos(x)` (plot from -2π to 2π)
- Derivative: `x**3 - 3*x + 1`
- Integral: `x**2` (definite with limits 0 and 1)
- Function analyze: `(x**3 - 3*x + 1)/(x-1)`
- 2-variable plot: `sin(x) + cos(y)` (Contour)
- Prime factors: `360`

**Notes:**
- Use `**` for power, or click `**2`/`**3` in sidebar.
- Use `sqrt()` for square root.
- For multi-variable plotting, use functions of 2 variables (x,y) to get contour or surface.
""")

# -------------------------
# Footer / tips
# -------------------------
st.markdown("---")
st.markdown("Built with SymPy and Streamlit — message me for custom features or to convert this into a web service with user accounts and persistent storage.")
