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
from mpl_toolkits.mplot3d import Axes3D  # noqa
import math

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Math Pro — All-in-One Calculator", layout="wide")
st.title("🧠 Math Pro — All-in-One Calculator (Algebra, Calculus, Trig, Plots & More)")

# -----------------------
# Parser config
# -----------------------
transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def safe_parse_to_sympy(s: str):
    """Parse string to SymPy expression reliably, converting ^ to ** first."""
    if s is None or str(s).strip() == "":
        raise ValueError("Empty input")
    s2 = str(s).replace("^", "**")
    return parse_expr(s2, transformations=transformations)

def latex_print(x):
    try:
        st.latex(sp.latex(x))
    except Exception:
        st.write(x)

# -----------------------
# Session state safe storage (store strings/primitive only)
# -----------------------
if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""
if "last_result_text" not in st.session_state:
    st.session_state["last_result_text"] = ""
if "preferred_color" not in st.session_state:
    st.session_state["preferred_color"] = "tab:blue"

# -----------------------
# Symbol palette (buttons append text)
# -----------------------
st.sidebar.header("Symbols — click to insert")
symbols = [
    "+", "-", "*", "/", "(", ")", "^", "**",
    "sqrt()", "cbrt()", "**2", "**3",
    "pi", "E",
    "sin()", "cos()", "tan()", "asin()", "acos()", "atan()",
    "log()", "ln()", "exp()"
]
cols = st.sidebar.columns(4)
for i, sym in enumerate(symbols):
    if cols[i % 4].button(sym):
        # map special displays to parser-friendly text
        mapping = {"^":"**","cbrt()":"root( ,3)"}
        st.session_state["main_input"] = st.session_state["main_input"] + mapping.get(sym, sym)

st.sidebar.markdown("---")
st.sidebar.write("Colors (plots):")
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown"]
selc = st.sidebar.selectbox("Plot color", colors, index=0)
st.session_state["preferred_color"] = selc

st.sidebar.markdown("---")
st.sidebar.write("Quick tips:")
st.sidebar.write("- Use `**` for power (or click `**2`,`**3`).")
st.sidebar.write("- Use `sqrt()` for square root.")
st.sidebar.write("- Use `,` to separate variables for multi-var expressions (e.g., `sin(x)+cos(y)`).")

# -----------------------
# Tools menu
# -----------------------
tool = st.sidebar.radio("Choose tool", [
    "Quick Eval / Arithmetic / Business Math",
    "Algebra (simplify/expand/factor & polynomial)",
    "Equation Solver + Graph (single var)",
    "Systems of Equations (multi-var)",
    "Trigonometry (eval + plot)",
    "Calculus (Derivative & Integral with steps)",
    "Function Analyzer (domain/range/monotonicity/extrema)",
    "Plot Function (1D/2D contour / 3D surface)",
    "Sets & Factors",
    "Help / Examples"
])

# -----------------------
# Input box (main)
# -----------------------
st.markdown("## Input")
st.markdown("Click symbols in the sidebar to append into this input box, or type directly. Stored as text (safe).")
st.session_state["main_input"] = st.text_area("Expression / Equation / Problem", value=st.session_state["main_input"], height=120, key="main_input_box")

# utility: safe numeric evaluate with masking invalids
def safe_lambdify_and_eval(expr_sympy, var, xs):
    f = sp.lambdify(var, expr_sympy, modules=["numpy", "math"])
    ys = np.full_like(xs, np.nan, dtype=float)
    for i, xv in enumerate(xs):
        try:
            val = f(xv)
            if val is None:
                ys[i] = np.nan
            else:
                ys[i] = float(val)
        except Exception:
            ys[i] = np.nan
    return ys

# --------------
# Tool: Quick Eval / Arithmetic / Business Math
# --------------
if tool == "Quick Eval / Arithmetic / Business Math":
    st.header("Arithmetic & Business Math")
    colA, colB = st.columns(2)
    with colA:
        expr = st.text_input("Expression to evaluate (use 'x' if needed)", value=st.session_state["main_input"])
        if st.button("Evaluate"):
            try:
                e = safe_parse_to_sympy(expr)
                st.write("Symbolic Simplified:")
                latex_print(sp.simplify(e))
                st.write("Numeric value:")
                with st.spinner("Computing..."):
                    st.write(float(sp.N(e)))
            except Exception as ex:
                st.error(f"Parse/Eval error: {ex}")

    with colB:
        st.subheader("Business calculators")
        st.markdown("Profit/Loss, Percentage change, Ratio, Discount, Simple Interest")
        A = st.number_input("Base / Cost (A)", value=100.0, key="baseA")
        B = st.number_input("New / Sell (B)", value=120.0, key="newB")
        if st.button("Percentage change (B vs A)"):
            if A == 0:
                st.warning("Base A is zero — undefined percentage")
            else:
                pct = ((B - A) / A) * 100
                st.write(f"Change: {pct:.2f}%")
        cp = st.number_input("Cost Price (CP)", value=100.0, key="cp")
        sp_ = st.number_input("Selling Price (SP)", value=120.0, key="sp")
        if st.button("Profit / Loss"):
            diff = sp_ - cp
            pct = (diff / cp) * 100 if cp != 0 else float('inf')
            if diff > 0:
                st.success(f"Profit = {diff:.2f} ( {pct:.2f}% )")
            elif diff < 0:
                st.error(f"Loss = {abs(diff):.2f} ( {abs(pct):.2f}% )")
            else:
                st.info("No profit, no loss")
        if st.button("Discount"):
            price = st.number_input("Original price", value=100.0, key="disc_price")
            disc = st.number_input("Discount %", value=10.0, key="disc_pct")
            newp = price * (1 - disc / 100.0)
            st.write(f"Discounted price: {newp:.2f}")
        if st.button("Simple Interest"):
            P = st.number_input("Principal P", value=1000.0, key="P")
            r = st.number_input("Rate % (annual)", value=5.0, key="r")
            t = st.number_input("Time (years)", value=1.0, key="t")
            si = P * r * t / 100.0
            st.write(f"SI = {si:.2f} , Amount = {P + si:.2f}")
        if st.button("Ratio Simplify"):
            n1 = st.number_input("Num1 (integer)", value=3, key="ratio1")
            n2 = st.number_input("Num2 (integer)", value=6, key="ratio2")
            try:
                g = math.gcd(int(n1), int(n2))
                st.write(f"Ratio: {int(n1/g)} : {int(n2/g)}")
            except Exception:
                st.error("Provide integer values for ratio.")

# --------------
# Tool: Algebra
# --------------
elif tool == "Algebra (simplify/expand/factor & polynomial)":
    st.header("Algebra Tools")
    expr = st.text_input("Expression (e.g., (x+1)**2 )", value=st.session_state["main_input"])
    action = st.selectbox("Action", ["Simplify", "Expand", "Factor", "Polynomial factorization", "Solve (single eq.)"])
    if st.button("Run Algebra"):
        try:
            s = safe_parse_to_sympy(expr)
            if action == "Simplify":
                res = sp.simplify(s)
                st.write("Simplified:"); latex_print(res)
            elif action == "Expand":
                res = sp.expand(s)
                st.write("Expanded:"); latex_print(res)
            elif action == "Factor":
                res = sp.factor(s)
                st.write("Factored:"); latex_print(res)
            elif action == "Polynomial factorization":
                # require symbol
                st.write("Polynomial factorization (attempt):")
                try:
                    st.write(sp.factor(s))
                except Exception as e:
                    st.error(e)
            elif action == "Solve (single eq.)":
                eq = st.text_input("Equation (e.g., x**2-4=0)", value=expr)
                if "=" in eq:
                    L, R = eq.split("=",1)
                    sol = sp.solve(sp.Eq(safe_parse_to_sympy(L), safe_parse_to_sympy(R)))
                else:
                    sol = sp.solve(safe_parse_to_sympy(eq))
                st.write("Solutions:", sol)
        except Exception as ex:
            st.error(ex)

# --------------
# Tool: Equation solver + graph (single variable)
# --------------
elif tool == "Equation Solver (single + graph)":
    st.header("Equation Solver + Graph (single variable)")
    eq = st.text_input("Equation (e.g., sin(x)-0.5=0 )", value=st.session_state["main_input"])
    varname = st.text_input("Variable", value="x")
    xmin = st.number_input("x min", value=-10.0)
    xmax = st.number_input("x max", value=10.0)
    if st.button("Solve & Plot"):
        try:
            var = sp.Symbol(varname)
            if "=" in eq:
                L, R = eq.split("=",1)
                expr = sp.simplify(safe_parse_to_sympy(L) - safe_parse_to_sympy(R))
            else:
                expr = safe_parse_to_sympy(eq)
            sols = sp.solve(sp.Eq(expr, 0), var)
            st.write("Symbolic solutions:", sols)
            numeric = []
            for s in sols:
                try:
                    numeric.append(float(sp.N(s)))
                except:
                    pass
            if numeric:
                st.write("Numeric approximations:", numeric)
            # Plotting line and mark roots
            # produce numeric y-values safely
            xs = np.linspace(xmin, xmax, 1000)
            ys = np.full_like(xs, np.nan, dtype=float)
            f = sp.lambdify(var, expr, "numpy")
            for i, xv in enumerate(xs):
                try:
                    ys[i] = float(f(xv))
                except Exception:
                    ys[i] = np.nan
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(xs, ys, color=st.session_state["preferred_color"], label=str(expr))
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(0, color="black", lw=0.6)
            # mark solutions on x-axis
            for s in numeric:
                if xmin <= s <= xmax:
                    ax.scatter(s, 0, color="red", zorder=5)
                    ax.annotate(f"{s:.4g}", (s,0), textcoords="offset points", xytext=(5,5))
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig)
        except Exception as ex:
            st.error(ex)

# --------------
# Tool: Systems of Equations
# --------------
elif tool == "Systems of Equations":
    st.header("Systems of Equations (linear/nonlinear)")
    st.markdown("Enter one equation per line (use =). Example:\n x + y = 5\n x - y = 1")
    sys_text = st.text_area("Equations", value="x + y = 5\nx - y = 1", height=180)
    if st.button("Solve System"):
        try:
            lines = [ln.strip() for ln in sys_text.replace(";", "\n").splitlines() if ln.strip()]
            eqs = []
            syms = set()
            for ln in lines:
                if "=" in ln:
                    L, R = ln.split("=",1)
                    Ls = safe_parse_to_sympy(L); Rs = safe_parse_to_sympy(R)
                    eqs.append(sp.Eq(Ls, Rs))
                    syms |= set(Ls.free_symbols) | set(Rs.free_symbols)
                else:
                    ex = safe_parse_to_sympy(ln)
                    eqs.append(sp.Eq(ex, 0))
                    syms |= set(ex.free_symbols)
            syms = sorted(list(syms), key=lambda s: s.name)
            sol = sp.solve(eqs, syms, dict=True)
            st.write("Solutions:", sol)
            if len(syms) == 2:
                x, y = syms
                # plot implicit curves
                xs = np.linspace(-10,10,300); ys = np.linspace(-10,10,300)
                X, Y = np.meshgrid(xs, ys)
                fig, ax = plt.subplots(figsize=(6,6))
                for e in eqs:
                    f = sp.lambdify((x,y), sp.simplify(e.lhs - e.rhs), "numpy")
                    Z = np.full_like(X, np.nan, dtype=float)
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            try:
                                Z[i,j] = float(f(X[i,j], Y[i,j]))
                            except:
                                Z[i,j] = np.nan
                    # contour level 0
                    try:
                        cs = ax.contour(X, Y, Z, levels=[0], linewidths=2)
                    except Exception:
                        pass
                ax.set_xlabel(str(x)); ax.set_ylabel(str(y))
                ax.grid(alpha=0.3)
                st.pyplot(fig)
        except Exception as ex:
            st.error(ex)

# --------------
# Tool: Trigonometry
# --------------
elif tool == "Trigonometry (eval + plot)":
    st.header("Trigonometry")
    trig_expr = st.text_input("Trig expression in x (e.g., sin(x)+cos(x))", value="sin(x)")
    angle = st.number_input("Evaluate at x (radians)", value=math.pi/4)
    xmin = st.number_input("Plot xmin", value=-2*math.pi)
    xmax = st.number_input("Plot xmax", value=2*math.pi)
    if st.button("Evaluate & Plot"):
        try:
            e = safe_parse_to_sympy(trig_expr)
            val = float(e.subs(sp.Symbol('x'), angle).evalf())
            st.write(f"Value at x={angle} → {val:.6g}")
            xs = np.linspace(xmin, xmax, 1000)
            ys = safe_lambdify_and_eval(e, sp.Symbol('x'), xs)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(xs, ys, color=st.session_state["preferred_color"], label=trig_expr)
            ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig)
        except Exception as ex:
            st.error(ex)

# --------------
# Tool: Calculus (Derivative & Integral)
# --------------
elif tool == "Calculus (Derivative & Integral with steps)":
    st.header("Calculus — Derivative & Integral")
    calc_mode = st.selectbox("Mode", ["Derivative", "Integral"])
    func = st.text_input("Function f(x)", value="x**3 - 3*x + 1")
    if calc_mode == "Derivative":
        order = st.number_input("Order", min_value=1, max_value=5, value=1)
        if st.button("Differentiate"):
            try:
                x = sp.Symbol('x')
                expr = safe_parse_to_sympy(func)
                deriv = sp.diff(expr, x, int(order))
                st.write("Derivative (symbolic):")
                latex_print(deriv)
                st.markdown("**Steps summary:**")
                st.write("- Parsed expression; applied differentiation rules (product/chain/power); simplified result.")
                st.write("Simplified derivative:"); latex_print(sp.simplify(deriv))
            except Exception as ex:
                st.error(ex)
    else:
        mode = st.selectbox("Integral Type", ["Indefinite", "Definite"])
        if mode == "Definite":
            low = st.text_input("Lower limit", value="0")
            high = st.text_input("Upper limit", value="1")
        if st.button("Integrate"):
            try:
                x = sp.Symbol('x')
                expr = safe_parse_to_sympy(func)
                if mode == "Indefinite":
                    F = sp.integrate(expr, x)
                    st.write("Antiderivative F(x):"); latex_print(F)
                    st.write("Steps summary: Recognize rule → integrate → simplify.")
                else:
                    low_e = safe_parse_to_sympy(low); high_e = safe_parse_to_sympy(high)
                    F = sp.integrate(expr, x)
                    val = sp.simplify(F.subs(x, high_e) - F.subs(x, low_e))
                    st.write(f"Definite integral [{low} , {high}]:"); latex_print(val)
            except Exception as ex:
                st.error(ex)

# --------------
# Tool: Function Analyzer
# --------------
elif tool == "Function Analyzer (domain/range/monotonicity/extrema)":
    st.header("Function Analyzer")
    ftext = st.text_input("Function f(x)", value="(x**3-3*x+1)/(x-1)")
    xmin = st.number_input("Plot xmin", value=-10.0)
    xmax = st.number_input("Plot xmax", value=10.0)
    if st.button("Analyze"):
        try:
            x = sp.Symbol('x')
            fexpr = safe_parse_to_sympy(ftext)
            st.write("Function:"); latex_print(fexpr)
            # Domain
            try:
                dom = continuous_domain(fexpr, x, sp.S.Reals)
                st.write("Domain (symbolic):", dom)
            except Exception:
                st.write("Domain: could not be determined symbolically.")
            # Range (attempt)
            try:
                rng_sym = function_range(fexpr, sp.S.Reals, x)
                st.write("Range (symbolic):", rng_sym)
            except Exception:
                # numeric approximate on [xmin,xmax]
                xs = np.linspace(xmin, xmax, 2000)
                ys = safe_lambdify_and_eval(fexpr, x, xs)
                st.write("Approx range on interval:", np.nanmin(ys), np.nanmax(ys))
            # derivative & critical points
            deriv = sp.simplify(sp.diff(fexpr, x))
            st.write("f'(x):"); latex_print(deriv)
            try:
                crit = sp.solve(sp.Eq(deriv,0), x)
                st.write("Critical points:", crit)
            except Exception:
                st.write("Critical points: could not compute symbolically")
            # monotonicity (sample intervals)
            try:
                crit_vals = [float(sp.N(c)) for c in crit if c.is_real]
                pts = [-1e6] + sorted(crit_vals) + [1e6]
                intervals = []
                dfunc = sp.lambdify(x, deriv, "numpy")
                for i in range(len(pts)-1):
                    mid = (pts[i] + pts[i+1]) / 2.0
                    try:
                        sign = "increasing" if dfunc(mid) > 0 else ("decreasing" if dfunc(mid) < 0 else "stationary")
                    except Exception:
                        sign = "unknown"
                    intervals.append(((pts[i], pts[i+1]), sign))
                st.write("Monotonicity (sampled intervals):")
                for iv, sign in intervals:
                    st.write(f"{iv} → {sign}")
            except Exception:
                st.write("Monotonicity: could not determine robustly.")
            # plot with derivative roots marked
            plot_1d = True
            xs = np.linspace(xmin, xmax, 800)
            ys = safe_lambdify_and_eval(fexpr, x, xs)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(xs, ys, color=st.session_state["preferred_color"], label=str(fexpr))
            ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
            # mark critical points
            try:
                for c in crit:
                    c_num = float(sp.N(c))
                    if xmin <= c_num <= xmax:
                        yv = float(sp.N(fexpr.subs(x, c_num)))
                        ax.scatter(c_num, yv, color='red', zorder=5)
                        ax.annotate(f"{c_num:.3g}", (c_num, yv), textcoords="offset points", xytext=(5,5))
            except Exception:
                pass
            ax.grid(alpha=0.3); ax.legend()
            st.pyplot(fig)
        except Exception as ex:
            st.error(ex)

# --------------
# Tool: Plot Function (1D, 2D contour, 3D)
# --------------
elif tool == "Plot Function (1D/2D/3D)":
    st.header("Plot Function")
    expr_text = st.text_input("Expression (e.g., sin(x), sin(x)+cos(y), x**2 + y**2)", value=st.session_state["main_input"])
    rng_min = st.number_input("Range min", value=-5.0)
    rng_max = st.number_input("Range max", value=5.0)
    if st.button("Plot"):
        try:
            expr = safe_parse_to_sympy(expr_text)
            vars_ = sorted(list(expr.free_symbols), key=lambda s: s.name)
            if len(vars_) == 0:
                st.write("Constant:", float(sp.N(expr)))
            elif len(vars_) == 1:
                plot_1d_expr = expr
                var = vars_[0]
                xs = np.linspace(rng_min, rng_max, 800)
                ys = safe_lambdify_and_eval(plot_1d_expr, var, xs)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(xs, ys, color=st.session_state["preferred_color"], label=str(expr))
                ax.axhline(0,color='black',lw=0.5); ax.axvline(0,color='black',lw=0.5)
                ax.grid(alpha=0.3); ax.legend()
                st.pyplot(fig)
            elif len(vars_) == 2:
                plot_contour(expr, vars_tuple=(vars_[0], vars_[1]), rng=(rng_min, rng_max))
            elif len(vars_) == 3:
                st.info("For 3 variables, app will attempt surface plot if expression can be interpreted as f(x,y).")
                try:
                    plot_surface_3d(expr, vars_tuple=(vars_[0], vars_[1], vars_[2]), rng=(rng_min, rng_max))
                except Exception as e:
                    st.error("3D plotting failed: " + str(e))
            else:
                st.error("More than 3 variables not supported for plotting.")
        except Exception as ex:
            st.error(ex)

# -----------------------
# Tool: Sets & Factors
# -----------------------
elif tool == "Sets & Factors":
    st.header("Sets & Factors")
    st.subheader("Set operations")
    A_in = st.text_input("Set A (comma or {1,2,3})", value="1,2,3,4")
    B_in = st.text_input("Set B", value="3,4,5")
    def parse_set(s):
        s = str(s).strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        res = set()
        for p in parts:
            try:
                res.add(int(p))
            except:
                try:
                    res.add(float(p))
                except:
                    res.add(p)
        return res
    A = parse_set(A_in); B = parse_set(B_in)
    op = st.selectbox("Operation", ["Union","Intersection","A - B","B - A","Membership"])
    if st.button("Compute sets"):
        if op == "Union":
            st.write("A ∪ B =", A | B)
        elif op == "Intersection":
            st.write("A ∩ B =", A & B)
        elif op == "A - B":
            st.write("A \\ B =", A - B)
        elif op == "B - A":
            st.write("B \\ A =", B - A)
        else:
            el = st.text_input("Element to check", value="3")
            try:
                val = int(el)
            except:
                try:
                    val = float(el)
                except:
                    val = el
            st.write(val, "in A?", val in A)

    st.markdown("---")
    st.subheader("Factors")
    n_in = st.text_input("Integer to factor (prime factors)", value="360")
    if st.button("Prime factorize"):
        try:
            n = int(n_in)
            st.write("Prime factorization:", sp.factorint(n))
        except Exception as ex:
            st.error(ex)
    st.subheader("Polynomial factor")
    poly_in = st.text_input("Polynomial (e.g., x**3 - 3*x + 1)", value="x**3 - 3*x + 1")
    if st.button("Factor polynomial"):
        try:
            poly_expr = safe_parse_to_sympy(poly_in)
            st.write("Factored form:"); latex_print(sp.factor(poly_expr))
        except Exception as ex:
            st.error(ex)

# -----------------------
# Help / Examples
# -----------------------
elif tool == "Help / Examples":
    st.header("Help & Examples")
    st.markdown("""
**Examples:**
- Arithmetic: `2 + 3*sqrt(16) / 2`  
- Algebra: `(x+1)**2` → Expand/Factor  
- Solve: `x**2 - 4 = 0`  
- System: `x + y = 5` (newline) `x - y = 1`  
- Trig: `sin(x) + cos(x)` (plot -2π..2π)  
- Derivative: `x**3 - 3*x + 1`  
- Integral: `x**2`, definite 0..1  
- Function analyze: `(x**3 - 3*x + 1)/(x-1)`  
- 2-var plot: `sin(x) + cos(y)` → contour  
- Factor int: `360`  (prime factorization)
""")

# -----------------------
# End
# -----------------------
st.markdown("---")
st.markdown("If you want: I can add (1) prettier interactive Plotly graphs, (2) persist user problems to DB, (3) nicer natural-language step-by-step text for key operations. Tell me which two to add next.")
