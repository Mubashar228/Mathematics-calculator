# app.py
import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.calculus.util import continuous_domain, function_range

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Math Pro — Advanced", layout="wide")
st.title("🧠 Math Pro — Advanced (Interactive + Stepwise)")

# -----------------------
# Parser config
# -----------------------
transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

def safe_parse(s: str):
    if s is None or str(s).strip() == "":
        raise ValueError("Empty expression")
    s2 = str(s).replace("^", "**")
    return parse_expr(s2, transformations=transformations)

def sympy_to_str_safe(obj):
    try:
        return str(obj)
    except Exception:
        return repr(obj)

def latex_safe(obj):
    try:
        return sp.latex(sp.simplify(obj))
    except Exception:
        return sympy_to_str_safe(obj)

# -----------------------
# Session state: only strings/primitives
# -----------------------
if "main_input" not in st.session_state:
    st.session_state["main_input"] = ""
if "plot_color" not in st.session_state:
    st.session_state["plot_color"] = "blue"

# -----------------------
# Sidebar: symbol palette & controls
# -----------------------
st.sidebar.header("Symbol Palette (click to insert)")
palette = [
    ["+", "-", "*", "/"],
    ["(", ")", "**", "^"],
    ["sqrt()", "root( ,3)", "**2", "**3"],
    ["pi", "E", "sin()", "cos()"],
    ["tan()", "asin()", "acos()", "atan()"],
    ["log()", "ln()", "exp()","abs()"],
    [",", " ", "->", ":"]
]
for row in palette:
    cols = st.sidebar.columns(len(row))
    for i, sym in enumerate(row):
        if cols[i].button(sym):
            # append mapping for some display-friendly tokens
            append_val = sym
            if sym == "^":
                append_val = "**"
            st.session_state["main_input"] = st.session_state["main_input"] + append_val

st.sidebar.markdown("---")
st.sidebar.subheader("Plot options")
color = st.sidebar.selectbox("Plot color", ["blue","red","green","orange","purple","black"], index=0)
st.session_state["plot_color"] = color
st.sidebar.markdown("---")
st.sidebar.write("Tips:")
st.sidebar.write("- Use `**` for powers (or click `**2`, `**3`).")
st.sidebar.write("- Use `sqrt()` for √, `root( ,3)` for ∛.")
st.sidebar.write("- For multi-variable expr use commas: e.g. `sin(x)+cos(y)`")

# -----------------------
# Tools menu
# -----------------------
tool = st.sidebar.selectbox("Choose tool", [
    "Quick Eval / Arithmetic / Business Math",
    "Algebra (simplify/expand/factor & solve)",
    "Equation Solver (single variable) + Plot",
    "Systems of Equations (multi-variable) + Plot",
    "Trigonometry (eval + plot)",
    "Calculus (Derivative & Integral with steps)",
    "Function Analyzer (domain/range/monotonicity/extrema)",
    "Plot Expression (1D / 2D contour / 3D surface)",
    "Sets & Factors",
    "Help & Examples"
])

# -----------------------
# Main input area
# -----------------------
st.markdown("### Input (click palette buttons to append, or type)")
st.session_state["main_input"] = st.text_area("Expression / Equation / Problem", value=st.session_state["main_input"], height=120, key="main_input_box")

# -----------------------
# Helper: numeric evaluation with masking
# -----------------------
def evaluate_numeric_expr(expr_sympy, var=None, xs=None):
    """Return numeric values for xs if var provided, else numeric scalar."""
    if var is None:
        return float(sp.N(expr_sympy))
    f = sp.lambdify(var, expr_sympy, modules=["numpy", "math"])
    ys = np.full_like(xs, np.nan, dtype=float)
    for i, xv in enumerate(xs):
        try:
            val = f(xv)
            ys[i] = float(val) if val is not None else np.nan
        except Exception:
            ys[i] = np.nan
    return ys

# -----------------------
# Step explanation helpers
# -----------------------
def explain_linear(expr, var):
    # Suppose expr is polynomial equal 0, ax + b = 0
    poly = sp.Poly(sp.expand(expr), var)
    a = poly.coeff_monomial(var)
    b = sp.simplify(poly.as_expr() - a*var)
    steps = []
    steps.append("Bring to standard form: ax + b = 0")
    steps.append(f"Compute a (coefficient of {var}): {a}")
    steps.append(f"Compute b (constant term): {b}")
    if a == 0:
        steps.append("Coefficient a is 0 → not linear or no solution/inf solutions.")
    else:
        sol = sp.simplify(-b / a)
        steps.append(f"Solve: x = -b / a = {sol}")
    return steps

def explain_quadratic(expr, var):
    poly = sp.Poly(sp.expand(expr), var)
    a, b, c = poly.all_coeffs() if poly.degree() == 2 else (None, None, None)
    steps = []
    steps.append(f"Quadratic: coefficients a={a}, b={b}, c={c}")
    disc = sp.simplify(b**2 - 4*a*c)
    steps.append(f"Discriminant Δ = b^2 - 4ac = {disc}")
    sqrt_disc = sp.sqrt(disc)
    steps.append(f"Sqrt(Δ) = {sqrt_disc}")
    sol1 = sp.simplify((-b + sqrt_disc) / (2*a))
    sol2 = sp.simplify((-b - sqrt_disc) / (2*a))
    steps.append(f"Solutions: x = (-b ± √Δ)/(2a) → {sol1}, {sol2}")
    return steps

def explain_derivative(expr_sympy, var):
    steps = []
    steps.append("Differentiate using symbolic rules (power/product/chain as needed).")
    deriv = sp.diff(expr_sympy, var)
    steps.append(f"f'(x) = {sp.simplify(deriv)}")
    return steps

def explain_integral(expr_sympy, var, lower=None, upper=None):
    steps = []
    steps.append("Find antiderivative F(x).")
    F = sp.integrate(expr_sympy, var)
    steps.append(f"Antiderivative F(x) = {sp.simplify(F)}")
    if lower is not None and upper is not None:
        val = sp.simplify(F.subs(var, upper) - F.subs(var, lower))
        steps.append(f"Evaluate definite integral: F({upper}) - F({lower}) = {val}")
    return steps

# -----------------------
# Plot helpers using Plotly
# -----------------------
def plot_1d_plotly(var_sym, expr_sym, xmin=-10, xmax=10, color="blue", mark_roots=True):
    xs = np.linspace(xmin, xmax, 1000)
    ys = np.full_like(xs, np.nan, dtype=float)
    f = sp.lambdify(var_sym, expr_sym, modules=["numpy", "math"])
    for i,x in enumerate(xs):
        try:
            val = f(x)
            ys[i] = float(val) if val is not None else np.nan
        except Exception:
            ys[i] = np.nan
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=str(expr_sym), line=dict(color=color)))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    # try to mark roots (numeric)
    if mark_roots:
        try:
            roots = sp.nroots(sp.Poly(sp.simplify(sp.together(expr_sym)).as_expr()))
        except Exception:
            roots = []
        for r in roots:
            try:
                rv = complex(r)
                if abs(rv.imag) < 1e-8 and xmin <= rv.real <= xmax:
                    fig.add_trace(go.Scatter(x=[rv.real], y=[0], mode="markers", marker=dict(color="red", size=8), name=f"root {rv.real:.4g}"))
            except Exception:
                pass
    fig.update_layout(title=f"Plot: {str(expr_sym)}", xaxis_title=str(var_sym), yaxis_title="value")
    st.plotly_chart(fig, use_container_width=True)

def plot_contour_plotly(x_sym, y_sym, expr_sym, rng=(-5,5), cmap="Viridis"):
    xs = np.linspace(rng[0], rng[1], 200)
    ys = np.linspace(rng[0], rng[1], 200)
    X, Y = np.meshgrid(xs, ys)
    f = sp.lambdify((x_sym, y_sym), expr_sym, modules=["numpy", "math"])
    Z = np.full_like(X, np.nan, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i,j] = float(f(X[i,j], Y[i,j]))
            except Exception:
                Z[i,j] = np.nan
    fig = go.Figure(data=go.Contour(z=Z, x=xs, y=ys, colorscale=cmap))
    fig.update_layout(title=f"Contour / heatmap: {str(expr_sym)}", xaxis_title=str(x_sym), yaxis_title=str(y_sym))
    st.plotly_chart(fig, use_container_width=True)

def plot_surface_plotly(x_sym, y_sym, expr_sym, rng=(-3,3), cmap="Viridis"):
    xs = np.linspace(rng[0], rng[1], 100)
    ys = np.linspace(rng[0], rng[1], 100)
    X, Y = np.meshgrid(xs, ys)
    f = sp.lambdify((x_sym, y_sym), expr_sym, modules=["numpy", "math"])
    Z = np.full_like(X, np.nan, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i,j] = float(f(X[i,j], Y[i,j]))
            except Exception:
                Z[i,j] = np.nan
    fig = go.Figure(data=[go.Surface(z=Z, x=xs, y=ys, colorscale=cmap)])
    fig.update_layout(title=f"Surface: {str(expr_sym)}", scene=dict(xaxis_title=str(x_sym), yaxis_title=str(y_sym), zaxis_title="f"))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tool implementations
# -----------------------

# 1) Quick Eval / Arithmetic / Business Math
if tool == "Quick Eval / Arithmetic / Business Math":
    st.header("Arithmetic & Business Math")
    col1, col2 = st.columns(2)
    with col1:
        expr = st.text_input("Expression to evaluate", value=st.session_state["main_input"])
        if st.button("Evaluate Expression"):
            try:
                sym = safe_parse(expr)
                st.write("Symbolic simplified:")
                st.latex(latex_safe(sp.simplify(sym)))
                st.write("Numeric value:")
                st.write(float(sp.N(sym)))
            except Exception as e:
                st.error("Parse/Eval error: " + str(e))

    with col2:
        st.subheader("Business tools")
        A = st.number_input("Base / Cost (A)", value=100.0)
        B = st.number_input("New / Sell (B)", value=120.0)
        if st.button("Percentage change (B vs A)"):
            if A == 0:
                st.warning("Base A is zero — undefined percentage")
            else:
                pct = ((B - A) / A) * 100
                st.write(f"Change: {pct:.2f}%")
        cp = st.number_input("Cost Price (CP)", value=100.0)
        sp_ = st.number_input("Selling Price (SP)", value=120.0)
        if st.button("Profit / Loss"):
            diff = sp_ - cp
            pct = (diff / cp) * 100 if cp != 0 else float('inf')
            if diff > 0:
                st.success(f"Profit = {diff:.2f} ( {pct:.2f}% )")
            elif diff < 0:
                st.error(f"Loss = {abs(diff):.2f} ( {abs(pct):.2f}% )")
            else:
                st.info("No profit, no loss")
        if st.button("Ratio simplify"):
            n1 = st.number_input("Num1 (integer)", value=3, key="ratio_n1")
            n2 = st.number_input("Num2 (integer)", value=6, key="ratio_n2")
            try:
                g = math.gcd(int(n1), int(n2))
                st.write(f"Ratio: {int(n1/g)} : {int(n2/g)}")
            except Exception:
                st.error("Provide integer values")

# 2) Algebra
elif tool == "Algebra (simplify/expand/factor & solve)":
    st.header("Algebra")
    expr = st.text_input("Expression (e.g., (x+1)**2 )", value=st.session_state["main_input"])
    action = st.selectbox("Action", ["Simplify", "Expand", "Factor", "Polynomial factorization", "Solve (single equation)"])
    if st.button("Run Algebra"):
        try:
            s = safe_parse(expr)
            if action == "Simplify":
                st.write("Result:")
                st.latex(latex_safe(sp.simplify(s)))
            elif action == "Expand":
                st.write("Result:")
                st.latex(latex_safe(sp.expand(s)))
            elif action == "Factor":
                st.write("Result:")
                st.latex(latex_safe(sp.factor(s)))
            elif action == "Polynomial factorization":
                st.write("Result:")
                st.latex(latex_safe(sp.factor(s)))
            elif action == "Solve (single equation)":
                eq = st.text_input("Equation (e.g., x**2 - 4 = 0)", value=expr)
                if st.button("Solve equation"):
                    try:
                        if "=" in eq:
                            L,R = eq.split("=",1)
                            sol = sp.solve(sp.Eq(safe_parse(L), safe_parse(R)))
                        else:
                            sol = sp.solve(safe_parse(eq))
                        st.write("Solutions:", sol)
                        # step explanations for common cases
                        if len(sol) > 0:
                            # if polynomial degree known:
                            try:
                                var = list(sol[0].free_symbols)[0] if isinstance(sol[0], sp.Expr) and sol[0].free_symbols else None
                            except Exception:
                                var = None
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error(e)

# 3) Equation solver (single var) + plot
elif tool == "Equation Solver (single variable) + Plot":
    st.header("Equation Solver (single variable) with interactive plot")
    eq = st.text_input("Equation (e.g., sin(x) - 0.5 = 0)", value=st.session_state["main_input"])
    varname = st.text_input("Variable (e.g., x)", value="x")
    xmin = st.number_input("Plot xmin", value=-10.0)
    xmax = st.number_input("Plot xmax", value=10.0)
    if st.button("Solve & Plot"):
        try:
            var = sp.Symbol(varname)
            if "=" in eq:
                L,R = eq.split("=",1)
                expr = sp.simplify(safe_parse(L) - safe_parse(R))
            else:
                expr = safe_parse(eq)
            sols = sp.solve(sp.Eq(expr, 0), var)
            st.write("Symbolic solutions:", sols)
            numeric = []
            for s in sols:
                try:
                    numeric.append(float(sp.N(s)))
                except Exception:
                    pass
            if numeric:
                st.write("Numeric approximations:", numeric)
            plot_1d_plotly(var, expr, xmin, xmax, color=st.session_state["plot_color"], mark_roots=True)
            # Provide step explanations for linear/quadratic if possible
            try:
                poly = sp.Poly(sp.expand(expr), var)
                deg = poly.degree()
                if deg == 1:
                    steps = explain_linear(expr, var)
                    st.markdown("**Step-by-step (linear)**")
                    for s in steps: st.write("- " + str(s))
                elif deg == 2:
                    steps = explain_quadratic(expr, var)
                    st.markdown("**Step-by-step (quadratic)**")
                    for s in steps: st.write("- " + str(s))
            except Exception:
                pass
        except Exception as e:
            st.error(e)

# 4) Systems of Equations
elif tool == "Systems of Equations (multi-variable) + Plot":
    st.header("Systems of Equations")
    st.markdown("Enter one equation per line. Example:\n x + y = 5\n x - y = 1")
    sys_text = st.text_area("Equations", value="x + y = 5\nx - y = 1", height=180)
    if st.button("Solve System"):
        try:
            lines = [ln.strip() for ln in sys_text.replace(";", "\n").splitlines() if ln.strip()]
            eqs = []
            syms = set()
            for ln in lines:
                if "=" in ln:
                    L,R = ln.split("=",1)
                    Ls = safe_parse(L); Rs = safe_parse(R)
                    eqs.append(sp.Eq(Ls, Rs))
                    syms |= set(Ls.free_symbols) | set(Rs.free_symbols)
                else:
                    ex = safe_parse(ln)
                    eqs.append(sp.Eq(ex, 0))
                    syms |= set(ex.free_symbols)
            syms = sorted(list(syms), key=lambda s: s.name)
            sol = sp.solve(eqs, syms, dict=True)
            st.write("Solutions:", sol)
            if len(syms) == 2:
                x, y = syms
                plot = True
                # create contour plot of each equation (implicit)
                xs = np.linspace(-10,10,300)
                ys = np.linspace(-10,10,300)
                X, Y = np.meshgrid(xs, ys)
                fig = go.Figure()
                for i, e in enumerate(eqs):
                    f = sp.lambdify((x,y), sp.simplify(e.lhs - e.rhs), modules=["numpy", "math"])
                    Z = np.full_like(X, np.nan, dtype=float)
                    for ii in range(X.shape[0]):
                        for jj in range(X.shape[1]):
                            try:
                                Z[ii,jj] = float(f(X[ii,jj], Y[ii,jj]))
                            except Exception:
                                Z[ii,jj] = np.nan
                    # plot zero contour as a filled contour with near-zero level
                    fig.add_trace(go.Contour(x=xs, y=ys, z=Z, contours=dict(showlabels=False, coloring='lines', start=0, end=0, size=1), line_width=2, name=f"Eq{i+1}"))
                fig.update_layout(title="Implicit plot of system equations", xaxis_title=str(x), yaxis_title=str(y))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(e)

# 5) Trigonometry
elif tool == "Trigonometry (eval + plot)":
    st.header("Trigonometry Evaluate & Plot")
    tex = st.text_input("Trig expression (in x)", value="sin(x)")
    at_x = st.number_input("Evaluate at x (radians)", value=math.pi/4)
    xmin = st.number_input("Plot xmin", value=-2*math.pi)
    xmax = st.number_input("Plot xmax", value=2*math.pi)
    if st.button("Eval & Plot"):
        try:
            expr = safe_parse(tex)
            val = float(expr.subs(sp.Symbol('x'), at_x).evalf())
            st.write(f"Value at x={at_x}: {val}")
            plot_1d_plotly(sp.Symbol('x'), expr, xmin, xmax, color=st.session_state["plot_color"], mark_roots=False)
        except Exception as e:
            st.error(e)

# 6) Calculus
elif tool == "Calculus (Derivative & Integral with steps)":
    st.header("Calculus: Derivative & Integral (with steps)")
    mode = st.selectbox("Mode", ["Derivative", "Integral"])
    ftext = st.text_input("Function f(x)", value="x**3 - 3*x + 1")
    if mode == "Derivative":
        order = st.number_input("Order", min_value=1, max_value=5, value=1)
        if st.button("Differentiate"):
            try:
                x = sp.Symbol('x')
                expr = safe_parse(ftext)
                deriv = sp.diff(expr, x, int(order))
                st.write("Derivative (symbolic):"); st.latex(latex_safe(deriv))
                st.markdown("**Step summary:**")
                for s in explain_derivative(expr, x):
                    st.write("- " + str(s))
            except Exception as e:
                st.error(e)
    else:
        mode2 = st.selectbox("Integral type", ["Indefinite", "Definite"])
        if mode2 == "Indefinite":
            if st.button("Integrate"):
                try:
                    x = sp.Symbol('x')
                    expr = safe_parse(ftext)
                    F = sp.integrate(expr, x)
                    st.write("Antiderivative F(x):"); st.latex(latex_safe(F))
                    st.markdown("**Step summary:**")
                    for s in explain_integral(expr, x):
                        st.write("- " + str(s))
                except Exception as e:
                    st.error(e)
        else:
            low = st.text_input("Lower (e.g., 0)", value="0")
            high = st.text_input("Upper (e.g., 1)", value="1")
            if st.button("Integrate (definite)"):
                try:
                    x = sp.Symbol('x')
                    expr = safe_parse(ftext)
                    low_s = safe_parse(low); high_s = safe_parse(high)
                    F = sp.integrate(expr, x)
                    val = sp.simplify(F.subs(x, high_s) - F.subs(x, low_s))
                    st.write(f"Definite integral [{low},{high}] = "); st.latex(latex_safe(val))
                    st.markdown("**Step summary:**")
                    for s in explain_integral(expr, x, low_s, high_s):
                        st.write("- " + str(s))
                except Exception as e:
                    st.error(e)

# 7) Function Analyzer
elif tool == "Function Analyzer (domain/range/monotonicity/extrema)":
    st.header("Function Analyzer")
    ftext = st.text_input("Function f(x)", value="(x**3 - 3*x + 1)/(x-1)")
    xmin = st.number_input("Plot xmin", value=-10.0)
    xmax = st.number_input("Plot xmax", value=10.0)
    if st.button("Analyze"):
        try:
            x = sp.Symbol('x')
            fexpr = safe_parse(ftext)
            st.write("Function:"); st.latex(latex_safe(fexpr))
            # Domain
            try:
                dom = continuous_domain(fexpr, x, sp.S.Reals)
                st.write("Domain (symbolic):", dom)
            except Exception:
                st.write("Domain: could not compute symbolically")
            # Range (symbolic or numeric approx)
            try:
                rng_sym = function_range(fexpr, sp.S.Reals, x)
                st.write("Range (symbolic):", rng_sym)
            except Exception:
                xs = np.linspace(xmin, xmax, 1000)
                ys = evaluate_numeric_expr(fexpr, x, xs)
                st.write("Approx range on interval:", np.nanmin(ys), np.nanmax(ys))
            # derivative & critical points
            deriv = sp.simplify(sp.diff(fexpr, x))
            st.write("f'(x):"); st.latex(latex_safe(deriv))
            try:
                crit = sp.solve(sp.Eq(deriv, 0), x)
                st.write("Critical points:", crit)
            except Exception:
                st.write("Critical points: could not compute")
            # monotonicity sample
            try:
                crit_vals = sorted([float(sp.N(c)) for c in crit if c.is_real])
                pts = [-1e6] + crit_vals + [1e6]
                dfunc = sp.lambdify(x, deriv, modules=["numpy", "math"])
                intervals = []
                for i in range(len(pts)-1):
                    mid = (pts[i] + pts[i+1]) / 2.0
                    try:
                        sign = "increasing" if dfunc(mid) > 0 else ("decreasing" if dfunc(mid) < 0 else "stationary")
                    except Exception:
                        sign = "unknown"
                    intervals.append(((pts[i], pts[i+1]), sign))
                st.write("Monotonicity (sampled):")
                for iv, sign in intervals:
                    st.write(f"{iv} → {sign}")
            except Exception:
                st.write("Monotonicity: failed")
            # plot function with critical points
            plot_1d_plotly(x, fexpr, xmin, xmax, color=st.session_state["plot_color"], mark_roots=False)
            # mark critical points on the plot separately
            fig = go.Figure()
            xs = np.linspace(xmin, xmax, 500)
            ys = evaluate_numeric_expr(fexpr, x, xs)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=str(fexpr), line=dict(color=st.session_state["plot_color"])))
            try:
                for c in crit:
                    cnum = float(sp.N(c))
                    if xmin <= cnum <= xmax:
                        yv = float(sp.N(fexpr.subs(x, cnum)))
                        fig.add_trace(go.Scatter(x=[cnum], y=[yv], mode="markers+text", marker=dict(color="red", size=8), text=[f"{cnum:.4g}"], textposition="top right", name=f"crit {cnum:.4g}"))
            except Exception:
                pass
            fig.update_layout(title="Function + Critical Points", xaxis_title=str(x), yaxis_title="f(x)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(e)

# 8) Plot Expression (1D/2D/3D)
elif tool == "Plot Expression (1D / 2D contour / 3D surface)":
    st.header("Plot Expression")
    expr_text = st.text_input("Expression (e.g., sin(x) or sin(x)+cos(y))", value=st.session_state["main_input"])
    rng_min = st.number_input("Range min", value=-5.0)
    rng_max = st.number_input("Range max", value=5.0)
    if st.button("Plot"):
        try:
            expr = safe_parse(expr_text)
            vars_ = sorted(list(expr.free_symbols), key=lambda s: s.name)
            if len(vars_) == 0:
                st.write("Constant:", float(sp.N(expr)))
            elif len(vars_) == 1:
                plot_1d_plotly(vars_[0], expr, rng_min, rng_max, color=st.session_state["plot_color"], mark_roots=True)
            elif len(vars_) == 2:
                plot_contour_plotly(vars_[0], vars_[1], expr, rng=(rng_min, rng_max))
            elif len(vars_) == 3:
                # try to interpret as z = f(x,y)
                # attempt surface with first two variables as x,y if expression depends on them
                st.info("For 3-var expressions a true 3D surface requires form z = f(x,y). App will attempt surface plot.")
                try:
                    # try to use first two variables as x,y
                    plot_surface_plotly(vars_[0], vars_[1], expr, rng=(rng_min, rng_max))
                except Exception as e:
                    st.error("3D surface plotting failed: " + str(e))
            else:
                st.error("More than 3 variables not supported for plotting.")
        except Exception as e:
            st.error(e)

# 9) Sets & Factors
elif tool == "Sets & Factors":
    st.header("Sets & Factors")
    st.subheader("Sets operations")
    A_in = st.text_input("Set A (comma or {a,b,c})", value="1,2,3")
    B_in = st.text_input("Set B", value="2,3,4")
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
    if st.button("Compute"):
        if op == "Union":
            st.write("A ∪ B =", A | B)
        elif op == "Intersection":
            st.write("A ∩ B =", A & B)
        elif op == "A - B":
            st.write("A \\ B =", A - B)
        elif op == "B - A":
            st.write("B \\ A =", B - A)
        else:
            el = st.text_input("Element to check", value="2")
            try:
                val = int(el)
            except:
                try:
                    val = float(el)
                except:
                    val = el
            st.write(val, "in A?", val in A)

    st.markdown("---")
    st.subheader("Prime factors & polynomial factorization")
    n_in = st.text_input("Integer to prime factorize", value="360")
    if st.button("Prime factorize"):
        try:
            n = int(n_in)
            st.write("Prime factorization:", sp.factorint(n))
        except Exception as e:
            st.error(e)
    poly_in = st.text_input("Polynomial to factor (e.g., x**3 - 3*x + 1)", value="x**3 - 3*x + 1")
    if st.button("Factor polynomial"):
        try:
            poly = safe_parse(poly_in)
            st.write("Factored:"); st.latex(latex_safe(sp.factor(poly)))
        except Exception as e:
            st.error(e)

# Help / Examples
elif tool == "Help & Examples":
    st.header("Help & Examples")
    st.markdown("""
**Quick examples you can paste into the input box:**

- Arithmetic: `2 + 3*sqrt(16) / 2`
- Algebra simplify/expand: `(x+1)**2`
- Solve: `x**2 - 4 = 0`
- System: (use one per line)  
  `x + y = 5`  
  `x - y = 1`
- Trig: `sin(x)`, plot -2π..2π
- Derivative: `x**3 - 3*x + 1`
- Integral: `x**2` (definite from 0 to 1)
- Function analyze: `(x**3 - 3*x + 1)/(x-1)`
- 2-variable plot: `sin(x) + cos(y)` → contour
- Prime factor: `360`
""")

# -----------------------
# End footer
# -----------------------
st.markdown("---")
st.markdown("If you want, I can now add: (A) Plotly UI improvements (hover templates), (B) Natural-language step-by-step explanations for the most common cases (linear/quadratic/derivative/integral), (C) Persist user sessions to a DB. Tell me two to implement next.")
