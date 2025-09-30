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
from datetime import datetime
transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))

st.set_page_config(page_title="Math Master+ — Steps, Functions & Graphs", layout="wide")

# ---------- Helpers ----------
def safe_parse(s):
    if s is None:
        raise ValueError("Empty input")
    try:
        return parse_expr(s, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Cannot parse expression: {e}")

def latex(expr):
    try:
        return sp.latex(sp.simplify(expr))
    except:
        return str(expr)

# ---------- Step-by-step solver for single equation ----------
def explain_solve_equation(equation_str, var_name='x'):
    steps = []
    x = sp.Symbol(var_name)
    # Parse equation
    if "=" in equation_str:
        left_str, right_str = equation_str.split("=", 1)
        left = safe_parse(left_str)
        right = safe_parse(right_str)
    else:
        left = safe_parse(equation_str)
        right = sp.Integer(0)
    steps.append({"text": "Original equation", "expr": sp.Eq(left, right)})

    # Move all to left -> simplify
    expr = sp.simplify(left - right)
    steps.append({"text": "Bring all terms to one side (standard form)", "expr": sp.Eq(expr, 0)})

    # If polynomial in var
    try:
        poly = sp.Poly(sp.expand(expr), x)
        if poly.is_polynomial(x):
            deg = poly.degree()
        else:
            poly = None
            deg = None
    except Exception:
        poly = None
        deg = None

    # Linear case
    if deg == 1:
        a = poly.coeffs()[0] if len(poly.coeffs()) >= 1 else 0
        # better get coefficients by degree
        a = poly.coeff_monomial(x)
        b = sp.expand(poly.as_expr() - a*x)
        # solve ax + b = 0 => x = -b/a
        steps.append({"text": "Linear equation detected. Isolate x.", "expr": None})
        solution = sp.solve(sp.Eq(expr, 0), x)
        steps.append({"text": "Solution", "expr": solution})
        return steps

    # Quadratic case
    if deg == 2:
        coeffs = poly.all_coeffs()  # [a, b, c]
        a, b, c = coeffs
        steps.append({"text": f"Quadratic detected with coefficients a={a}, b={b}, c={c}", "expr": None})
        disc = sp.simplify(b**2 - 4*a*c)
        steps.append({"text": "Compute discriminant Δ = b² - 4ac", "expr": disc})
        sqrt_disc = sp.sqrt(disc) if disc != 0 else sp.Integer(0)
        steps.append({"text": "Square root of discriminant", "expr": sqrt_disc})
        sol1 = sp.simplify((-b + sqrt_disc) / (2*a))
        sol2 = sp.simplify((-b - sqrt_disc) / (2*a))
        steps.append({"text": "Quadratic formula solutions (x = (-b ± √Δ) / (2a))", "expr": [sol1, sol2]})
        return steps

    # Try factoring
    fact = sp.factor(expr)
    if fact != expr:
        steps.append({"text": "Factorized expression", "expr": sp.Eq(fact, 0)})
        # If factorization yields product, set each factor zero
        if isinstance(fact, sp.Mul):
            factors = fact.as_ordered_factors()
            sol = []
            for f in factors:
                try:
                    sols_f = sp.solve(sp.Eq(f, 0), x)
                    sol.extend(sols_f if isinstance(sols_f, list) else [sols_f])
                except Exception:
                    pass
            steps.append({"text": "Solve each factor = 0", "expr": sol})
            return steps

    # Fallback: numeric/symbolic solve
    try:
        sols = sp.solve(sp.Eq(expr, 0), x)
        steps.append({"text": "General solution (via sympy.solve)", "expr": sols})
    except Exception as e:
        steps.append({"text": "Could not solve symbolically: " + str(e), "expr": None})
    return steps

# ---------- Explain derivative ----------
def explain_derivative(expr_str, var_name='x', order=1):
    x = sp.Symbol(var_name)
    expr = safe_parse(expr_str)
    steps = []
    steps.append({"text":"Original function f(x)", "expr": expr})
    # derivative
    d = sp.diff(expr, x, order)
    steps.append({"text": f"{order} order derivative", "expr": d})
    # simplify
    ds = sp.simplify(d)
    steps.append({"text":"Simplified derivative", "expr": ds})
    # second derivative for critical analysis if order==1
    if order == 1:
        dd = sp.diff(ds, x)
        steps.append({"text":"Second derivative (for concavity / extremum test)", "expr": sp.simplify(dd)})
    return steps

# ---------- Explain integral ----------
def explain_integral(expr_str, var_name='x', lower=None, upper=None):
    x = sp.Symbol(var_name)
    expr = safe_parse(expr_str)
    steps = []
    steps.append({"text":"Integrand", "expr": expr})
    if lower is None and upper is None:
        F = sp.integrate(expr, x)
        steps.append({"text":"Indefinite integral (antiderivative)", "expr": F})
    else:
        l = safe_parse(str(lower))
        u = safe_parse(str(upper))
        F = sp.integrate(expr, x)
        steps.append({"text":"Antiderivative F(x)", "expr": F})
        val = sp.simplify(F.subs(x, u) - F.subs(x, l))
        steps.append({"text": f"Definite integral from {l} to {u}", "expr": val})
    return steps

# ---------- Function analysis ----------
from sympy.calculus.util import continuous_domain, function_range
def analyze_function(expr_str, var_name='x', domain_hint=None, numeric_test_range=(-10,10)):
    x = sp.Symbol(var_name)
    expr = safe_parse(expr_str)
    res = {}
    # domain
    try:
        dom = continuous_domain(expr, x, sp.S.Reals)
        res['domain'] = dom
    except Exception:
        res['domain'] = "Could not determine domain symbolically"
    # range (attempt)
    try:
        rng = function_range(expr, sp.S.Reals, x)
        res['range'] = rng
    except Exception:
        res['range'] = "Range detection failed or too complex"
    # derivative
    try:
        deriv = sp.simplify(sp.diff(expr, x))
        res['derivative'] = deriv
    except Exception:
        res['derivative'] = "Could not differentiate"
    # critical points solve derivative == 0
    crits = []
    try:
        sols = sp.solve(sp.Eq(sp.simplify(sp.diff(expr, x)), 0), x)
        # keep numeric-real ones
        real_crits = []
        for s in sols:
            try:
                s_eval = complex(sp.N(s))
                if abs(s_eval.imag) < 1e-8:
                    real_crits.append(sp.N(s_eval.real))
            except Exception:
                pass
        crits = sorted(list(set(real_crits)))
        res['critical_points'] = crits
    except Exception:
        res['critical_points'] = "Could not find critical points symbolically"
    # monotonicity: test derivative sign on intervals between critical points
    try:
        pts = [-1e6] + [float(c) for c in crits] + [1e6]
        intervals = []
        deriv_f = sp.lambdify(x, deriv, modules=["numpy", "math"])
        for i in range(len(pts)-1):
            a = pts[i]; b = pts[i+1]
            test = (a + b) / 2.0
            try:
                val = deriv_f(test)
                sign = "increasing" if val > 0 else ("decreasing" if val < 0 else "stationary")
            except Exception:
                sign = "unknown"
            intervals.append(((a,b), sign))
        res['monotonicity'] = intervals
    except Exception:
        res['monotonicity'] = "Could not determine monotonicity"
    # extrema via second derivative test where possible
    try:
        second = sp.simplify(sp.diff(expr, x, 2))
        extrema = []
        for c in crits:
            try:
                sec_val = float(second.subs(x, c))
                if sec_val > 0:
                    extrema.append((float(c), "local minima"))
                elif sec_val < 0:
                    extrema.append((float(c), "local maxima"))
                else:
                    extrema.append((float(c), "inconclusive"))
            except Exception:
                extrema.append((float(c), "inconclusive"))
        res['second_derivative'] = second
        res['extrema'] = extrema
    except Exception:
        res['second_derivative'] = "N/A"
        res['extrema'] = "N/A"
    # asymptotes: vertical from denom zeros, horizontal from limits
    try:
        texpr = sp.together(expr)
        denom = sp.denom(texpr)
        verts = []
        if denom != 1:
            roots = sp.solve(sp.Eq(denom, 0), x)
            for r in roots:
                try:
                    rv = complex(sp.N(r))
                    if abs(rv.imag) < 1e-8:
                        verts.append(sp.N(rv.real))
                except Exception:
                    pass
        res['vertical_asymptotes'] = verts
    except Exception:
        res['vertical_asymptotes'] = []
    # horizontal/oblique limits
    try:
        lim_pos = sp.limit(expr, x, sp.oo)
        lim_neg = sp.limit(expr, x, -sp.oo)
        res['limit_pos_inf'] = lim_pos
        res['limit_neg_inf'] = lim_neg
    except Exception:
        res['limit_pos_inf'] = "N/A"
        res['limit_neg_inf'] = "N/A"
    return res

# ---------- Plotting with critical points ----------
def plot_with_critical(expr_str, var_name='x', x_min=-10, x_max=10, points=1000):
    x = sp.Symbol(var_name)
    expr = safe_parse(expr_str)
    f = sp.lambdify(x, expr, modules=["numpy", "math"])
    xs = np.linspace(x_min, x_max, points)
    try:
        ys = f(xs)
    except Exception as e:
        # maybe domain issues: mask where invalid
        ys = np.full_like(xs, np.nan, dtype=float)
        for i, xv in enumerate(xs):
            try:
                ys[i] = float(f(xv))
            except Exception:
                ys[i] = np.nan
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, ys, label=str(expr))
    # critical points
    try:
        deriv = sp.simplify(sp.diff(expr, x))
        crits = sp.solve(sp.Eq(deriv, 0), x)
        real_crits = []
        for c in crits:
            try:
                c_eval = float(sp.N(c))
                if x_min <= c_eval <= x_max:
                    real_crits.append(c_eval)
            except Exception:
                pass
        for c in real_crits:
            try:
                y = float(sp.N(sp.N(expr.subs(x, c))))
                ax.scatter([c], [y], color='red', zorder=5)
                ax.annotate(f"c={round(c,4)}", xy=(c,y), xytext=(5,5), textcoords='offset points')
            except Exception:
                pass
    except Exception:
        pass
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.clf()

# ---------- UI ----------
st.title("Math Master+ (Step-by-step & Function Analyzer)")
st.markdown("Choose a tool from the left. This app returns symbolic answers and **step-by-step explanations** (where possible).")

tools = [
    "Equation solver (detailed steps)",
    "Function analysis (domain, range, monotonicity, extrema, asymptotes)",
    "Derivative (steps)",
    "Integral (steps)",
    "Plot function(s) with critical points",
    "Algebra quick tools (simplify/expand/factor)",
    "Systems, sets, arithmetic (basic)"
]
choice = st.sidebar.selectbox("Tool", tools)

# ---------- Equation solver ----------
if choice == tools[0]:
    st.header("Equation solver — step-by-step")
    eq = st.text_input("Enter equation (use =), e.g. x**2 - 4 = 0", value="x**2 - 4 = 0")
    var = st.text_input("Variable (default x)", value="x")
    if st.button("Explain & Solve"):
        try:
            steps = explain_solve_equation(eq, var)
            for i, s in enumerate(steps):
                st.markdown(f"**Step {i+1}: {s['text']}**")
                if s["expr"] is not None:
                    st.latex(sp.latex(s["expr"]))
                else:
                    st.write(s.get("text_detail",""))
        except Exception as e:
            st.error(e)

# ---------- Function analysis ----------
elif choice == tools[1]:
    st.header("Function analysis")
    expr_in = st.text_input("Enter f(x), e.g. (x**3 - 3*x + 1) / (x-1)", value="(x**3 - 3*x + 1)/(x-1)")
    var = st.text_input("Variable (default x)", value="x")
    x_min = st.number_input("Plot x min", value=-10.0)
    x_max = st.number_input("Plot x max", value=10.0)
    if st.button("Analyze function"):
        try:
            res = analyze_function(expr_in, var, numeric_test_range=(x_min, x_max))
            st.subheader("Summary")
            st.write("Expression:", expr_in)
            st.write("Domain:", res.get('domain'))
            st.write("Range:", res.get('range'))
            st.write("Derivative f'(x):")
            if 'derivative' in res:
                st.latex(sp.latex(res['derivative']))
            st.write("Critical points (real):", res.get('critical_points'))
            st.write("Monotonicity (intervals):")
            mono = res.get('monotonicity')
            if isinstance(mono, list):
                for iv, sign in mono:
                    st.write(f"{iv} → {sign}")
            else:
                st.write(mono)
            st.write("Extrema (second derivative test):", res.get('extrema'))
            st.write("Vertical asymptotes:", res.get('vertical_asymptotes'))
            st.write("Limit x→+∞:", res.get('limit_pos_inf'), " Limit x→-∞:", res.get('limit_neg_inf'))
            st.subheader("Plot with critical points")
            plot_with_critical(expr_in, var, float(x_min), float(x_max))
        except Exception as e:
            st.error("Analysis failed: " + str(e))

# ---------- Derivative ----------
elif choice == tools[2]:
    st.header("Derivative (symbolic) with steps")
    expr = st.text_input("Enter function f(x)", value="x**2 * sin(x)")
    var = st.text_input("Variable (default x)", value="x")
    order = st.number_input("Order", min_value=1, max_value=5, value=1)
    if st.button("Differentiate"):
        try:
            steps = explain_derivative(expr, var, int(order))
            for i, s in enumerate(steps):
                st.markdown(f"**Step {i+1}: {s['text']}**")
                if s['expr'] is not None:
                    st.latex(sp.latex(s['expr']))
        except Exception as e:
            st.error(e)

# ---------- Integral ----------
elif choice == tools[3]:
    st.header("Integral (indefinite & definite) with steps")
    expr = st.text_input("Enter integrand f(x)", value="x**2")
    var = st.text_input("Variable (default x)", value="x")
    mode = st.selectbox("Mode", ["Indefinite", "Definite"])
    if mode == "Definite":
        a = st.text_input("Lower limit", value="0")
        b = st.text_input("Upper limit", value="1")
    if st.button("Integrate"):
        try:
            if mode == "Indefinite":
                steps = explain_integral(expr, var, None, None)
            else:
                steps = explain_integral(expr, var, a, b)
            for i, s in enumerate(steps):
                st.markdown(f"**Step {i+1}: {s['text']}**")
                if s['expr'] is not None:
                    st.latex(sp.latex(s['expr']))
        except Exception as e:
            st.error(e)

# ---------- Plot ----------
elif choice == tools[4]:
    st.header("Plot function(s) and mark critical points")
    text = st.text_area("Enter functions (one per line), example: sin(x)\nx**3 - 3*x + 1", height=180)
    xmin = st.number_input("x min", value=-10.0)
    xmax = st.number_input("x max", value=10.0)
    if st.button("Plot"):
        exprs = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ex in exprs:
            st.subheader(f"Plot: {ex}")
            try:
                plot_with_critical(ex, 'x', float(xmin), float(xmax))
            except Exception as e:
                st.error(f"Cannot plot {ex}: {e}")

# ---------- Algebra quick tools ----------
elif choice == tools[5]:
    st.header("Algebra quick tools")
    expr = st.text_input("Expression", value="(x+1)**2")
    action = st.selectbox("Action", ["Simplify", "Expand", "Factor", "Substitute"])
    if st.button("Run"):
        try:
            e = safe_parse(expr)
            if action == "Simplify":
                st.write(sp.simplify(e))
            elif action == "Expand":
                st.write(sp.expand(e))
            elif action == "Factor":
                st.write(sp.factor(e))
            elif action == "Substitute":
                s = st.text_input("Substitute (e.g., x=2,y=3)", value="x=2")
                subs = {}
                for kv in s.split(","):
                    if "=" in kv:
                        k,v = kv.split("=")
                        subs[sp.Symbol(k.strip())] = safe_parse(v.strip())
                st.write(e.subs(subs))
        except Exception as ex:
            st.error(ex)

# ---------- Systems, sets, arithmetic ----------
elif choice == tools[6]:
    st.header("Systems, sets and basic arithmetic")
    st.markdown("Use existing app earlier — this area supports systems, sets, profit/loss etc.")
    st.info("You can use the previous Math Master code or ask me to add a dedicated UI here.")

st.markdown("---")
st.markdown("**Notes:**\n- Steps are generated heuristically: linear and quadratic equations get clear algebraic steps. For very complex symbolic problems I show symbolic transforms and final answers.\n- Function analysis uses symbolic heuristics (derivatives, critical points, limits). For extremely complicated functions some parts may return 'unknown' — ask me and I'll help expand the analysis for that case.")
