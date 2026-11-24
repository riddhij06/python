"""
ClayPrint AI - Smart Material, Cost and Carbon Footprint Calculator
for 3D Printed Wall Panels (Shadu Mati / clay-based vs concrete).

Run:  streamlit run app.py
"""
import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="ClayPrint AI", page_icon="🏗️", layout="wide")

# ----------------------------------------------------------------- THEME
st.markdown("""
<style>
.block-container{padding-top:1rem}
.hdr{background:linear-gradient(90deg,#12263f 0%,#1f4068 70%,#e07a1f 140%);padding:18px 26px;border-radius:10px;
     color:#fff;margin-bottom:12px;border-bottom:5px solid #e07a1f}
.hdr h1{margin:0;font-size:1.9rem;color:#fff}.hdr p{margin:2px 0 0;color:#d6e2f0}
.tag-exp{background:#1b7f3b;color:#fff;padding:2px 8px;border-radius:4px;font-size:.78rem}
.tag-ai{background:#7a3db8;color:#fff;padding:2px 8px;border-radius:4px;font-size:.78rem}
.tag-usr{background:#555;color:#fff;padding:2px 8px;border-radius:4px;font-size:.78rem}
div[data-testid="stMetric"]{background:#f1f5fa;border-left:5px solid #1f4068;padding:8px 12px;border-radius:6px}
div[data-testid="stMetric"] *{color:#12263f !important}
</style>
<div class="hdr"><h1>🏗️ ClayPrint AI</h1>
<p>Smart Material, Cost &amp; Carbon Footprint Calculator for 3D-Printed Wall Panels (Shadu Mati vs Concrete)</p></div>
""", unsafe_allow_html=True)

TAG_EXP = '<span class="tag-exp">EXPERIMENTAL (user data)</span>'
TAG_AI = '<span class="tag-ai">AI PREDICTION – not a test result</span>'
TAG_USR = '<span class="tag-usr">USER-SPECIFIED</span>'

FEATURES = ["shadu_mati", "crushed_sand", "fly_ash", "fibre", "water", "curing_days"]
TARGETS = {"compressive_strength_MPa": "Compressive strength (MPa)",
           "water_absorption_pct": "Water absorption (%)",
           "shrinkage_pct": "Shrinkage (%)",
           "printability_score": "Printability (your score scale)"}
CLAY_ING = ["Shadu Mati", "Crushed sand", "Fly ash", "Fibres", "Water"]
CON_ING = ["Cement", "Sand", "Aggregates", "Water", "Fly ash", "Admixtures"]

# ------------------------------------------------------------- DEFAULTS
# IMPORTANT: these are INDICATIVE PLACEHOLDERS (typical literature ranges), NOT verified project data.
# Replace with supplier quotes, measured densities and EPD / ICE / national LCI values.
MAT_DEFAULT = pd.DataFrame({
    "Material": ["Shadu Mati", "Crushed sand", "Fly ash", "Fibres", "Water",
                 "Cement", "Sand", "Aggregates", "Admixtures"],
    "Density (kg/m³)": [2600, 2650, 2200, 1100, 1000, 3150, 2650, 2650, 1100],
    "Rate (₹/kg)": [2.0, 1.2, 1.5, 80.0, 0.05, 7.0, 1.2, 0.9, 120.0],
    "EF (kgCO₂e/kg)": [0.010, 0.008, 0.008, 0.50, 0.0003, 0.90, 0.005, 0.0075, 1.20],
    "Embodied energy (MJ/kg)": [0.15, 0.10, 0.10, 10.0, 0.005, 5.0, 0.08, 0.10, 20.0],
})
PROC_DEFAULT = dict(
    waste=5.0, tariff=8.0, grid_ef=0.82, printer_kw=3.0, pump_kw=1.5, speed=60.0, overhead=20.0,
    mix_kwh_t=6.0, workers=3, lab_rate=150.0, handling_h=1.5, equip_rate=400.0,
    haul_km=50.0, deliv_km=30.0, tkm_rate=4.0, tkm_ef=0.09, tkm_mj=1.2,
    cure_clay=20.0, cure_con=150.0)

# ------------------------------------------------------------ HELPERS
def fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def mix_density(frac, dens):
    """1/rho = sum(w_i/rho_i)  (ideal volume additivity, zero air voids)."""
    return 1.0 / sum(frac[k] / dens[k] for k in frac)


def geometry(W, H, T, layer, nozzle, infill, n_shell, n, speed, overhead):
    V_panel = W * H * T / 1e9
    f_shell = min(1.0, 2 * n_shell * nozzle / T)
    frac = f_shell + (1 - f_shell) * infill / 100.0
    V_print = V_panel * frac
    layers = math.ceil(H / layer)
    path_mm = V_print * 1e9 / (nozzle * layer)
    t_h = path_mm / speed / 3600.0 * (1 + overhead / 100.0)
    return dict(V_panel=V_panel, f_shell=f_shell, frac=frac, V_print=V_print, layers=layers,
                path_m=path_mm / 1000, t_panel_h=t_h, n=n, V_tot=V_print * n,
                print_h=t_h * n, area=W * H / 1e6)


def evaluate(net, M, P, geo, curing_l):
    """Cost, carbon, energy, water, waste for a dict {material: net kg}."""
    w = P["waste"] / 100
    gross = {k: v * (1 + w) for k, v in net.items()}
    net_m, gross_m = sum(net.values()), sum(gross.values())
    raw_net = sum(v * M.loc[k, "rate"] for k, v in net.items())
    waste_cost = raw_net * w
    t = geo["print_h"]
    kwh_p = (P["printer_kw"] + P["pump_kw"]) * t
    kwh_m = P["mix_kwh_t"] * gross_m / 1000
    kwh = kwh_p + kwh_m
    tkm_in, tkm_out = gross_m / 1000 * P["haul_km"], net_m / 1000 * P["deliv_km"]
    cost = {"Raw materials (net)": raw_net, "Wastage": waste_cost,
            "Electricity": kwh * P["tariff"],
            "Labour": (t + P["handling_h"] * geo["n"]) * P["workers"] * P["lab_rate"],
            "Equipment": t * P["equip_rate"],
            "Transportation": (tkm_in + tkm_out) * P["tkm_rate"]}
    a1 = sum(v * M.loc[k, "ef"] for k, v in gross.items())
    env = {"A1 Materials": a1, "A2 Inbound transport": tkm_in * P["tkm_ef"],
           "A3 Mixing + printing electricity": kwh * P["grid_ef"],
           "A4 Delivery to site": tkm_out * P["tkm_ef"]}
    mj = {"Materials": sum(v * M.loc[k, "ee"] for k, v in gross.items()),
          "Electricity": kwh * 3.6, "Transport": (tkm_in + tkm_out) * P["tkm_mj"]}
    water_l = gross.get("Water", 0) + curing_l * geo["n"]
    return dict(net=net, gross=gross, net_m=net_m, gross_m=gross_m, cost=cost, total=sum(cost.values()),
                env=env, co2=sum(env.values()), mj=mj, energy=sum(mj.values()), kwh=kwh,
                water_l=water_l, waste_kg=gross_m - net_m, tkm=tkm_in + tkm_out)


# ----------------------------------------------------------------- ML
@st.cache_resource(show_spinner=False)
def train_target(df, target):
    d = df[FEATURES + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    if n < 6:
        return None
    X, y = d[FEATURES].values, d[target].values
    models = {"Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
              "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
              "Linear Regression": make_pipeline(StandardScaler(), LinearRegression())}
    cv = KFold(n_splits=min(5, n), shuffle=True, random_state=42)
    rows = []
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=cv)
        rows.append({"Model": name, "CV R²": r2_score(y, p), "CV MAE": mean_absolute_error(y, p)})
        m.fit(X, y)
    met = pd.DataFrame(rows)
    best = met.sort_values("CV R²", ascending=False).iloc[0]["Model"]
    return dict(models=models, metrics=met, best=best, n=n)


def predict_all(res, x):
    out = {}
    for name, m in res["models"].items():
        out[name] = float(m.predict(x)[0])
    rf = res["models"]["Random Forest"]
    out["RF_std"] = float(np.std([t.predict(x)[0] for t in rf.estimators_]))
    return out


# ------------------------------------------------------------- REPORTS
def build_excel(sections):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for title, df in sections:
            name = "".join(c for c in title if c.isalnum() or c in " _-")[:31]
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    60, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 3)
    return bio.getvalue()


def build_pdf(sections, notes, fig_data):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ss = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title="ClayPrint AI Report")
    el = [Paragraph("ClayPrint AI – Engineering Report", ss["Title"]),
          Paragraph(f"3D-printed wall panels: Shadu Mati vs Concrete. Generated {dt.datetime.now():%d %b %Y %H:%M}",
                    ss["Normal"]), Spacer(1, 6)]
    for title, df in sections:
        el.append(Paragraph(title, ss["Heading2"]))
        data = [list(map(str, df.columns))] + [[str(v) for v in r] for r in df.values.tolist()]
        ncol = len(df.columns)
        cw = (A4[0] - 30 * mm) / ncol
        t = Table(data, colWidths=[cw] * ncol, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12263f")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 7),
                               ("GRID", (0, 0), (-1, -1), .3, colors.grey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f9")]),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el += [t, Spacer(1, 8)]
    if fig_data is not None:
        fig, ax = plt.subplots(figsize=(7, 3))
        x = np.arange(len(fig_data))
        ax.bar(x - .2, fig_data["Clay"], .4, label="Clay (Shadu Mati)", color="#c0773a")
        ax.bar(x + .2, fig_data["Concrete"], .4, label="Concrete", color="#5b6b7c")
        ax.set_xticks(x)
        ax.set_xticklabels(fig_data["Metric"], fontsize=7)
        ax.legend(fontsize=7)
        ax.set_title("Clay vs concrete (same panel design)", fontsize=9)
        ib = io.BytesIO()
        fig.tight_layout()
        fig.savefig(ib, format="png", dpi=150)
        plt.close(fig)
        ib.seek(0)
        el.append(Image(ib, width=170 * mm, height=73 * mm))
    el.append(PageBreak())
    el.append(Paragraph("Formulas, assumptions and data provenance", ss["Heading2"]))
    for n in notes:
        el.append(Paragraph("• " + n, ss["Normal"]))
        el.append(Spacer(1, 3))
    doc.build(el)
    return buf.getvalue()


# ================================================================ TABS
tabs = st.tabs(["1 Inputs", "2 Rates & Factors", "3 Quantities", "4 Cost", "5 AI Prediction",
                "6 Clay vs Concrete", "7 Environment", "8 AI Optimisation", "9 Reports & Assumptions"])
sections = []

# ------------------------------------------------------------ 1 INPUTS
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Clay mix (% by mass of wet mix)")
        pc = {}
        defaults = [30.0, 30.0, 10.0, 0.5, 29.5]
        for i, nme in enumerate(CLAY_ING):
            pc[nme] = st.number_input(f"{nme} (%)", 0.0, 100.0, defaults[i], 0.1, key=f"pc{i}")
        tot = sum(pc.values())
        if abs(tot - 100) > 0.01:
            st.warning(f"Clay percentages sum to {tot:.2f}%. They will be normalised to 100%.")
        curing_days = st.number_input("Curing / test age (days)", 1, 365, 28)
        dens_override = st.number_input("Measured fresh density of clay mix (kg/m³, 0 = compute from ingredients)",
                                        0.0, 4000.0, 0.0, 10.0,
                                        help="Computed value assumes zero air voids; a measured value is more reliable.")
    with c2:
        st.subheader("Concrete mix (kg per m³)")
        cdef = [400.0, 800.0, 1000.0, 180.0, 0.0, 4.0]
        kc = {nme: st.number_input(f"{nme} (kg/m³)", 0.0, 3000.0, cdef[i], 1.0, key=f"kc{i}")
              for i, nme in enumerate(CON_ING)}
        st.markdown(f"Concrete properties {TAG_USR}", unsafe_allow_html=True)
        con_fc = st.number_input("Concrete compressive strength (MPa) – from mix design / tests", 0.0, 150.0, 30.0)
        con_abs = st.number_input("Concrete water absorption (%) (0 = unknown)", 0.0, 30.0, 0.0)
        con_shr = st.number_input("Concrete shrinkage (%) (0 = unknown)", 0.0, 5.0, 0.0, 0.01)
    st.subheader("Wall-panel design")
    d1, d2, d3, d4 = st.columns(4)
    W = d1.number_input("Panel width (mm)", 100.0, 20000.0, 1200.0, 10.0)
    H = d2.number_input("Panel height (mm)", 100.0, 20000.0, 2400.0, 10.0)
    T = d3.number_input("Panel thickness (mm)", 20.0, 1000.0, 150.0, 5.0)
    n_pan = d4.number_input("Number of panels", 1, 100000, 10)
    e1, e2, e3, e4 = st.columns(4)
    layer = e1.number_input("Layer thickness (mm)", 2.0, 50.0, 10.0, 0.5)
    nozzle = e2.number_input("Nozzle width (mm)", 5.0, 100.0, 25.0, 1.0)
    infill = e3.number_input("Infill (%)", 0.0, 100.0, 40.0, 1.0)
    n_shell = e4.number_input("Perimeter shells per face", 1, 5, 1)
    if layer > 0.75 * nozzle:
        st.warning("Layer thickness > 75% of nozzle width: typically poor layer bonding / printability.")
    if 2 * n_shell * nozzle > T:
        st.warning("Shell walls fill the whole thickness; infill setting has no effect.")

# ------------------------------------------------- 2 RATES & FACTORS
with tabs[1]:
    st.info("⚠️ Defaults are **indicative placeholders** (typical literature ranges) – they are **not verified data**. "
            "Replace with supplier quotes, measured densities and values from your EPD / ICE database / national LCI "
            "(e.g. CEA grid factor, IPCC, ICE). All fields are editable.")
    mat_ed = st.data_editor(MAT_DEFAULT, hide_index=True, disabled=["Material"], width="stretch", key="mat")
    M = mat_ed.rename(columns={"Density (kg/m³)": "dens", "Rate (₹/kg)": "rate", "EF (kgCO₂e/kg)": "ef",
                               "Embodied energy (MJ/kg)": "ee"}).set_index("Material")
    st.subheader("Process, labour and transport parameters")
    P = {}
    spec = [("waste", "Material wastage (%)"), ("tariff", "Electricity tariff (₹/kWh)"),
            ("grid_ef", "Grid emission factor (kgCO₂e/kWh)"), ("printer_kw", "Printer power (kW)"),
            ("pump_kw", "Pump / mixer-feeder power (kW)"), ("speed", "Print speed (mm/s)"),
            ("overhead", "Non-printing time overhead (%)"), ("mix_kwh_t", "Mixing energy (kWh per tonne)"),
            ("workers", "Workers"), ("lab_rate", "Labour rate (₹/h per worker)"),
            ("handling_h", "Handling/finishing time (h per panel)"), ("equip_rate", "Equipment cost (₹/h of printing)"),
            ("haul_km", "Material haul distance (km)"), ("deliv_km", "Panel delivery distance (km)"),
            ("tkm_rate", "Freight cost (₹/t·km)"), ("tkm_ef", "Freight emission factor (kgCO₂e/t·km)"),
            ("tkm_mj", "Freight energy (MJ/t·km)"), ("cure_clay", "Curing/cleaning water – clay (L/panel)"),
            ("cure_con", "Curing/cleaning water – concrete (L/panel)")]
    cols = st.columns(4)
    for i, (k, lab) in enumerate(spec):
        P[k] = cols[i % 4].number_input(lab, 0.0, 1e7, float(PROC_DEFAULT[k]), key="p_" + k)
    if P["speed"] <= 0:
        st.error("Print speed must be > 0")
        st.stop()
    sections.append(("Rates and Factors", mat_ed.round(4)))
    sections.append(("Process Parameters", pd.DataFrame({"Parameter": [l for _, l in spec],
                                                          "Value": [P[k] for k, _ in spec]})))

# --------------------------------------------------- derived geometry
geo = geometry(W, H, T, layer, nozzle, infill, n_shell, n_pan, P["speed"], P["overhead"])
frac_c = {k: v / sum(pc.values()) for k, v in pc.items()} if sum(pc.values()) > 0 else None
if frac_c is None:
    st.error("Enter clay composition.")
    st.stop()
dens_map = {k: M.loc[k, "dens"] for k in CLAY_ING}
rho_calc = mix_density(frac_c, dens_map)
rho_clay = dens_override if dens_override > 0 else rho_calc
net_clay = {k: geo["V_tot"] * rho_clay * frac_c[k] for k in CLAY_ING}
net_con = {k: geo["V_tot"] * kc[k] for k in CON_ING if kc[k] > 0}
rho_con = sum(kc.values())
clay = evaluate(net_clay, M, P, geo, P["cure_clay"])
con = evaluate(net_con, M, P, geo, P["cure_con"])

# -------------------------------------------------------- 3 QUANTITIES
with tabs[2]:
    st.subheader("Panel geometry and print parameters")
    g = pd.DataFrame({"Quantity": ["Gross panel volume (m³/panel)", "Shell fraction of thickness", "Printed solid fraction",
                                   "Printed volume per panel (m³)", "Total printed volume (m³)", "Number of layers",
                                   "Print path per panel (m)", "Print time per panel (h)", "Total print time (h)",
                                   "Panel face area (m²/panel)"],
                      "Value": [geo["V_panel"], geo["f_shell"], geo["frac"], geo["V_print"], geo["V_tot"], geo["layers"],
                                geo["path_m"], geo["t_panel_h"], geo["print_h"], geo["area"]]})
    g["Value"] = g["Value"].map(lambda v: f"{v:,.4f}")
    st.dataframe(g, hide_index=True, width="stretch")
    sections.append(("Geometry", g))
    with st.expander("Formulas used"):
        st.latex(r"V_{panel}=W\,H\,t \qquad f_{shell}=\min\!\left(1,\tfrac{2\,n_s\,w_n}{t}\right)")
        st.latex(r"V_{print}=V_{panel}\left[f_{shell}+(1-f_{shell})\,\tfrac{infill}{100}\right]")
        st.latex(r"L_{path}=\tfrac{V_{print}}{w_n\,h_l}\qquad t_{print}=\tfrac{L_{path}}{v}(1+\text{overhead})")
        st.latex(r"\tfrac{1}{\rho_{mix}}=\sum \tfrac{w_i}{\rho_i}\qquad m_i = V_{print}\,\rho_{mix}\,w_i")
        st.caption("Assumes solid shells on both faces and a uniform infill fraction; no air-void correction unless a "
                   "measured fresh density is entered.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clay mix density (kg/m³)", fmt(rho_clay, 0), "measured" if dens_override > 0 else "computed")
    m2.metric("Clay mass incl. wastage (kg)", fmt(clay["gross_m"], 1))
    m3.metric("Concrete mass incl. wastage (kg)", fmt(con["gross_m"], 1))
    qc = pd.DataFrame({"Ingredient": CLAY_ING, "Mix %": [frac_c[k] * 100 for k in CLAY_ING],
                       "Net (kg)": [clay["net"][k] for k in CLAY_ING],
                       "Gross incl. wastage (kg)": [clay["gross"][k] for k in CLAY_ING],
                       "Gross (g)": [clay["gross"][k] * 1000 for k in CLAY_ING],
                       "Per panel (kg)": [clay["gross"][k] / n_pan for k in CLAY_ING]}).round(3)
    st.markdown("**Clay (Shadu Mati) mix quantities**")
    st.dataframe(qc, hide_index=True, width="stretch")
    qk = pd.DataFrame({"Ingredient": list(net_con), "kg/m³": [kc[k] for k in net_con],
                       "Net (kg)": [net_con[k] for k in net_con],
                       "Gross incl. wastage (kg)": [con["gross"][k] for k in net_con],
                       "Gross (g)": [con["gross"][k] * 1000 for k in net_con],
                       "Per panel (kg)": [con["gross"][k] / n_pan for k in net_con]}).round(3)
    st.markdown("**Concrete mix quantities**")
    st.dataframe(qk, hide_index=True, width="stretch")
    sections += [("Clay Quantities", qc), ("Concrete Quantities", qk)]
    st.plotly_chart(px.pie(qc, names="Ingredient", values="Net (kg)", title="Clay mix by mass",
                           color_discrete_sequence=px.colors.sequential.Oranges_r), width="stretch")

# ---------------------------------------------------------------- 4 COST
with tabs[3]:
    cdf = pd.DataFrame({"Item": list(clay["cost"]) + ["TOTAL MANUFACTURING COST"],
                        "Clay (₹)": list(clay["cost"].values()) + [clay["total"]],
                        "Concrete (₹)": list(con["cost"].values()) + [con["total"]]}).round(2)
    st.dataframe(cdf, hide_index=True, width="stretch")
    sections.append(("Cost", cdf))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clay total (₹)", fmt(clay["total"], 0))
    k2.metric("Clay per panel (₹)", fmt(clay["total"] / n_pan, 0))
    k3.metric("Clay per m² face (₹)", fmt(clay["total"] / (geo["area"] * n_pan), 0))
    k4.metric("Concrete total (₹)", fmt(con["total"], 0))
    fig = go.Figure()
    for nm, r, col in [("Clay", clay, "#c0773a"), ("Concrete", con, "#5b6b7c")]:
        fig.add_bar(name=nm, x=list(r["cost"]), y=list(r["cost"].values()), marker_color=col)
    fig.update_layout(barmode="group", title="Cost breakdown (₹)")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Formulas used"):
        st.latex(r"C_{raw}=\sum m_{i,net}\,r_i \quad C_{waste}=C_{raw}\cdot\tfrac{w\%}{100}")
        st.latex(r"C_{elec}=\big[(P_{printer}+P_{pump})\,t_{print}+e_{mix}\,m_{gross}/1000\big]\,\text{tariff}")
        st.latex(r"C_{lab}=(t_{print}+t_{handling}\,N)\,n_{workers}\,r_{lab}\quad C_{equip}=t_{print}\,r_{equip}")
        st.latex(r"C_{trans}=\left(\tfrac{m_{gross}}{1000}D_{haul}+\tfrac{m_{net}}{1000}D_{deliv}\right) r_{t\cdot km}")

# ------------------------------------------------------- 5 AI PREDICTION
exp_vals = {}
pred_best, pred_tab = {}, {}
with tabs[4]:
    st.markdown(f"Training data = **your actual experimental results** {TAG_EXP}. "
                f"Model outputs are {TAG_AI}. No dataset is bundled: the app never invents experimental data.",
                unsafe_allow_html=True)
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=FEATURES + list(TARGETS))
    tmpl = pd.DataFrame(columns=FEATURES + list(TARGETS)).to_csv(index=False).encode()
    u1, u2 = st.columns([3, 1])
    up = u1.file_uploader("Upload experimental dataset (CSV / XLSX)", type=["csv", "xlsx"])
    u2.download_button("Download blank template", tmpl, "clayprint_dataset_template.csv", "text/csv")
    if up is not None and st.session_state.get("loaded") != up.name:
        try:
            df_up = pd.read_csv(up) if up.name.endswith("csv") else pd.read_excel(up)
            miss = [c for c in FEATURES if c not in df_up.columns]
            if miss:
                st.error(f"Missing required columns: {miss}")
            else:
                for c in TARGETS:
                    if c not in df_up.columns:
                        df_up[c] = np.nan
                st.session_state.data = df_up[FEATURES + list(TARGETS)]
                st.session_state.loaded = up.name
        except Exception as e:
            st.error(f"Could not read file: {e}")
    st.caption("Columns: composition in % by mass (shadu_mati, crushed_sand, fly_ash, fibre, water), curing_days, then "
               "measured targets. Targets may be partially empty (each property is trained on its own complete rows).")
    data = st.data_editor(st.session_state.data, num_rows="dynamic", width="stretch", key="ed")
    data = data.apply(pd.to_numeric, errors="coerce")
    results = {t: train_target(data, t) for t in TARGETS}
    cnt = pd.DataFrame({"Property": [TARGETS[t] for t in TARGETS],
                        "Complete rows": [int(data[FEATURES + [t]].dropna().shape[0]) for t in TARGETS]})
    st.dataframe(cnt, hide_index=True)
    if all(r is None for r in results.values()):
        st.warning("Need at least 6 complete experimental rows for a property before a model can be trained. "
                   "AI prediction is disabled until then.")
    else:
        st.subheader("Model validation (k-fold cross-validation on your data)")
        for t, r in results.items():
            if r:
                st.markdown(f"**{TARGETS[t]}** – n = {r['n']}, best model: **{r['best']}**")
                st.dataframe(r["metrics"].round(3), hide_index=True)
                if r["n"] < 20:
                    st.caption("⚠ Small dataset: CV metrics are unreliable; treat predictions as indicative only.")
        x = pd.DataFrame([[frac_c["Shadu Mati"] * 100, frac_c["Crushed sand"] * 100, frac_c["Fly ash"] * 100,
                           frac_c["Fibres"] * 100, frac_c["Water"] * 100, curing_days]], columns=FEATURES)
        st.subheader("Prediction for the current mix " )
        st.markdown(TAG_AI, unsafe_allow_html=True)
        rows = []
        for t, r in results.items():
            if not r:
                continue
            p = predict_all(r, x)
            lo, hi = data[FEATURES].min(), data[FEATURES].max()
            out = [c for c in FEATURES if not (lo[c] <= x[c].iloc[0] <= hi[c])]
            pred_best[t] = p[r["best"]]
            pred_tab[t] = p
            rows.append({"Property": TARGETS[t], "Random Forest": p["Random Forest"], "Decision Tree": p["Decision Tree"],
                         "Linear Regression": p["Linear Regression"], "Best model": r["best"],
                         "Selected value": p[r["best"]], "RF tree spread (±1σ)": p["RF_std"],
                         "Inside training range?": "yes" if not out else "NO: " + ", ".join(out)})
        pt = pd.DataFrame(rows).round(3)
        st.dataframe(pt, hide_index=True, width="stretch")
        pt2 = pt.copy()
        pt2.insert(1, "Type", "AI prediction")
        sections.append(("AI Predictions", pt2))
        if (pt["Inside training range?"] != "yes").any():
            st.warning("Extrapolation outside the experimental range – predictions are not reliable.")
        st.plotly_chart(px.bar(pt.melt(id_vars="Property", value_vars=["Random Forest", "Decision Tree", "Linear Regression"]),
                               x="Property", y="value", color="variable", barmode="group",
                               title="Model predictions (different units per property)"), width="stretch")
        with st.expander("Model details"):
            st.write("Inputs: composition (%) + curing days. Random Forest: 300 trees. Decision Tree: max depth 4. "
                     "Linear Regression: standardised features. Best model = highest cross-validated R².")
    st.subheader("Actual experimental results for this mix (optional)")
    st.markdown(TAG_EXP, unsafe_allow_html=True)
    ec = st.columns(4)
    for i, (t, lab) in enumerate(TARGETS.items()):
        if ec[i].checkbox(f"Measured: {lab}", key=f"has{i}"):
            exp_vals[t] = ec[i].number_input(lab, 0.0, 1e4, 0.0, key=f"ev{i}")

# --------------------------------------------- property provenance
def clay_prop(t):
    if t in exp_vals:
        return exp_vals[t], "Experimental (user-entered)"
    if t in pred_best:
        return pred_best[t], "AI prediction"
    return None, "not available"

clay_fc, fc_src = clay_prop("compressive_strength_MPa")
clay_abs, abs_src = clay_prop("water_absorption_pct")
clay_shr, shr_src = clay_prop("shrinkage_pct")

# ------------------------------------------------- 6 COMPARISON
with tabs[5]:
    st.markdown("Identical panel geometry, print path and panel count for both materials.")
    perMPa = lambda v, f: v / f if f else None
    comp = [
        ("Mix density (kg/m³)", rho_clay, rho_con, "Calculated / measured"),
        ("Total material mass incl. wastage (kg)", clay["gross_m"], con["gross_m"], "Calculated"),
        ("Total manufacturing cost (₹)", clay["total"], con["total"], "Calculated"),
        ("Cost per m² of wall (₹)", clay["total"] / (geo["area"] * n_pan), con["total"] / (geo["area"] * n_pan), "Calculated"),
        ("Energy: embodied + process (MJ)", clay["energy"], con["energy"], "Calculated"),
        ("Electricity use (kWh)", clay["kwh"], con["kwh"], "Calculated"),
        ("Carbon footprint (kgCO₂e)", clay["co2"], con["co2"], "Calculated (A1–A4)"),
        ("Water consumption (L)", clay["water_l"], con["water_l"], "Calculated"),
        ("Compressive strength (MPa)", clay_fc, con_fc, f"Clay: {fc_src}; Concrete: user-specified"),
        ("Water absorption (%)", clay_abs, con_abs or None, f"Clay: {abs_src}; Concrete: user-specified"),
        ("Shrinkage (%)", clay_shr, con_shr or None, f"Clay: {shr_src}; Concrete: user-specified"),
        ("Cost per MPa (₹/MPa)", perMPa(clay["total"], clay_fc), perMPa(con["total"], con_fc), "Derived"),
        ("kgCO₂e per MPa", perMPa(clay["co2"], clay_fc), perMPa(con["co2"], con_fc), "Derived"),
        ("Strength / density (MPa per 1000 kg/m³)", perMPa(clay_fc, rho_clay / 1000) if clay_fc else None,
         perMPa(con_fc, rho_con / 1000), "Derived")]
    cd = pd.DataFrame(comp, columns=["Metric", "Clay (Shadu Mati)", "Concrete", "Basis / source"])
    cd["Clay vs concrete (%)"] = [None if (a is None or b in (None, 0)) else (a / b - 1) * 100
                                 for a, b in zip(cd["Clay (Shadu Mati)"], cd["Concrete"])]
    show = cd.copy()
    for c in ["Clay (Shadu Mati)", "Concrete", "Clay vs concrete (%)"]:
        show[c] = show[c].map(lambda v: fmt(v, 2))
    st.dataframe(show, hide_index=True, width="stretch")
    sections.append(("Clay vs Concrete", show))
    if clay_fc is None:
        st.warning("Clay strength unavailable – provide experimental data or train a model (tab 5) for performance ratios.")
    keys = ["Total material mass incl. wastage (kg)", "Total manufacturing cost (₹)", "Energy: embodied + process (MJ)",
            "Carbon footprint (kgCO₂e)", "Water consumption (L)"]
    sub = cd[cd.Metric.isin(keys)]
    norm = pd.DataFrame({"Metric": ["Mass", "Cost", "Energy", "CO₂e", "Water"],
                         "Clay": (sub["Clay (Shadu Mati)"] / sub["Concrete"] * 100).values,
                         "Concrete": [100] * 5})
    figc = px.bar(norm.melt("Metric"), x="Metric", y="value", color="variable", barmode="group",
                  color_discrete_map={"Clay": "#c0773a", "Concrete": "#5b6b7c"},
                  title="Relative to concrete = 100 (lower is better)", labels={"value": "Index"})
    st.plotly_chart(figc, width="stretch")
    comp_fig_data = norm
    if clay_fc:
        radar = go.Figure()
        cats = ["Low cost", "Low CO₂e", "Low energy", "Light weight", "Low water"]
        radar.add_trace(go.Scatterpolar(r=[100 / v for v in norm["Clay"]], theta=cats, fill="toself", name="Clay (rel. to concrete=1)",
                                        line_color="#c0773a"))
        radar.add_trace(go.Scatterpolar(r=[1] * 5, theta=cats, fill="toself", name="Concrete", line_color="#5b6b7c"))
        radar.update_layout(title="Performance radar (outward = better than concrete)")
        st.plotly_chart(radar, width="stretch")
    st.caption("Strength is NOT normalised into any 'overall score' automatically: panels should be compared only at "
               "functionally equivalent strength / durability / thermal requirements. Concrete properties are user inputs.")

# -------------------------------------------------------- 7 ENVIRONMENT
with tabs[6]:
    st.caption("System boundary: cradle-to-site (A1 raw materials, A2 inbound transport, A3 mixing + printing electricity, "
               "A4 delivery). Excludes use phase, end-of-life and capital-equipment manufacture. Emission factors are "
               "those in tab 2 – verify before reporting.")
    ev = pd.DataFrame({"Stage": list(clay["env"]), "Clay (kgCO₂e)": list(clay["env"].values()),
                       "Concrete (kgCO₂e)": list(con["env"].values())})
    ev.loc[len(ev)] = ["TOTAL", clay["co2"], con["co2"]]
    ev = ev.round(2)
    st.dataframe(ev, hide_index=True, width="stretch")
    summ = pd.DataFrame({"Indicator": ["Carbon footprint (kgCO₂e)", "Per panel (kgCO₂e)", "Per m² wall (kgCO₂e)",
                                       "Embodied + process energy (MJ)", "Water consumption (L)", "Material waste (kg)",
                                       "Waste share of gross mass (%)"],
                         "Clay": [clay["co2"], clay["co2"] / n_pan, clay["co2"] / (geo["area"] * n_pan), clay["energy"],
                                  clay["water_l"], clay["waste_kg"], 100 * clay["waste_kg"] / clay["gross_m"]],
                         "Concrete": [con["co2"], con["co2"] / n_pan, con["co2"] / (geo["area"] * n_pan), con["energy"],
                                      con["water_l"], con["waste_kg"], 100 * con["waste_kg"] / con["gross_m"]]}).round(2)
    st.dataframe(summ, hide_index=True, width="stretch")
    sections += [("Carbon by Stage", ev), ("Environmental Summary", summ)]
    e1, e2 = st.columns(2)
    e1.plotly_chart(px.bar(ev[:-1].melt("Stage"), x="Stage", y="value", color="variable", barmode="group",
                           title="kgCO₂e by life-cycle stage",
                           color_discrete_sequence=["#c0773a", "#5b6b7c"]), width="stretch")
    mj = pd.DataFrame({"Source": list(clay["mj"]), "Clay": list(clay["mj"].values()), "Concrete": list(con["mj"].values())})
    e2.plotly_chart(px.bar(mj.melt("Source"), x="Source", y="value", color="variable", barmode="stack",
                           title="Energy (MJ)", color_discrete_sequence=["#c0773a", "#5b6b7c"]), width="stretch")
    with st.expander("Formulas used"):
        st.latex(r"CO_2e=\sum m_{i,gross}\,EF_i+ \text{tkm}\cdot EF_{freight}+kWh\cdot EF_{grid}")
        st.latex(r"E=\sum m_{i,gross}\,EE_i + 3.6\,kWh + \text{tkm}\cdot e_{freight}")
        st.latex(r"W=m_{water,gross}+W_{curing}\,N \qquad \text{Waste}=m_{gross}-m_{net}")
        st.caption("Unfired clay: no kiln energy is assumed. Carbonation uptake and fibre biogenic carbon are ignored.")

# ----------------------------------------------------- 8 OPTIMISATION
with tabs[7]:
    st.markdown(f"Searches compositions inside the **range of your experimental data**, ranks them by cost and CO₂e, and "
                f"keeps only those whose {TAG_AI} satisfy your requirements. Validate any recommendation by testing.",
                unsafe_allow_html=True)
    if all(results[t] is None for t in TARGETS):
        st.warning("Train at least one property model in tab 5 first (≥ 6 complete experimental rows).")
    else:
        o1, o2, o3, o4 = st.columns(4)
        req_fc = o1.number_input("Min compressive strength (MPa)", 0.0, 200.0, 5.0)
        req_abs = o2.number_input("Max water absorption (%)", 0.0, 100.0, 15.0)
        req_shr = o3.number_input("Max shrinkage (%)", 0.0, 20.0, 3.0)
        req_pr = o4.number_input("Min printability score (0 = ignore)", 0.0, 100.0, 0.0)
        alpha = st.slider("Weight on cost vs carbon (1 = cost only, 0 = carbon only)", 0.0, 1.0, 0.5, 0.05)
        nsamp = st.select_slider("Candidates to evaluate", [5000, 20000, 50000, 100000], 50000)
        if st.button("🔍 Run optimisation", type="primary"):
            rng = np.random.default_rng(1)
            cols4 = ["shadu_mati", "crushed_sand", "fly_ash", "fibre"]
            lo, hi = data[FEATURES].min(), data[FEATURES].max()
            S = np.column_stack([rng.uniform(lo[c], hi[c], nsamp) for c in cols4])
            wat = 100 - S.sum(1)
            ok = (wat >= lo["water"] - 1e-9) & (wat <= hi["water"] + 1e-9)
            S, wat = S[ok], wat[ok]
            X = pd.DataFrame(np.column_stack([S, wat, np.full(len(S), curing_days)]), columns=FEATURES)
            if len(X) == 0:
                st.error("No candidates satisfy the 100% mass balance within the data ranges.")
            else:
                feas = np.ones(len(X), bool)
                predcols = {}
                for t, kind, lim in [("compressive_strength_MPa", "min", req_fc), ("water_absorption_pct", "max", req_abs),
                                     ("shrinkage_pct", "max", req_shr), ("printability_score", "min", req_pr)]:
                    r = results[t]
                    if r is None:
                        continue
                    pv = r["models"][r["best"]].predict(X.values)
                    predcols[t] = pv
                    if kind == "min" and not (t == "printability_score" and lim == 0):
                        feas &= pv >= lim
                    elif kind == "max":
                        feas &= pv <= lim
                fr = S / 100
                fr_all = np.column_stack([fr, wat / 100])
                dens_v = np.array([M.loc[k, "dens"] for k in CLAY_ING])
                rho = 1 / (fr_all / dens_v).sum(1)
                perkg_cost = fr_all @ np.array([M.loc[k, "rate"] for k in CLAY_ING]) + P["haul_km"] / 1000 * P["tkm_rate"]
                perkg_co2 = fr_all @ np.array([M.loc[k, "ef"] for k in CLAY_ING]) + P["haul_km"] / 1000 * P["tkm_ef"]
                if dens_override > 0:
                    rho = np.full(len(rho), dens_override)
                cost_m3, co2_m3 = rho * perkg_cost, rho * perkg_co2
                cur_cost = rho_clay * sum(frac_c[k] * M.loc[k, "rate"] for k in CLAY_ING) + rho_clay * P["haul_km"] / 1000 * P["tkm_rate"]
                cur_co2 = rho_clay * sum(frac_c[k] * M.loc[k, "ef"] for k in CLAY_ING) + rho_clay * P["haul_km"] / 1000 * P["tkm_ef"]
                score = alpha * cost_m3 / cur_cost + (1 - alpha) * co2_m3 / cur_co2
                idx = np.where(feas)[0]
                if len(idx) == 0:
                    st.session_state.opt = "none"
                else:
                    idx = idx[np.argsort(score[idx])]
                    picked, seen = [], set()
                    for i in idx:
                        key = tuple(np.round(X.values[i, :5] / 2).astype(int))
                        if key in seen:
                            continue
                        seen.add(key)
                        picked.append(i)
                        if len(picked) == 5:
                            break
                    rows = []
                    for rank, i in enumerate(picked, 1):
                        r_ = {"Rank": rank, "Shadu Mati %": X.shadu_mati[i], "Crushed sand %": X.crushed_sand[i],
                              "Fly ash %": X.fly_ash[i], "Fibre %": X.fibre[i], "Water %": X.water[i],
                              "Material cost (₹/m³)": cost_m3[i],
                              "Cost saving (%)": 100 * (1 - cost_m3[i] / cur_cost),
                              "Material CO₂e (kg/m³)": co2_m3[i],
                              "CO₂e saving (%)": 100 * (1 - co2_m3[i] / cur_co2),
                              "Project CO₂e saving (kg)": (cur_co2 - co2_m3[i]) * geo["V_tot"]}
                        for t, pv in predcols.items():
                            r_["Pred. " + TARGETS[t]] = pv[i]
                        rows.append(r_)
                    st.session_state.opt = (pd.DataFrame(rows).round(3), cur_cost, cur_co2, int(feas.sum()), len(X))
        opt = st.session_state.get("opt")
        if opt == "none":
            st.error("No candidate meets all requirements (as predicted). Relax requirements or add experimental data.")
        elif opt:
            odf, cc, ce, nf, nt = opt
            st.success(f"{nf:,} of {nt:,} candidate mixes satisfy the requirements (per AI prediction).")
            st.markdown(f"Current mix (materials + inbound haul only): **₹{cc:,.0f}/m³**, **{ce:,.1f} kgCO₂e/m³**  ·  {TAG_AI}",
                        unsafe_allow_html=True)
            st.dataframe(odf, hide_index=True, width="stretch")
            sections.append(("AI Optimisation", odf))
            st.plotly_chart(px.scatter(odf, x="Material cost (₹/m³)", y="Material CO₂e (kg/m³)", text="Rank",
                                       title="Recommended mixes (cost vs carbon)"), width="stretch")
            st.caption("Optimisation uses a surrogate model: recommendations are hypotheses to be verified by laboratory "
                       "testing before use in structural or building applications.")

# ---------------------------------------------------------- 9 REPORTS
with tabs[8]:
    notes = [
        "Printed volume: V_print = W·H·t·[f_shell + (1−f_shell)·infill], f_shell = min(1, 2·n_s·w_nozzle / t).",
        "Mix density: 1/ρ = Σ(w_i/ρ_i) (no air voids) unless a measured fresh density is entered.",
        "Cost = raw materials + wastage + electricity + labour + equipment + transport (formulas in tab 4).",
        "Carbon (cradle-to-site, A1–A4) = Σ m_i·EF_i + freight tkm·EF + kWh·EF_grid. Embodied energy likewise.",
        "Default rates, densities and emission factors are INDICATIVE PLACEHOLDERS and not verified data; edit them in tab 2.",
        "Clay mechanical properties are either user-entered experimental results or AI predictions trained on the "
        "user's dataset; each is labelled in the reports. No experimental data is bundled with the software.",
        "Concrete strength / absorption / shrinkage are user-specified inputs, not computed.",
        "AI predictions are only valid inside the range of the training data and for the stated curing age.",
        "Print time assumes constant speed and a single-bead path with given overhead; fire/kiln energy not applicable.",
        "System boundary excludes use phase, end-of-life, capital goods and biogenic carbon accounting."]
    ass = pd.DataFrame({"Formulas, assumptions and provenance": notes})
    st.dataframe(ass, hide_index=True, width="stretch")
    prov = pd.DataFrame({"Property": ["Clay compressive strength", "Clay water absorption", "Clay shrinkage"],
                         "Value": [fmt(clay_fc), fmt(clay_abs), fmt(clay_shr)],
                         "Source": [fc_src, abs_src, shr_src]})
    sections.insert(0, ("Project Inputs", pd.DataFrame({
        "Parameter": ["Panel W×H×t (mm)", "Layer / nozzle (mm)", "Infill (%)", "Shell per face", "Panels", "Curing days"]
                     + [f"Clay: {k} (%)" for k in CLAY_ING] + [f"Concrete: {k} (kg/m³)" for k in CON_ING],
        "Value": [f"{W:g} × {H:g} × {T:g}", f"{layer:g} / {nozzle:g}", infill, n_shell, n_pan, curing_days]
                 + [round(frac_c[k] * 100, 3) for k in CLAY_ING] + [kc[k] for k in CON_ING]})))
    sections.append(("Property Provenance", prov))
    sections.append(("Experimental Dataset", data.round(4) if len(data) else pd.DataFrame(columns=FEATURES + list(TARGETS))))
    st.dataframe(prov, hide_index=True)
    pdf_sections = [(t, d) for t, d in sections if t not in ("Experimental Dataset",)]
    pdf_sections = [(t, d.round(3) if d.select_dtypes("number").shape[1] else d) for t, d in pdf_sections]
    b1, b2 = st.columns(2)
    b1.download_button("📄 Download PDF report", build_pdf(pdf_sections, notes, comp_fig_data),
                       "ClayPrintAI_report.pdf", "application/pdf")
    b2.download_button("📊 Download Excel report", build_excel(sections + [("Notes", ass)]),
                       "ClayPrintAI_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
