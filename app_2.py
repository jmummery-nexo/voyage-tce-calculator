import streamlit as st
import pandas as pd
import difflib, re
import json, uuid, datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional

# ==== Styling / page ====
st.set_page_config(page_title="Voyage TCE Tool", page_icon="🛳️", layout="wide")
st.title("🛳️ Voyage TCE & Target-Rate Calculator")

# Session storage for saved runs
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []  # list of dicts

# =========================
# Upload ports.csv (required)
# =========================
ports_file = st.file_uploader(
    "Upload your ports.csv (must contain Name, Latitude, Longitude columns)",
    type=["csv"]
)
if not ports_file:
    st.info("Please upload `ports.csv` to continue.")
    st.stop()

ports_df = pd.read_csv(ports_file)

def _col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of columns {candidates} in ports CSV.")

NAME_COL = _col(ports_df, ["Name", "name", "PORT", "Port", "PortName", "port_name"])
LAT_COL  = _col(ports_df, ["Latitude", "latitude", "lat", "Lat"])
LON_COL  = _col(ports_df, ["Longitude", "longitude", "lon", "Lon", "lng", "Lng"])

def _norm(s: str) -> str:
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def build_ports_index(df: pd.DataFrame) -> Dict[str, Tuple[Tuple[float, float], str]]:
    idx: Dict[str, Tuple[Tuple[float, float], str]] = {}
    for _, r in df.iterrows():
        name = str(r[NAME_COL]).strip()
        lat = float(r[LAT_COL]); lon = float(r[LON_COL])
        coord = (lon, lat)
        canonical = name
        keys = {name, name.casefold(), _norm(name)}
        base = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
        if base and base != name:
            keys |= {base, base.casefold(), _norm(base)}
        for k in keys:
            if k and k not in idx:
                idx[k] = (coord, canonical)
    return idx

PORTS_INDEX = build_ports_index(ports_df)
ALL_CANONICAL_NAMES = sorted({v[1] for v in PORTS_INDEX.values()})

def resolve_port_by_name(name: str) -> Tuple[Tuple[float, float], str]:
    for key in (name, name.casefold(), _norm(name)):
        if key in PORTS_INDEX:
            return PORTS_INDEX[key]
    best = difflib.get_close_matches(name, ALL_CANONICAL_NAMES, n=1, cutoff=0.6)
    if best:
        canon = best[0]
        return PORTS_INDEX[canon]
    raise KeyError(f"Port '{name}' not found. Closest: {difflib.get_close_matches(name, ALL_CANONICAL_NAMES, n=5, cutoff=0.0)}")

# ---- searoute import ----
try:
    import searoute as sr
except Exception as e:
    st.error("Python package `searoute` is not available. Install it: `pip install searoute`.")
    st.stop()

# =========================
# Domain model (your tables)
# =========================
@dataclass
class VesselCard:
    main_engine_mt_perd: float
    aux_engine_mt_perd: float
    loading_mt_perd: float
    discharging_mt_perd: float
    idle_mt_perd: float
    cargo_heating_mt_perd: float
    tank_cleaning_mt_perd: float
    eca_sailing_main_mt_perd: Optional[float] = None

VESSELS: Dict[str, VesselCard] = {
    "9500-11000 DWT":    VesselCard(18.0, 0.5, 1.5, 2.5, 1.3, 2.5, 2.5, 18.0),
    "13000 DWT":         VesselCard(13.2, 1.8, 2.16, 3.24, 1.32, 5.5, 5.6, 13.2),
    "NON ECO MR":        VesselCard(26.0, 0.5, 8.5, 19.0, 6.5, 17.5, 8.5, 26.0),
    "ECO MR":            VesselCard(26.0, 0.5, 4.7, 7.0, 4.0, 7.0, 7.6, 26.0),
    "31000 - 33000 DWT": VesselCard(25.0, 0.5, 4.1, 7.5, 3.9, 20.0, 6.0, 26.0),
    "25000 - 27000 DWT": VesselCard(17.0, 0.5, 3.0, 7.5, 3.6, 7.0, 4.0, 17.0),
    "18000 - 22000 DWT": VesselCard(18.0, 0.5, 3.0, 5.6, 1.6, 7.0, 3.0, 18.0),
    "Up-to 7000 DWT":    VesselCard(9.5, 0.5, 1.4, 2.4, 0.6, 7.0, 1.8, 9.5),
    "3500-5500 DWT":     VesselCard(8.0, 0.5, 1.2, 2.3, 0.5, 4.2, 1.0, 8.0),
}

LegType = Literal["laden", "ballast"]

@dataclass
class Prices:
    vlsfo_usd_mt: float = 650.0
    mgo_usd_mt: float   = 850.0

@dataclass
class PortOps:
    load_days: float
    disch_days: float
    idle_days: float = 0.0
    heating_days_port: float = 0.0
    cleaning_days: float = 0.0
    heating_days_sea: float = 0.0
    heating_eca_frac_sea: Optional[float] = None
    port_fees_usd: float = 0.0
    canal_fees_usd: float = 0.0
    other_costs_usd: float = 0.0
    misc_costs_usd: float = 0.0

@dataclass
class HouseView:
    commission_pct: float = 0.0375
    weather_pad_frac: float = 0.0
    eca_frac_laden: float = 0.30
    eca_frac_ballast: float = 0.30

@dataclass
class LegByName:
    origin: str
    dest: str
    leg_type: LegType
    speed_kn: float
    eca_frac: Optional[float] = None

def nm_distance_by_name(origin_name: str, dest_name: str) -> float:
    (olon, olat), _ = resolve_port_by_name(origin_name)
    (dlon, dlat), _ = resolve_port_by_name(dest_name)
    feat = sr.searoute((olon, olat), (dlon, dlat), units="naut")
    return float(feat["properties"]["length"])

def price_split_cost(mt_non_eca: float, mt_eca: float, prices: Prices) -> float:
    return mt_non_eca * prices.vlsfo_usd_mt + mt_eca * prices.mgo_usd_mt

def calc_tce_usd_per_day_from_names(
    usd_per_ton: float,
    lifted_tonnes: float,
    vessel_name: str,
    legs: List[LegByName],
    prices: Prices,
    ports: PortOps,
    hv: HouseView,
):
    assert vessel_name in VESSELS, f"Unknown vessel '{vessel_name}'. Options: {list(VESSELS)}"
    v = VESSELS[vessel_name]
    me_eca = v.eca_sailing_main_mt_perd or v.main_engine_mt_perd

    gross = usd_per_ton * lifted_tonnes
    comm  = gross * hv.commission_pct
    net_freight = gross - comm

    total_nm = 0.0
    sea_days_total = 0.0
    sea_days_laden = 0.0
    sea_days_ballast = 0.0
    sea_cost = 0.0
    per_leg = []
    sea_days_eca_weighted = 0.0

    for i, leg in enumerate(legs, 1):
        nm = nm_distance_by_name(leg.origin, leg.dest)
        total_nm += nm
        sea_days = nm / (leg.speed_kn * 24.0)
        sea_days *= (1.0 + hv.weather_pad_frac)

        eca = leg.eca_frac if leg.eca_frac is not None else (
            hv.eca_frac_laden if leg.leg_type == "laden" else hv.eca_frac_ballast
        )
        sea_days_eca_weighted += sea_days * eca
        sea_days_total += sea_days
        if leg.leg_type == "laden": sea_days_laden += sea_days
        else:                        sea_days_ballast += sea_days

        me_non_eca_mt  = sea_days * (1.0 - eca) * v.main_engine_mt_perd
        me_eca_mt      = sea_days * eca * me_eca
        aux_non_eca_mt = sea_days * (1.0 - eca) * v.aux_engine_mt_perd
        aux_eca_mt     = sea_days * eca * v.aux_engine_mt_perd

        leg_cost = price_split_cost(me_non_eca_mt + aux_non_eca_mt,
                                    me_eca_mt + aux_eca_mt, prices)
        sea_cost += leg_cost

        per_leg.append({
            "leg": i, "origin": leg.origin, "dest": leg.dest, "type": leg.leg_type,
            "nm": round(nm, 1), "sea_days": round(sea_days, 3),
            "eca_frac": round(eca, 3), "bunker_cost_usd": round(leg_cost, 0),
        })

    # Sea heating (from ports.*)
    heating_days_sea = ports.heating_days_sea
    heating_eca_frac_sea = ports.heating_eca_frac_sea
    if heating_days_sea > 0:
        eca_share = (sea_days_eca_weighted / sea_days_total) if heating_eca_frac_sea is None else heating_eca_frac_sea
        heat_non_eca_mt = heating_days_sea * (1.0 - eca_share) * v.cargo_heating_mt_perd
        heat_eca_mt     = heating_days_sea * eca_share * v.cargo_heating_mt_perd
        sea_heat_cost = price_split_cost(heat_non_eca_mt, heat_eca_mt, prices)
        sea_cost += sea_heat_cost
    else:
        sea_heat_cost = 0.0

    # Port bunkers (MGO) — cleaning = cost, not time
    port_mt = (
        ports.load_days * v.loading_mt_perd
        + ports.disch_days * v.discharging_mt_perd
        + ports.idle_days * v.idle_mt_perd
        + ports.heating_days_port * v.cargo_heating_mt_perd
        + ports.cleaning_days * v.tank_cleaning_mt_perd
    )
    port_bunker_cost = port_mt * prices.mgo_usd_mt

    voyage_costs = (
        sea_cost
        + port_bunker_cost
        + ports.port_fees_usd
        + ports.canal_fees_usd
        + ports.other_costs_usd
        + ports.misc_costs_usd
    )

    port_days_total = ports.load_days + ports.disch_days + ports.idle_days + ports.heating_days_port
    voyage_days = sea_days_total + port_days_total

    net_after_costs = net_freight - voyage_costs
    tce = net_after_costs / voyage_days if voyage_days > 0 else float("nan")

    return {
        "route_summary": {
            "total_nm": round(total_nm, 1),
            "sea_days_total": round(sea_days_total, 3),
            "sea_days_laden": round(sea_days_laden, 3),
            "sea_days_ballast": round(sea_days_ballast, 3),
            "port_days_total": round(port_days_total, 3),
            "voyage_days_total": round(voyage_days, 3),
        },
        "revenue_costs": {
            "gross_freight_usd": round(gross, 0),
            "commissions_usd": round(comm, 0),
            "net_freight_usd": round(net_freight, 0),
            "sea_bunker_cost_usd": round(sea_cost - sea_heat_cost, 0),
            "sea_heating_cost_usd": round(sea_heat_cost, 0),
            "port_bunker_cost_usd": round(port_bunker_cost, 0),
            "port_fees_usd": round(ports.port_fees_usd, 0),
            "canal_fees_usd": round(ports.canal_fees_usd, 0),
            "other_costs_usd": round(ports.other_costs_usd, 0),
            "misc_costs_usd": round(ports.misc_costs_usd, 0),
            "voyage_costs_total_usd": round(voyage_costs, 0),
            "net_after_costs_usd": round(net_after_costs, 0),
        },
        "TCE_USD_per_day": round(tce, 2),
        "per_leg": per_leg,
    }

# ============== UI: Inputs ==============
with st.sidebar:
    st.header("Inputs")
    st.text_input("Scenario name", value="Run 1", key="scenario_name")

    vessel_name = st.selectbox(
        "Vessel", list(VESSELS.keys()),
        index=list(VESSELS.keys()).index("Up-to 7000 DWT") if "Up-to 7000 DWT" in VESSELS else 0
    )
    lifted_tonnes = st.number_input("Lifted tonnes", min_value=1.0, value=6000.0, step=100.0)
    usd_per_ton = st.number_input("Freight (USD/mt) for TCE calc", min_value=0.0, value=46.0, step=0.1)

    st.subheader("Fuel Prices")
    vlsfo = st.number_input("VLSFO (USD/mt)", min_value=0.0, value=739.0, step=1.0)
    mgo   = st.number_input("MGO (USD/mt)",   min_value=0.0, value=739.0, step=1.0)

    st.subheader("House View")
    commission = st.number_input("Commission (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.25) / 100.0
    weather_pad = st.number_input("Weather pad (fraction)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    eca_laden   = st.number_input("Default ECA frac (laden)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    eca_ballast = st.number_input("Default ECA frac (ballast)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    st.subheader("Port / Other Days")
    load_days = st.number_input("Loading days", min_value=0.0, value=1.66, step=0.01)
    disch_days = st.number_input("Discharging days", min_value=0.0, value=1.66, step=0.01)
    idle_days = st.number_input("Idle days", min_value=0.0, value=1.5, step=0.1)
    heating_days_port = st.number_input("Heating days (in port)", min_value=0.0, value=0.0, step=0.1)
    cleaning_days = st.number_input("Cleaning days (cost only)", min_value=0.0, value=1.2, step=0.1)
    heating_days_sea = st.number_input("Heating days (at sea)", min_value=0.0, value=4.0, step=0.1)
    heating_eca_frac_sea = st.number_input("Heating ECA fraction at sea (None=auto)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    st.subheader("Fees / Costs (USD)")
    port_fees = st.number_input("Port fees", min_value=0.0, value=25000.0, step=500.0)
    canal_fees = st.number_input("Canal fees", min_value=0.0, value=0.0, step=100.0)
    other_costs = st.number_input("Other costs", min_value=0.0, value=0.0, step=100.0)
    misc_costs = st.number_input("Misc costs", min_value=0.0, value=5000.0, step=100.0)

# Legs editor
st.subheader("Legs")
legs_df = pd.DataFrame([
    {"origin":"Thessaloniki", "dest":"Damietta", "type":"ballast", "speed_kn":12.0, "eca_frac":1.0},
    {"origin":"Damietta", "dest":"Barcelona", "type":"laden", "speed_kn":12.0, "eca_frac":1.0},
])
legs_df = st.data_editor(
    legs_df, num_rows="dynamic",
    column_config={
        "type": st.column_config.SelectboxColumn("type", options=["laden","ballast"]),
        "eca_frac": st.column_config.NumberColumn("eca_frac", min_value=0.0, max_value=1.0, step=0.05),
        "speed_kn": st.column_config.NumberColumn("speed_kn", min_value=1.0, max_value=30.0, step=0.5),
    },
    use_container_width=True
)

# Build objects
prices = Prices(vlsfo_usd_mt=vlsfo, mgo_usd_mt=mgo)
hv = HouseView(commission_pct=commission, weather_pad_frac=weather_pad, eca_frac_laden=eca_laden, eca_frac_ballast=eca_ballast)
ports = PortOps(
    load_days=load_days, disch_days=disch_days, idle_days=idle_days,
    heating_days_port=heating_days_port, cleaning_days=cleaning_days,
    heating_days_sea=heating_days_sea, heating_eca_frac_sea=heating_eca_frac_sea,
    port_fees_usd=port_fees, canal_fees_usd=canal_fees,
    other_costs_usd=other_costs, misc_costs_usd=misc_costs
)
legs: List[LegByName] = []
for _, r in legs_df.iterrows():
    legs.append(LegByName(origin=str(r["origin"]), dest=str(r["dest"]),
                          leg_type=str(r["type"]), speed_kn=float(r["speed_kn"]),
                          eca_frac=None if pd.isna(r["eca_frac"]) else float(r["eca_frac"])))

# ============== Actions / Tabs ==============
tab1, tab2 = st.tabs(["USD/mt → TCE", "Target TCE → USD/mt"])

with tab1:
    st.header("USD/mt → TCE")
    if st.button("Calculate TCE", type="primary"):
        res = calc_tce_usd_per_day_from_names(
            usd_per_ton=usd_per_ton,
            lifted_tonnes=lifted_tonnes,
            vessel_name=vessel_name,
            legs=legs,
            prices=prices,
            ports=ports,
            hv=hv
        )
        # Pretty print
        rs = res["route_summary"]; rc = res["revenue_costs"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Route Summary")
            st.write(pd.DataFrame([rs]).T.rename(columns={0:"value"}))
        with c2:
            st.subheader("Revenue & Costs (USD)")
            st.write(pd.DataFrame([rc]).T.rename(columns={0:"value"}))
        st.subheader("Per-leg breakdown")
        st.dataframe(pd.DataFrame(res["per_leg"]), use_container_width=True)
        st.metric("TCE (USD/day)", f"{res['TCE_USD_per_day']:,.2f}")

        # Export this table
        st.download_button("Download revenue & costs CSV",
                           data=pd.DataFrame([rc]).to_csv(index=False),
                           file_name="revenue_costs.csv", mime="text/csv")

        # Save run (in-session)
        if st.button("💾 Save this run", key="save_usdpt_to_tce"):
            run = {
                "id": str(uuid.uuid4())[:8],
                "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "scenario": st.session_state.get("scenario_name") or "",
                "flow": "usdpt_to_tce",
                "inputs": {
                    "vessel_name": vessel_name,
                    "lifted_tonnes": lifted_tonnes,
                    "usd_per_ton": usd_per_ton,
                    "prices": vars(prices),
                    "ports": vars(ports),
                    "hv": vars(hv),
                    "legs": [leg.__dict__ for leg in legs],
                },
                "outputs": res,
            }
            st.session_state.saved_runs.append(run)
            st.success(f"Saved run {run['id']}")

with tab2:
    st.header("Target TCE → required USD/mt")
    target_tce = st.number_input("Target TCE (USD/day)", min_value=0.0, value=11000.0, step=100.0)

    def usdpt_for_target_tce_from_names_using_current_inputs(
        target_tce_usd_day: float,
        lifted_tonnes: float,
        vessel_name: str,
        legs: List[LegByName],
        prices: Prices,
        ports: PortOps,
        hv: HouseView,
        round_to_cents: int = 2
    ) -> float:
        if lifted_tonnes <= 0:
            raise ValueError("lifted_tonnes must be > 0")
        if not (0.0 <= hv.commission_pct < 1.0):
            raise ValueError("commission_pct must be in [0,1)")
        zero_rate = calc_tce_usd_per_day_from_names(
            usd_per_ton=0.0,
            lifted_tonnes=lifted_tonnes,
            vessel_name=vessel_name,
            legs=legs,
            prices=prices,
            ports=ports,
            hv=hv
        )
        voyage_days = float(zero_rate["route_summary"]["voyage_days_total"])
        voyage_costs = float(zero_rate["revenue_costs"]["voyage_costs_total_usd"])
        denom = lifted_tonnes * (1.0 - hv.commission_pct)
        required_rate = (target_tce_usd_day * voyage_days + voyage_costs) / denom
        return round(required_rate, round_to_cents)

    if st.button("Backsolve USD/mt", type="primary"):
        req_rate = usdpt_for_target_tce_from_names_using_current_inputs(
            target_tce_usd_day=target_tce,
            lifted_tonnes=lifted_tonnes,
            vessel_name=vessel_name,
            legs=legs,
            prices=prices,
            ports=ports,
            hv=hv
        )
        st.metric("Required USD/mt", f"{req_rate:,.2f}")

        # Optional verification
        verify = calc_tce_usd_per_day_from_names(
            usd_per_ton=req_rate,
            lifted_tonnes=lifted_tonnes,
            vessel_name=vessel_name,
            legs=legs,
            prices=prices,
            ports=ports,
            hv=hv
        )
        st.caption(f"Backsolved TCE (USD/day): {verify['TCE_USD_per_day']:,.2f}")

        # Save backsolve run (in-session)
        if st.button("💾 Save this backsolve", key="save_tce_to_usdpt"):
            run = {
                "id": str(uuid.uuid4())[:8],
                "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "scenario": st.session_state.get("scenario_name") or "",
                "flow": "tce_to_usdpt",
                "inputs": {
                    "vessel_name": vessel_name,
                    "lifted_tonnes": lifted_tonnes,
                    "target_tce": target_tce,
                    "prices": vars(prices),
                    "ports": vars(ports),
                    "hv": vars(hv),
                    "legs": [leg.__dict__ for leg in legs],
                },
                "outputs": {
                    "required_usd_per_ton": req_rate,
                    "TCE_USD_per_day": verify["TCE_USD_per_day"],
                    "route_summary": verify["route_summary"],
                    "revenue_costs": verify["revenue_costs"],
                    "per_leg": verify["per_leg"],
                },
            }
            st.session_state.saved_runs.append(run)
            st.success(f"Saved run {run['id']}")

# ===== Saved runs viewer / export / import =====
with st.expander("📚 Saved runs (this session)", expanded=False):
    runs = st.session_state.saved_runs
    if runs:
        rows = []
        for r in runs:
            outputs = r.get("outputs", {})
            rev_costs = outputs.get("revenue_costs", {})
            rows.append({
                "id": r["id"],
                "ts": r["ts"],
                "scenario": r.get("scenario", ""),
                "flow": r.get("flow", ""),
                "USD/mt": r.get("inputs", {}).get("usd_per_ton"),
                "Target TCE": r.get("inputs", {}).get("target_tce"),
                "TCE (USD/day)": outputs.get("TCE_USD_per_day"),
                "Net after costs (USD)": rev_costs.get("net_after_costs_usd"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.download_button(
            "⬇️ Download all runs (JSON)",
            data=json.dumps(runs, indent=2),
            file_name="runs.json",
            mime="application/json",
        )

        up = st.file_uploader("⬆️ Import runs.json", type="json", key="import_runs")
        if up is not None:
            try:
                st.session_state.saved_runs = json.loads(up.read())
                st.success("Imported runs.json")
            except Exception as e:
                st.error(f"Import failed: {e}")
    else:
        st.info("No saved runs yet.")
