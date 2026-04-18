import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandapower.networks as pn
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Smart Grid Fault Diagnostics", page_icon="⚡", layout="wide")

# ── 1. Configuration & Model Def ───────────────────────────────────────────
CFG = {
    "n_buses":         39,
    "n_features":      5,
    "window_size":     50,
    "gcn_hidden1":     64,
    "gcn_hidden2":     128,
    "lstm_hidden":     256,
    "lstm_layers":     2,
    "dropout_lstm":    0.0, # Use 0 for eval
    "dropout_mlp":     0.0,
    "fault_types":     ["SLG", "LL", "DLG", "3PH"],
    "model_path":      Path("data/best_model.pt"),
    "data_path":       Path("data/fault_dataset.npz"),
}
FEAT_NAMES = ["Voltage Mag (pu)", "Voltage Angle (rad)", "Active Power (MW)", "Reactive Power (MVar)", "Freq Deviation (Hz)"]
FAULT_NAMES = CFG["fault_types"] + ["Normal"]

class STGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_buses    = cfg["n_buses"]
        n_feat     = cfg["n_features"]
        T          = cfg["window_size"]
        h1         = cfg["gcn_hidden1"]
        h2         = cfg["gcn_hidden2"]
        h_lstm     = cfg["lstm_hidden"]
        n_lstm     = cfg["lstm_layers"]
        drop_lstm  = cfg["dropout_lstm"]
        drop_mlp   = cfg["dropout_mlp"]
        n_loc      = n_buses + 1
        n_type     = len(cfg["fault_types"]) + 1

        self.gcn1 = GCNConv(n_feat * T, h1)
        self.gcn2 = GCNConv(h1, h2)
        self.bn1  = nn.BatchNorm1d(h1)
        self.bn2  = nn.BatchNorm1d(h2)

        self.lstm = nn.LSTM(
            input_size  = h2, hidden_size = h_lstm, num_layers  = n_lstm,
            batch_first = True, dropout = drop_lstm if n_lstm > 1 else 0.0,
        )

        self.loc_head = nn.Sequential(
            nn.Linear(h_lstm, 128), nn.ReLU(), nn.Dropout(drop_mlp),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop_mlp), nn.Linear(64, n_loc),
        )

        self.type_head = nn.Sequential(
            nn.Linear(h_lstm, 64), nn.ReLU(), nn.Dropout(drop_mlp), nn.Linear(64, n_type),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = F.relu(self.bn1(self.gcn1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.gcn2(x, edge_index, edge_attr)))
        
        n_graphs = batch.max().item() + 1
        x = x.view(n_graphs, CFG["n_buses"], -1)
        
        lstm_out, _ = self.lstm(x)
        g = lstm_out.mean(dim=1)
        
        loc_logits  = self.loc_head(g)
        type_logits = self.type_head(g)
        return loc_logits, type_logits

# ── 2. Caching Data & Models ───────────────────────────────────────────────
@st.cache_resource
def load_graph():
    net = pn.case39()
    src, dst, weights = [], [], []
    for _, line in net.line.iterrows():
        i, j = int(line.from_bus), int(line.to_bus)
        z = max(np.sqrt(line.r_ohm_per_km**2 + line.x_ohm_per_km**2) * line.length_km, 1e-6)
        src += [i, j]; dst += [j, i]; weights += [1.0/z, 1.0/z]
    for _, trafo in net.trafo.iterrows():
        i, j = int(trafo.hv_bus), int(trafo.lv_bus)
        w = 1.0 / max(trafo.vk_percent / 100.0, 1e-6)
        src += [i, j]; dst += [j, i]; weights += [w, w]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight, src, dst

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGNN(CFG).to(device)
    if CFG["model_path"].exists():
        model.load_state_dict(torch.load(CFG["model_path"], map_location=device))
    model.eval()
    return model, device

@st.cache_data
def load_dataset():
    if not CFG["data_path"].exists():
        st.error("Dataset not found! Please ensure data/fault_dataset.npz exists.")
        st.stop()
    data = np.load(CFG["data_path"])
    # Return features, locs, types
    return data["X"], data["y_loc"], data["y_type"]

# Wait for Streamlit to trigger
edge_index, edge_weight, src, dst = load_graph()
model, device = load_model()
X_full, y_loc_full, y_type_full = load_dataset()

# Norm scaling approximation (normally done via Scikit-Learn scaler in notebook)
X_mean = X_full.mean(axis=(0, 1, 3), keepdims=True)
X_std  = X_full.std(axis=(0, 1, 3), keepdims=True) + 1e-6

# ── 3. UI Layout ───────────────────────────────────────────────────────────
st.title("⚡ Smart Grid Fault Diagnostics System")
st.markdown("""
Welcome to the interactive **Live AI Simulator** for power grid fault detection.
This app streams simulated 50ms (5-feature) time-windows from the grid to our **Spatial-Temporal Graph Neural Network**, which automatically identifies anomalies, classifies the fault type, and pinpoints the exact affected bus.
""")

st.sidebar.header("🕹️ Simulation Controls")

# Session state to hold current simulation index
if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = 0

def generate_random_sim():
    st.session_state.sim_idx = int(np.random.randint(0, len(X_full)))

st.sidebar.button("🔄 Trigger Random Fault / Sensor Stream", on_click=generate_random_sim, use_container_width=True)

# Select manual index
idx = st.sidebar.number_input("Or input sample ID manually (0 to N):", 0, len(X_full)-1, st.session_state.sim_idx)
st.session_state.sim_idx = idx

# Extracted sample
X_sample = X_full[idx]
loc_true = int(y_loc_full[idx])
type_true = int(y_type_full[idx])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Ground Truth (Hidden from Model)")
if loc_true == -1:
    st.sidebar.success("✅ **Status:** Normal Operation")
else:
    st.sidebar.error(f"⚠️ **Status:** FAULT\n\n**Type:** {FAULT_NAMES[type_true]}\n\n**Location:** Bus {loc_true}")

# ── 4. Model Inference ─────────────────────────────────────────────────────
# Normalize
X_norm = (X_sample - X_mean[0,0,:,0].reshape(1,5,1)) / X_std[0,0,:,0].reshape(1,5,1)
# PyG Data
x_flat = torch.tensor(X_norm.reshape(CFG["n_buses"], -1), dtype=torch.float)
data_obj = Data(x=x_flat, edge_index=edge_index, edge_attr=edge_weight)
data_obj = data_obj.to(device)

with torch.no_grad():
    loc_logits, type_logits = model(data_obj)
    
    loc_probs = F.softmax(loc_logits, dim=-1)[0].cpu().numpy()
    type_probs = F.softmax(type_logits, dim=-1)[0].cpu().numpy()

    loc_pred = int(np.argmax(loc_probs))
    type_pred = int(np.argmax(type_probs))
    
    loc_pred_adj = -1 if loc_pred == CFG["n_buses"] else loc_pred
    loc_conf = loc_probs[loc_pred] * 100
    type_conf = type_probs[type_pred] * 100

# ── 5. Main Dashboard ──────────────────────────────────────────────────────
col1, col2 = st.columns((1, 2))

with col1:
    st.subheader("🤖 AI Diagnostic Output")
    
    if type_pred == 4: # Normal
        st.success(f"### ✅ Normal Operation\nConfidence: **{type_conf:.1f}%**")
    else:
        st.error(f"### 🚨 System Fault Detected!\n**Fault Type:** {FAULT_NAMES[type_pred]} ({type_conf:.1f}% confidence)\n\n**Localization:** Bus {loc_pred_adj} ({loc_conf:.1f}% confidence)")

    st.markdown("---")
    st.write("**Top 3 Suspected Buses (Probability):**")
    sorted_locs = np.argsort(loc_probs)[::-1]
    for i in range(3):
        b = sorted_locs[i]
        b_name = "Normal" if b == CFG["n_buses"] else f"Bus {b}"
        st.progress(float(loc_probs[b]), text=f"{b_name}: {loc_probs[b]*100:.1f}%")

with col2:
    st.subheader("🌐 Grid Topology Topology Map")
    # NetworkX plot with Plotly
    G = nx.Graph()
    G.add_nodes_from(range(CFG["n_buses"]))
    for u, v in zip(src, dst):
        G.add_edge(u, v)
    
    pos = nx.kamada_kawai_layout(G)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_colors = []
    for node in G.nodes():
        if node == loc_pred_adj and type_pred != 4:
            node_colors.append('red') # Model prediction
        elif node == loc_true and loc_true != -1:
             node_colors.append('orange') # Ground truth (if model missed it)
        else:
            node_colors.append('#1f77b4') # Normal

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        hoverinfo='text',
        text=[f"{n}" for n in G.nodes()],
        textposition="bottom center",
        marker=dict(showscale=False, color=node_colors, size=15,
                    line_width=2, line_color='white')
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400
                 ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📈 Live PMU Sensor Dashboard")
st.markdown("Viewing a 50ms sliding window snapshot. If a fault is present, waveforms exhibit severe transient perturbations.")

# Let the user select a feature and bus to view
row_col1, row_col2 = st.columns(2)
with row_col1:
    sel_feat = st.selectbox("Select Feature to Plot:", range(5), format_func=lambda x: FEAT_NAMES[x])
with row_col2:
    bus_options = [loc_pred_adj] if loc_pred_adj != -1 else [0]
    bus_options += [i for i in range(CFG["n_buses"]) if i not in bus_options]
    sel_bus = st.selectbox("Select Bus to Monitor:", bus_options, format_func=lambda x: f"Bus {x} " + ("(Predicted Fault Location)" if x == loc_pred_adj else "(Normal)"))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=X_sample[sel_bus, sel_feat, :], mode='lines', name=f'{FEAT_NAMES[sel_feat]} at Bus {sel_bus}', line=dict(width=3, color='royalblue')))
fig2.update_layout(title="", xaxis_title="Timestep (100Hz = 10ms per tick)", yaxis_title="Magnitude", height=300, margin=dict(b=0,t=30))
st.plotly_chart(fig2, use_container_width=True)
