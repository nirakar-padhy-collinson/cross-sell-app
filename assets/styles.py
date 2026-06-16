CUSTOM_CSS = """
<style>
:root {
    --bg-top: #f8f4ee;
    --bg-bottom: #efe6d9;
    --surface: rgba(255, 252, 248, 0.82);
    --surface-strong: rgba(255, 252, 248, 0.94);
    --surface-dark: #13202d;
    --ink-strong: #182230;
    --ink-soft: #5b6778;
    --accent: #b07a3f;
    --accent-deep: #8d5f2d;
    --line: rgba(34, 45, 60, 0.10);
    --line-soft: rgba(34, 45, 60, 0.06);
    --success: #3f7b57;
    --amber: #b68643;
    --orange: #cb7a43;
    --danger: #b4584f;
    --shadow-lg: 0 28px 60px rgba(20, 34, 50, 0.10);
    --shadow-md: 0 18px 36px rgba(20, 34, 50, 0.08);
    --shadow-sm: 0 10px 24px rgba(20, 34, 50, 0.06);
}

html, body, [class*="css"] {
    font-family: "Source Sans Pro", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(190, 153, 102, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(17, 33, 49, 0.10), transparent 24%),
        linear-gradient(180deg, var(--bg-top) 0%, #f5ede3 46%, var(--bg-bottom) 100%);
}

.block-container {
    padding-top: 1.35rem;
    padding-right: 2.2rem;
    padding-bottom: 2.6rem;
    padding-left: 2.2rem;
    max-width: none;
}

[data-testid="stHeader"] {
    background: rgba(250, 246, 239, 0.74);
    backdrop-filter: blur(14px);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(17, 33, 49, 0.99) 0%, rgba(22, 42, 62, 0.99) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] > div:first-child {
    background:
        radial-gradient(circle at top right, rgba(176, 122, 63, 0.18), transparent 24%),
        linear-gradient(180deg, rgba(19, 32, 45, 1) 0%, rgba(22, 42, 62, 0.98) 100%);
}

[data-testid="stSidebar"] * {
    color: #f2ebe1;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
    padding-bottom: 1.4rem;
}

[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stButton,
[data-testid="stSidebar"] .stCaption {
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    min-height: 3rem;
    border-radius: 18px;
    background: rgba(255, 250, 242, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.10);
}

[data-testid="stSidebar"] .stButton > button {
    min-height: 3rem;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: linear-gradient(180deg, rgba(187, 137, 76, 0.98) 0%, rgba(141, 95, 45, 0.98) 100%);
    color: #fff8f0;
    font-weight: 700;
    box-shadow: 0 12px 24px rgba(6, 12, 22, 0.22);
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(255, 241, 225, 0.26);
    filter: brightness(1.03);
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    background: transparent;
    border-radius: 14px;
    padding: 0.2rem 0;
    margin-bottom: 0.2rem;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-selected="true"] {
    background: transparent;
}

[data-testid="stSidebar"] .stRadio label p {
    color: #f2ebe1;
    font-weight: 700;
    line-height: 1.45;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] .stRadio > label {
    color: rgba(233, 223, 209, 0.42);
    font-weight: 700;
    margin-bottom: 0.45rem;
}

[data-testid="stSidebar"] .stCaption {
    color: rgba(232, 221, 206, 0.72);
    font-size: 0.92rem;
    line-height: 1.65;
}

.sidebar-hero-card {
    background: linear-gradient(180deg, rgba(59, 72, 88, 0.92) 0%, rgba(44, 60, 78, 0.90) 100%);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 28px;
    padding: 1.15rem 1.15rem 1.2rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    margin: 0.15rem 0 1.6rem;
}

.sidebar-hero-title {
    color: #fff3e8;
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 1.2rem;
    line-height: 1.1;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.7rem;
}

.sidebar-hero-copy {
    color: rgba(237, 226, 212, 0.9);
    font-size: 0.98rem;
    line-height: 1.55;
}

.sidebar-section {
    margin: 0 0 0.75rem;
}

.sidebar-section-title {
    color: #f6eadc;
    font-size: 0.98rem;
    line-height: 1.2;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.55rem;
}

.sidebar-section-copy {
    color: rgba(232, 221, 206, 0.76);
    font-size: 0.93rem;
    line-height: 1.6;
}

.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
    margin: 1.35rem 0 1.35rem;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(130deg, rgba(255, 251, 247, 0.96) 0%, rgba(244, 236, 225, 0.94) 55%, rgba(230, 217, 198, 0.88) 100%);
    border: 1px solid rgba(34, 45, 60, 0.08);
    border-radius: 32px;
    padding: 2.2rem 2.2rem 2rem;
    box-shadow: var(--shadow-lg);
    margin-bottom: 1.3rem;
}

.hero::after {
    content: "";
    position: absolute;
    inset: auto -10% -45% 42%;
    height: 260px;
    background: radial-gradient(circle, rgba(176, 122, 63, 0.20) 0%, rgba(176, 122, 63, 0) 72%);
    pointer-events: none;
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.65fr) minmax(280px, 0.9fr);
    gap: 1.2rem;
    align-items: stretch;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.38rem 0.72rem;
    margin-bottom: 1rem;
    border-radius: 999px;
    background: rgba(17, 29, 43, 0.05);
    border: 1px solid rgba(17, 29, 43, 0.08);
    color: #5b6674;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0;
    color: var(--ink-strong);
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 3rem;
    line-height: 0.98;
    font-weight: 700;
    letter-spacing: -0.05em;
}

.hero p {
    margin: 0.95rem 0 0 0;
    max-width: 46rem;
    color: var(--ink-soft);
    font-size: 1.04rem;
    line-height: 1.7;
}

.hero-panel {
    position: relative;
    z-index: 1;
    align-self: end;
    background: linear-gradient(180deg, rgba(19, 32, 45, 0.96) 0%, rgba(26, 44, 62, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 28px;
    padding: 1.18rem 1.18rem 1.08rem;
    box-shadow: 0 24px 50px rgba(17, 27, 39, 0.24);
}

.hero-panel-label {
    color: rgba(242, 228, 208, 0.78);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.hero-panel-value {
    color: #fff7ef;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

.hero-panel-copy {
    color: rgba(240, 230, 217, 0.82);
    font-size: 0.95rem;
    line-height: 1.55;
    margin-top: 0.6rem;
}

.glass-card {
    background: var(--surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.22rem 1.22rem;
    box-shadow: var(--shadow-md);
}

.section-shell {
    background: linear-gradient(180deg, rgba(255, 252, 248, 0.68) 0%, rgba(255, 252, 248, 0.52) 100%);
    border: 1px solid var(--line-soft);
    border-radius: 22px;
    padding: 1rem 1.05rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.42);
}

.section-kicker {
    color: #7e6a50;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.section-copy {
    color: var(--ink-soft);
    font-size: 0.94rem;
    line-height: 1.6;
    margin: 0;
}

.panel-intro {
    margin: 0.2rem 0 1rem;
}

.panel-kicker {
    color: #876b45;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 0.28rem;
}

.panel-title {
    color: var(--ink-strong);
    font-size: 1.45rem;
    line-height: 1.15;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.32rem;
}

.panel-copy {
    color: var(--ink-soft);
    font-size: 0.93rem;
    line-height: 1.55;
    max-width: 58rem;
}

.portfolio-spacer {
    height: 1.2rem;
}

.chart-card {
    background: rgba(255, 252, 248, 0.80);
    border: 1px solid rgba(34, 45, 60, 0.07);
    border-radius: 24px;
    padding: 1rem 1rem 0.8rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 0.45rem;
}

.chart-card h4 {
    color: var(--ink-strong);
    margin: 0 0 0.2rem 0;
    font-size: 1.05rem;
    line-height: 1.2;
}

.chart-card .stCaption {
    margin-bottom: 0.45rem;
}

.metric-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: linear-gradient(180deg, rgba(255, 252, 248, 0.92) 0%, rgba(248, 242, 234, 0.88) 100%);
    border: 1px solid rgba(34, 45, 60, 0.09);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    min-height: 142px;
    height: 100%;
    box-shadow: 0 12px 26px rgba(20, 34, 50, 0.06);
}

.metric-label {
    color: #6f7986;
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
    min-height: 2.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metric-value {
    font-size: 1.68rem;
    font-weight: 700;
    color: var(--ink-strong);
    letter-spacing: -0.04em;
}

.metric-sub {
    color: #7a8593;
    font-size: 0.84rem;
    margin-top: auto;
}

.section-title {
    color: var(--ink-strong);
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
    letter-spacing: -0.02em;
}

.section-banner {
    display: inline-flex;
    align-items: center;
    padding: 0.52rem 0.82rem;
    margin: 0.15rem 0 1rem;
    border-radius: 999px;
    background: rgba(176, 122, 63, 0.10);
    border: 1px solid rgba(176, 122, 63, 0.16);
    color: #6c5639;
    font-size: 0.9rem;
}

.stat-card {
    height: 100%;
    min-height: 148px;
    padding: 0.92rem 1rem 0.95rem;
    border-radius: 20px;
    background: rgba(255, 252, 248, 0.74);
    border: 1px solid rgba(34, 45, 60, 0.08);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 10px 22px rgba(20, 34, 50, 0.05);
}

.stat-card-label {
    color: #617083;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stat-card-value {
    margin-top: 0.48rem;
    color: var(--ink-strong);
    font-size: clamp(1.45rem, 1.85vw, 2.25rem);
    line-height: 1.08;
    font-weight: 700;
    letter-spacing: -0.04em;
    word-break: break-word;
}

.stat-card-description {
    margin-top: 0.7rem;
    color: #6f7a88;
    font-size: 0.88rem;
    line-height: 1.45;
    padding-top: 0.35rem;
}

.driver-item {
    border-radius: 16px;
    padding: 0.82rem 0.9rem;
    border: 1px solid rgba(34, 45, 60, 0.08);
    margin-bottom: 0.55rem;
    font-size: 0.92rem;
    line-height: 1.5;
    background: rgba(255, 252, 248, 0.82);
}

.driver-positive {
    border-color: rgba(60, 132, 85, 0.18);
    background: rgba(230, 245, 234, 0.84);
}

.driver-negative {
    border-color: rgba(183, 73, 58, 0.18);
    background: rgba(252, 232, 229, 0.86);
}

.decision-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 0.34rem 0.72rem;
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
}

.badge-priority {
    background: rgba(55, 126, 78, 0.16);
    color: #2d6c42;
    border-color: rgba(55, 126, 78, 0.18);
}

.badge-target {
    background: rgba(32, 94, 146, 0.14);
    color: #275f90;
    border-color: rgba(32, 94, 146, 0.16);
}

.badge-nurture {
    background: rgba(196, 154, 73, 0.18);
    color: #89672c;
    border-color: rgba(196, 154, 73, 0.18);
}

.badge-hold {
    background: rgba(198, 130, 52, 0.14);
    color: #946125;
    border-color: rgba(198, 130, 52, 0.18);
}

.badge-suppress {
    background: rgba(188, 78, 67, 0.14);
    color: #9b3f35;
    border-color: rgba(188, 78, 67, 0.18);
}

.workflow-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
}

.workflow-step {
    background: rgba(255, 252, 248, 0.72);
    border: 1px solid rgba(34, 45, 60, 0.08);
    border-radius: 20px;
    padding: 1rem 1rem;
    box-shadow: var(--shadow-sm);
}

.workflow-step-number {
    color: #9e7a47;
    font-weight: 700;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    text-transform: uppercase;
}

.workflow-step-title {
    margin-top: 0.38rem;
    color: var(--ink-strong);
    font-weight: 700;
    font-size: 1rem;
}

.workflow-step p {
    color: var(--ink-soft);
    font-size: 0.9rem;
    line-height: 1.55;
    margin-bottom: 0;
}

[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(34, 45, 60, 0.08);
    box-shadow: 0 10px 24px rgba(20, 34, 50, 0.04);
    background: rgba(255, 252, 248, 0.84);
}

[data-testid="stDataFrame"] [role="row"]:nth-child(even) {
    background: rgba(247, 241, 233, 0.5);
}

[data-testid="stDataFrame"] [role="row"]:hover {
    background: rgba(236, 226, 212, 0.52);
}

[data-testid="stDataFrame"] [role="columnheader"] {
    background: rgba(240, 232, 221, 0.88);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    background: rgba(255, 251, 246, 0.68);
    border: 1px solid var(--line-soft);
    padding: 0.35rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab"] {
    min-height: 3rem;
    background: rgba(255, 252, 248, 0.55);
    border-radius: 999px;
    border: 1px solid transparent;
    padding: 0.55rem 1rem;
    color: #5e6877;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(186, 137, 76, 0.18) 0%, rgba(255, 250, 245, 0.95) 100%) !important;
    color: #704a1f !important;
    border-color: rgba(176, 122, 63, 0.18) !important;
    box-shadow: 0 6px 16px rgba(176, 122, 63, 0.10);
}

.stButton > button,
.stForm button[kind="primary"],
.stFormSubmitButton > button,
.stDownloadButton > button {
    min-height: 3rem;
    border-radius: 18px;
    border: 1px solid rgba(176, 122, 63, 0.20);
    background: linear-gradient(180deg, rgba(187, 137, 76, 0.94) 0%, rgba(141, 95, 45, 0.96) 100%);
    color: #fff8f0;
    font-weight: 700;
    box-shadow: 0 12px 22px rgba(141, 95, 45, 0.16);
}

.stForm button[kind="primary"] p,
.stFormSubmitButton > button p,
.stButton > button p {
    color: #fff8f0 !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stForm button[kind="primary"]:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    filter: brightness(1.03);
    border-color: rgba(141, 95, 45, 0.28);
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea,
.stDateInput input {
    min-height: 3rem;
    border-radius: 16px;
    background: rgba(255, 251, 246, 0.86);
    border: 1px solid rgba(34, 45, 60, 0.10);
}

.stSelectbox > div[data-baseweb="select"] > div,
.stMultiSelect > div[data-baseweb="select"] > div {
    min-height: 3rem;
    border-radius: 16px;
    background: rgba(255, 251, 246, 0.86);
    border: 1px solid rgba(34, 45, 60, 0.10);
}

.stSlider [data-baseweb="slider"] {
    padding-top: 0.35rem;
}

.stRadio > div {
    gap: 0.45rem;
}

.stRadio [role="radiogroup"] {
    background: rgba(255, 251, 246, 0.72);
    border: 1px solid var(--line-soft);
    border-radius: 18px;
    padding: 0.25rem;
}

.stRadio [role="radiogroup"] label {
    border-radius: 14px;
    padding: 0.42rem 0.8rem;
}

.stToggle label[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stMultiSelect label p,
.stTextInput label p,
.stNumberInput label p,
.stSlider label p,
.stRadio label p {
    font-weight: 700;
    color: var(--ink-strong);
}

.stCaption {
    color: #6a7482;
}

[data-testid="stMetric"] {
    background: rgba(255, 252, 248, 0.78);
    border: 1px solid var(--line-soft);
    border-radius: 18px;
    padding: 0.9rem 1rem;
}

.stAlert {
    border-radius: 18px;
}

@media (max-width: 1100px) {
    .hero-grid,
    .workflow-grid {
        grid-template-columns: 1fr;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
"""
