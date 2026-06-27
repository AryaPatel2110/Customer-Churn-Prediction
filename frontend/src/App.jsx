import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  Database,
  Gauge,
  HeartHandshake,
  LayoutDashboard,
  Menu,
  Radar,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingDown,
  Upload,
  UserRound,
  Users,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiFetch } from "./staticApi";

const RISK_COLORS = { High: "#ef6a47", Medium: "#f0b44c", Low: "#1e9b71" };
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const number = new Intl.NumberFormat("en-US");
const percent = (value) => `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;

function Logo({ inverse = false }) {
  return (
    <div className={`logo ${inverse ? "logo-inverse" : ""}`}>
      <span className="logo-mark"><Radar size={19} strokeWidth={2.5} /></span>
      <span>Churn<span>Scope</span></span>
    </div>
  );
}

function Landing({ onOpenDashboard }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const workflow = [
    { icon: Database, title: "Customer signals", text: "Account, billing, support and engagement activity" },
    { icon: BrainCircuit, title: "Predictive model", text: "Feature engineering and LightGBM risk scoring" },
    { icon: Target, title: "Risk identified", text: "Every customer sorted into an operational risk band" },
    { icon: HeartHandshake, title: "Retention action", text: "Clear next steps for proactive re-engagement" },
  ];

  return (
    <div className="landing">
      <header className="landing-nav">
        <Logo inverse />
        <nav className={menuOpen ? "nav-links open" : "nav-links"}>
          <a href="#how-it-works">How it works</a>
          <a href="#model">The model</a>
          <a href="#outcomes">Outcomes</a>
        </nav>
        <button className="button button-light nav-cta" onClick={onOpenDashboard}>
          Open dashboard <ArrowRight size={16} />
        </button>
        <button className="icon-button menu-button" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation">
          {menuOpen ? <X /> : <Menu />}
        </button>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span /> Customer retention intelligence</div>
            <h1>Know who might leave.<br /><em>Act before they do.</em></h1>
            <p>
              ChurnScope turns customer behavior into clear, actionable risk signals—so your team can focus
              retention effort where it matters most.
            </p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={onOpenDashboard}>
                Explore customer risk <ArrowRight size={17} />
              </button>
              <a className="text-link" href="#how-it-works">See how it works <ChevronRight size={16} /></a>
            </div>
            <div className="hero-proof">
              <div className="avatar-stack"><span>ML</span><span>BI</span><span>CS</span></div>
              <p><strong>104,480</strong> existing customers scored from your test portfolio</p>
            </div>
          </div>

          <div className="hero-visual" aria-label="Customer risk preview">
            <div className="visual-grid" />
            <div className="floating-label label-model"><BrainCircuit size={15} /> LightGBM + SMOTE</div>
            <div className="risk-preview">
              <div className="preview-head">
                <div>
                  <span>Portfolio signal</span>
                  <strong>Customer risk monitor</strong>
                </div>
                <span className="live-pill"><i /> Live model</span>
              </div>
              <div className="preview-score">
                <div className="score-ring">
                  <svg viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="48" />
                    <circle className="score-progress" cx="60" cy="60" r="48" pathLength="100" />
                  </svg>
                  <div><strong>82</strong><span>risk score</span></div>
                </div>
                <div className="signal-list">
                  <span>Primary signals</span>
                  <div><i className="signal-dot coral" /> Low user rating <b>+24%</b></div>
                  <div><i className="signal-dot amber" /> Support activity <b>+19%</b></div>
                  <div><i className="signal-dot green" /> Low engagement <b>+16%</b></div>
                </div>
              </div>
              <div className="preview-customer">
                <span className="customer-avatar">RT</span>
                <div><strong>Customer RT8P2K</strong><span>Premium · 17 months</span></div>
                <span className="risk-badge high">High risk</span>
              </div>
              <div className="action-note">
                <WandSparkles size={18} />
                <div><span>Recommended next action</span><strong>Route to a retention specialist within 24 hours.</strong></div>
                <ArrowRight size={17} />
              </div>
            </div>
            <div className="floating-label label-accuracy"><ShieldCheck size={16} /><span><b>95.79%</b> ROC-AUC</span></div>
          </div>
        </section>

        <section className="trust-strip">
          <span>Built for proactive customer teams</span>
          <div><Activity size={18} /> Behavioral signals</div>
          <div><Gauge size={18} /> Risk scoring</div>
          <div><Users size={18} /> Customer prioritization</div>
          <div><Zap size={18} /> Retention playbooks</div>
        </section>

        <section className="section workflow-section" id="how-it-works">
          <div className="section-intro">
            <div className="eyebrow dark"><span /> From signals to intervention</div>
            <h2>A clear path from customer behavior to retention.</h2>
            <p>Your workflow, translated into one connected system your team can understand and act on.</p>
          </div>
          <div className="workflow-grid">
            {workflow.map((item, index) => {
              const Icon = item.icon;
              return (
                <div className="workflow-card" key={item.title}>
                  <div className="workflow-number">0{index + 1}</div>
                  <span className="workflow-icon"><Icon size={23} /></span>
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                  {index < workflow.length - 1 && <ArrowRight className="workflow-arrow" size={19} />}
                </div>
              );
            })}
          </div>
        </section>

        <section className="section model-section" id="model">
          <div className="model-panel">
            <div className="model-copy">
              <div className="eyebrow light"><span /> Model transparency</div>
              <h2>Built on behavior.<br />Designed for action.</h2>
              <p>
                The model combines customer history, engagement, spending and service friction into a single
                probability your retention team can use.
              </p>
              <ul>
                <li><Check size={17} /> Trained on 243,787 labeled customer records</li>
                <li><Check size={17} /> SMOTE balances churn patterns during training</li>
                <li><Check size={17} /> 15 selected features drive the final prediction</li>
              </ul>
              <button className="button button-outline-light" onClick={onOpenDashboard}>
                Inspect model performance <ArrowRight size={16} />
              </button>
            </div>
            <div className="feature-map">
              <div className="feature-center"><BrainCircuit size={28} /><strong>Churn<br />probability</strong></div>
              {["Engagement", "Support", "Spending", "Tenure", "Content", "Rating"].map((label, index) => (
                <div className={`feature-node node-${index + 1}`} key={label}>{label}</div>
              ))}
              <svg viewBox="0 0 500 360" aria-hidden="true">
                {[["250","180","95","65"],["250","180","402","58"],["250","180","430","190"],["250","180","365","310"],["250","180","120","305"],["250","180","64","180"]].map((line, i) => (
                  <line key={i} x1={line[0]} y1={line[1]} x2={line[2]} y2={line[3]} />
                ))}
              </svg>
            </div>
          </div>
        </section>

        <section className="section outcomes-section" id="outcomes">
          <div className="outcome-copy">
            <div className="eyebrow dark"><span /> Retention, focused</div>
            <h2>Stop treating every customer the same.</h2>
            <p>
              See the people behind the probability, understand why they are at risk, and give your team a
              practical next action.
            </p>
          </div>
          <div className="outcome-stats">
            <div><strong>0.9579</strong><span>ROC-AUC</span></div>
            <div><strong>90%</strong><span>Model accuracy</span></div>
            <div><strong>3</strong><span>Actionable risk bands</span></div>
          </div>
        </section>

        <section className="final-cta">
          <div>
            <span>Ready to see the risk in your portfolio?</span>
            <h2>Your next retained customer starts here.</h2>
          </div>
          <button className="button button-light" onClick={onOpenDashboard}>
            Open the dashboard <ArrowRight size={17} />
          </button>
        </section>
      </main>

      <footer>
        <Logo />
        <p>Customer churn intelligence for proactive retention.</p>
        <span>Model: LightGBM + SMOTE</span>
      </footer>
    </div>
  );
}

function RiskBadge({ band }) {
  return <span className={`risk-badge ${band.toLowerCase()}`}><i />{band}</span>;
}

function MetricCard({ icon: Icon, label, value, note, tone = "green" }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${tone}`}><Icon size={19} /></div>
      <div className="metric-label">{label}</div>
      <strong>{value}</strong>
      <span>{note}</span>
    </div>
  );
}

function DashboardOverview({ summary, setPage, setSelectedCustomer }) {
  const subscription = summary.subscription_risk.map((item) => ({
    name: item.SubscriptionType,
    risk: +(item.average_risk * 100).toFixed(1),
  }));
  return (
    <>
      <div className="dashboard-heading">
        <div><span>Portfolio overview</span><h1>Good morning, Analyst.</h1><p>Here is what the churn model sees across your existing customer base.</p></div>
        <button className="button button-dark" onClick={() => setPage("predict")}><Sparkles size={16} /> New prediction</button>
      </div>

      <div className="attention-banner">
        <div className="attention-icon"><TrendingDown size={23} /></div>
        <div><strong>{number.format(summary.high_risk)} customers need immediate attention</strong><span>These customers have a churn probability of 50% or higher.</span></div>
        <button onClick={() => setPage("customers")}>Review customers <ArrowRight size={16} /></button>
      </div>

      <div className="metrics-grid">
        <MetricCard icon={Users} label="Existing customers" value={number.format(summary.total_customers)} note="Loaded from test.csv" />
        <MetricCard icon={Target} label="Predicted to churn" value={number.format(summary.predicted_churn)} note={`${percent(summary.predicted_churn_rate)} of portfolio`} tone="coral" />
        <MetricCard icon={Gauge} label="Average risk" value={percent(summary.average_probability)} note={`${number.format(summary.medium_risk)} need monitoring`} tone="amber" />
        <MetricCard icon={CircleDollarSign} label="Revenue exposure" value={money.format(summary.revenue_at_risk)} note="Probability-weighted monthly" tone="navy" />
      </div>

      <div className="dashboard-grid">
        <section className="panel risk-panel">
          <div className="panel-heading"><div><h2>Risk distribution</h2><p>Customers by operational band</p></div><span className="panel-chip">Live portfolio</span></div>
          <div className="risk-chart-layout">
            <div className="donut-wrap">
              <ResponsiveContainer width="100%" height={230}>
                <PieChart>
                  <Pie data={summary.risk_distribution} dataKey="value" innerRadius={65} outerRadius={88} paddingAngle={3} stroke="none">
                    {summary.risk_distribution.map((entry) => <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />)}
                  </Pie>
                  <Tooltip formatter={(value) => number.format(value)} />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-label"><strong>{number.format(summary.total_customers)}</strong><span>Total</span></div>
            </div>
            <div className="risk-legend">
              {summary.risk_distribution.map((item) => (
                <div key={item.name}><i style={{ background: RISK_COLORS[item.name] }} /><span>{item.name} risk</span><strong>{number.format(item.value)}</strong><em>{percent(item.value / summary.total_customers)}</em></div>
              ))}
            </div>
          </div>
        </section>

        <section className="panel subscription-panel">
          <div className="panel-heading"><div><h2>Risk by subscription</h2><p>Average predicted probability</p></div></div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={subscription} layout="vertical" margin={{ top: 14, right: 25, left: 0, bottom: 0 }}>
              <CartesianGrid horizontal={false} stroke="#e8ece8" />
              <XAxis type="number" unit="%" axisLine={false} tickLine={false} tick={{ fill: "#7b847e", fontSize: 11 }} />
              <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} width={76} tick={{ fill: "#354039", fontSize: 12 }} />
              <Tooltip formatter={(value) => `${value}%`} cursor={{ fill: "#f4f6f2" }} />
              <Bar dataKey="risk" fill="#174f45" radius={[0, 5, 5, 0]} barSize={22} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <section className="panel customer-panel">
        <div className="panel-heading">
          <div><h2>Highest-risk customers</h2><p>Start here for proactive retention outreach</p></div>
          <button className="text-button" onClick={() => setPage("customers")}>View all customers <ArrowRight size={15} /></button>
        </div>
        <CustomerTable rows={summary.top_risk_customers} onSelect={setSelectedCustomer} compact />
      </section>
    </>
  );
}

function CustomerTable({ rows, onSelect, compact = false }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Customer</th><th>Subscription</th><th>Monthly charge</th><th>Engagement</th><th>Risk score</th><th>Band</th><th /></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.customer_id} onClick={() => onSelect(row)}>
              <td><div className="customer-cell"><span>{row.customer_id.slice(0, 2)}</span><div><strong>{row.customer_id}</strong><small>{row.account_age} months</small></div></div></td>
              <td>{row.subscription_type}</td>
              <td>{money.format(row.monthly_charges)}</td>
              <td><strong>{row.viewing_hours.toFixed(1)}h</strong><small className="table-sub"> / week</small></td>
              <td><div className="risk-score-cell"><div><i style={{ width: `${Math.max(row.probability * 100, 2)}%`, background: RISK_COLORS[row.risk_band] }} /></div><strong>{percent(row.probability)}</strong></div></td>
              <td><RiskBadge band={row.risk_band} /></td>
              <td><ChevronRight size={16} /></td>
            </tr>
          ))}
          {!rows.length && <tr><td colSpan="7" className="empty-state">No customers match these filters.</td></tr>}
        </tbody>
      </table>
      {!compact && rows.length > 0 && <div className="table-hint">Select any row to inspect customer signals and recommended actions.</div>}
    </div>
  );
}

function CustomersPage({ setSelectedCustomer }) {
  const [filters, setFilters] = useState({ search: "", risk: "All", subscription: "All", sort: "risk_desc", page: 1 });
  const [result, setResult] = useState({ items: [], total: 0, pages: 1 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams({ ...filters, page_size: 20 });
    const timer = setTimeout(() => {
      setLoading(true);
      apiFetch(`/portfolio/customers?${params}`)
        .then(setResult)
        .finally(() => setLoading(false));
    }, filters.search ? 250 : 0);
    return () => clearTimeout(timer);
  }, [filters]);

  const update = (key, value) => setFilters((current) => ({ ...current, [key]: value, page: key === "page" ? value : 1 }));

  return (
    <>
      <div className="dashboard-heading">
        <div><span>Customer intelligence</span><h1>Existing customers</h1><p>Search, filter and prioritize all customers scored from <code>data/test.csv</code>.</p></div>
        <div className="count-badge"><Users size={16} /> {number.format(result.total)} customers</div>
      </div>
      <section className="panel customer-browser">
        <div className="filter-bar">
          <label className="search-field"><Search size={17} /><input value={filters.search} onChange={(e) => update("search", e.target.value)} placeholder="Search customer ID…" /></label>
          <select value={filters.risk} onChange={(e) => update("risk", e.target.value)}><option>All</option><option>High</option><option>Medium</option><option>Low</option></select>
          <select value={filters.subscription} onChange={(e) => update("subscription", e.target.value)}><option>All</option><option>Basic</option><option>Standard</option><option>Premium</option></select>
          <select value={filters.sort} onChange={(e) => update("sort", e.target.value)}>
            <option value="risk_desc">Highest risk first</option><option value="risk_asc">Lowest risk first</option>
            <option value="charges_desc">Highest charge</option><option value="tenure_desc">Longest tenure</option>
          </select>
        </div>
        {loading ? <LoadingRows /> : <CustomerTable rows={result.items} onSelect={setSelectedCustomer} />}
        <div className="pagination">
          <span>Page {filters.page} of {result.pages}</span>
          <div>
            <button disabled={filters.page === 1} onClick={() => update("page", filters.page - 1)} aria-label="Previous page"><ChevronLeft size={17} /></button>
            <button disabled={filters.page >= result.pages} onClick={() => update("page", filters.page + 1)} aria-label="Next page"><ChevronRight size={17} /></button>
          </div>
        </div>
      </section>
    </>
  );
}

function LoadingRows() {
  return <div className="loading-rows">{[1,2,3,4,5,6].map((item) => <div key={item}><i /><span /><span /><span /></div>)}</div>;
}

const initialForm = {
  AccountAge: 18, MonthlyCharges: 34, TotalCharges: 612, SubscriptionType: "Basic",
  ViewingHoursPerWeek: 3, AverageViewingDuration: 28, ContentDownloadsPerMonth: 1,
  UserRating: 2.4, SupportTicketsPerMonth: 5, WatchlistSize: 2,
};

function PredictPage() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true); setError("");
    try {
      setResult(await apiFetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(form) }));
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const fields = [
    ["AccountAge", "Account age", "months", 1], ["MonthlyCharges", "Monthly charges", "$ / month", 0],
    ["TotalCharges", "Total charges", "$ lifetime", 0], ["ViewingHoursPerWeek", "Viewing hours", "hours / week", 0],
    ["AverageViewingDuration", "Average session", "minutes", 0], ["ContentDownloadsPerMonth", "Downloads", "per month", 0],
    ["UserRating", "User rating", "1–5", 1], ["SupportTicketsPerMonth", "Support tickets", "per month", 0],
    ["WatchlistSize", "Watchlist size", "titles", 0],
  ];

  return (
    <>
      <div className="dashboard-heading">
        <div><span>Prediction lab</span><h1>Score a customer</h1><p>Enter customer signals to generate an immediate churn prediction and retention plan.</p></div>
        <span className="model-status"><i /> Model ready</span>
      </div>
      <div className="predict-layout">
        <form className="panel prediction-form" onSubmit={submit}>
          <div className="panel-heading"><div><h2>Customer signals</h2><p>All fields match the deployed preprocessing contract.</p></div><Upload size={19} /></div>
          <div className="form-grid">
            <label className="wide"><span>Subscription type</span><select value={form.SubscriptionType} onChange={(e) => set("SubscriptionType", e.target.value)}><option>Basic</option><option>Standard</option><option>Premium</option></select></label>
            {fields.map(([key, label, suffix, min]) => (
              <label key={key}><span>{label}</span><div className="input-suffix"><input type="number" min={min} max={key === "UserRating" ? 5 : undefined} step={["AccountAge","ContentDownloadsPerMonth","SupportTicketsPerMonth","WatchlistSize"].includes(key) ? 1 : 0.1} value={form[key]} onChange={(e) => set(key, +e.target.value)} required /><em>{suffix}</em></div></label>
            ))}
          </div>
          {error && <div className="form-error">{error}</div>}
          <button className="button button-dark predict-button" disabled={loading}>{loading ? "Scoring customer…" : <><Sparkles size={17} /> Generate prediction</>}</button>
        </form>
        <section className={`panel prediction-result ${result ? "has-result" : ""}`}>
          {!result ? (
            <div className="result-placeholder"><span><BrainCircuit size={30} /></span><h2>Prediction output</h2><p>Complete the customer profile to reveal their churn probability, risk factors and recommended intervention.</p></div>
          ) : (
            <>
              <div className="result-top"><span>Model output</span><RiskBadge band={result.risk_band} /></div>
              <div className="probability-display"><span>Churn probability</span><strong>{percent(result.probability)}</strong><div><i style={{ width: `${Math.max(result.probability * 100, 2)}%` }} /></div></div>
              <div className="result-block"><h3><Radar size={17} /> Risk explanation</h3>{result.reasons.map((reason) => <p key={reason}>{reason}</p>)}</div>
              <div className="result-block action"><h3><HeartHandshake size={17} /> Retention playbook</h3>{result.retention_plan.map((item, i) => <p key={item}><span>{i + 1}</span>{item}</p>)}</div>
              {result.warnings.length > 0 && <div className="warning-note">{result.warnings.join(" ")}</div>}
            </>
          )}
        </section>
      </div>
    </>
  );
}

function ModelPage({ model }) {
  const maxImportance = Math.max(...model.drivers.map((item) => item.importance), 1);
  return (
    <>
      <div className="dashboard-heading"><div><span>Model transparency</span><h1>How prediction works</h1><p>From raw customer behavior to an operational retention decision.</p></div><span className="model-status"><i /> Saved artifact active</span></div>
      <div className="model-metrics">
        <MetricCard icon={BrainCircuit} label="Model" value={model.metrics.Model} note={model.metrics.Runtime} />
        <MetricCard icon={ShieldCheck} label="ROC-AUC" value={model.metrics["ROC-AUC"]} note="Final notebook evaluation" tone="navy" />
        <MetricCard icon={Target} label="Accuracy" value={model.metrics.Accuracy} note="Held-out evaluation" tone="amber" />
        <MetricCard icon={Database} label="Training records" value={number.format(model.training_rows)} note={`${percent(model.training_churn_rate)} observed churn`} tone="coral" />
      </div>
      <section className="panel pipeline-panel">
        <div className="panel-heading"><div><h2>Prediction pipeline</h2><p>Deterministic preprocessing followed by the trained classifier.</p></div><span className="panel-chip">{model.pipeline_source}</span></div>
        <div className="pipeline-flow">
          {model.workflow.map((step, index) => <div key={step}><span>0{index + 1}</span><strong>{step}</strong>{index < model.workflow.length - 1 && <ArrowRight size={17} />}</div>)}
        </div>
      </section>
      <div className="dashboard-grid model-detail-grid">
        <section className="panel">
          <div className="panel-heading"><div><h2>Global feature importance</h2><p>Top LightGBM drivers</p></div></div>
          <div className="importance-list">
            {model.drivers.map((item, index) => <div key={item.feature}><span>{String(index + 1).padStart(2, "0")}</span><p>{item.feature}</p><div><i style={{ width: `${item.importance / maxImportance * 100}%` }} /></div><strong>{item.importance}</strong></div>)}
          </div>
        </section>
        <section className="panel feature-detail">
          <div className="panel-heading"><div><h2>Feature contract</h2><p>Inputs and engineered model fields</p></div></div>
          <h3>10 raw customer signals</h3>
          <div className="feature-tags">{model.raw_features.map((item) => <span key={item}>{item}</span>)}</div>
          <h3>15 selected model features</h3>
          <div className="feature-tags selected">{model.model_features.map((item) => <span key={item}>{item}</span>)}</div>
        </section>
      </div>
    </>
  );
}

function CustomerDrawer({ customer, onClose }) {
  if (!customer) return null;
  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside className="customer-drawer" onMouseDown={(e) => e.stopPropagation()}>
        <button className="drawer-close" onClick={onClose} aria-label="Close customer detail"><X size={19} /></button>
        <div className="drawer-profile"><span>{customer.customer_id.slice(0, 2)}</span><div><small>Customer profile</small><h2>{customer.customer_id}</h2><p>{customer.subscription_type} · {customer.account_age} months</p></div></div>
        <div className="drawer-score"><div><span>Churn probability</span><strong>{percent(customer.probability)}</strong></div><RiskBadge band={customer.risk_band} /></div>
        <div className="drawer-stats"><div><span>Monthly charge</span><strong>{money.format(customer.monthly_charges)}</strong></div><div><span>Weekly viewing</span><strong>{customer.viewing_hours.toFixed(1)}h</strong></div><div><span>User rating</span><strong>{customer.user_rating.toFixed(1)} / 5</strong></div><div><span>Support tickets</span><strong>{customer.support_tickets}</strong></div></div>
        <div className="drawer-section"><h3>Why this risk level?</h3>{customer.reasons.map((reason) => <p key={reason}><Radar size={15} />{reason}</p>)}</div>
        <div className="drawer-section plan"><h3>Recommended playbook</h3>{customer.retention_plan.map((item, index) => <p key={item}><span>{index + 1}</span>{item}</p>)}</div>
      </aside>
    </div>
  );
}

function Dashboard({ onBack }) {
  const [page, setPage] = useState("overview");
  const [summary, setSummary] = useState(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState(null);

  useEffect(() => {
    Promise.all([apiFetch("/portfolio/summary"), apiFetch("/model")])
      .then(([summaryData, modelData]) => { setSummary(summaryData); setModel(modelData); })
      .catch((err) => setError(err.message));
  }, []);

  const nav = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "customers", label: "Customers", icon: Users },
    { id: "predict", label: "Predict", icon: Sparkles },
    { id: "model", label: "Model", icon: BrainCircuit },
  ];

  if (error) return <div className="app-error"><Radar size={36} /><h1>Dashboard unavailable</h1><p>{error}</p><button className="button button-dark" onClick={onBack}>Return home</button></div>;
  if (!summary || !model) return <div className="app-loading"><span><Radar size={30} /></span><strong>Scoring existing customers</strong><p>Running 104,480 records through the deployed model…</p></div>;

  return (
    <div className="dashboard-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-logo"><Logo inverse /><button onClick={() => setSidebarOpen(false)}><X /></button></div>
        <nav>
          <span className="nav-label">Workspace</span>
          {nav.map((item) => { const Icon = item.icon; return <button key={item.id} className={page === item.id ? "active" : ""} onClick={() => { setPage(item.id); setSidebarOpen(false); }}><Icon size={18} />{item.label}{page === item.id && <i />}</button>; })}
        </nav>
        <div className="sidebar-model"><span><Activity size={16} /></span><div><strong>Model online</strong><small>LightGBM · v1.0</small></div><i /></div>
        <button className="back-home" onClick={onBack}><ArrowLeft size={16} /> Back to website</button>
      </aside>
      <div className="dashboard-main">
        <header className="dashboard-topbar">
          <button className="mobile-menu" onClick={() => setSidebarOpen(true)}><Menu /></button>
          <div className="dataset-source"><Database size={16} /><span>Source</span><strong>data/test.csv</strong></div>
          <div className="topbar-right"><span className="updated"><i /> Portfolio scored</span><div className="analyst-avatar">AN</div></div>
        </header>
        <main className="dashboard-content">
          {page === "overview" && <DashboardOverview summary={summary} setPage={setPage} setSelectedCustomer={setSelectedCustomer} />}
          {page === "customers" && <CustomersPage setSelectedCustomer={setSelectedCustomer} />}
          {page === "predict" && <PredictPage />}
          {page === "model" && <ModelPage model={model} />}
        </main>
      </div>
      <CustomerDrawer customer={selectedCustomer} onClose={() => setSelectedCustomer(null)} />
    </div>
  );
}

export default function App() {
  const [view, setView] = useState(() => window.location.hash === "#dashboard" ? "dashboard" : "landing");
  const openDashboard = () => { window.location.hash = "dashboard"; setView("dashboard"); window.scrollTo(0, 0); };
  const openLanding = () => { history.pushState("", document.title, window.location.pathname); setView("landing"); window.scrollTo(0, 0); };
  return view === "dashboard" ? <Dashboard onBack={openLanding} /> : <Landing onOpenDashboard={openDashboard} />;
}
