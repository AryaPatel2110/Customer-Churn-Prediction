const SERVER_MODE = import.meta.env?.VITE_API_MODE === "server";
const API = "/api";

let customersPromise;
let predictorPromise;

const RETENTION_PLANS = {
  High: [
    "Route to retention specialist within 24 hours.",
    "Offer plan optimization, discount, or pause option.",
    "Prioritize support follow-up for unresolved tickets.",
  ],
  Medium: [
    "Trigger targeted content recommendations.",
    "Send engagement campaign within the next billing cycle.",
    "Monitor support and viewing behavior weekly.",
  ],
  Low: [
    "Keep customer in normal monitoring.",
    "Use loyalty or upgrade messaging only when relevant.",
    "Avoid unnecessary discounting.",
  ],
};

const RECOMMENDATIONS = {
  High: "At-risk customer. Prioritize a retention offer and support follow-up.",
  Medium: "Watch closely. Improve engagement with targeted content and check-ins.",
  Low: "Healthy customer. Maintain experience quality and monitor normally.",
};

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "The request could not be completed.");
  }
  return response.json();
}

function riskBand(probability) {
  if (probability >= 0.5) return "High";
  if (probability >= 0.2) return "Medium";
  return "Low";
}

function reasonsFor(customer) {
  const get = (apiKey, staticKey) => customer[apiKey] ?? customer[staticKey];
  const reasons = [];
  if (get("SupportTicketsPerMonth", "support_tickets") >= 4) {
    reasons.push("Support load is elevated, which can signal unresolved service friction.");
  }
  if (get("UserRating", "user_rating") <= 2.5) {
    reasons.push("User rating is low, suggesting dissatisfaction with the experience.");
  }
  if (get("ViewingHoursPerWeek", "viewing_hours") <= 3) {
    reasons.push("Weekly viewing engagement is low.");
  }
  if (get("ContentDownloadsPerMonth", "downloads") <= 1) {
    reasons.push("Content download activity is limited.");
  }
  if (get("MonthlyCharges", "monthly_charges") >= 90) {
    reasons.push("Monthly charges are high, increasing price sensitivity.");
  }
  if (get("WatchlistSize", "watchlist_size") <= 2) {
    reasons.push("Watchlist size is small, suggesting weak content intent.");
  }
  if (!reasons.length) {
    reasons.push("No single behavioral warning dominates; risk is mainly model-driven.");
  }
  return reasons;
}

function decodeCustomer(row) {
  const customer = {
    customer_id: row[0],
    subscription_type: row[1],
    payment_method: row[2],
    content_type: row[3],
    device: row[4],
    account_age: row[5],
    monthly_charges: row[6],
    viewing_hours: row[7],
    user_rating: row[8],
    support_tickets: row[9],
    probability: row[10],
    prediction: row[11],
    risk_band: row[12],
    downloads: row[13],
    watchlist_size: row[14],
  };
  customer.recommended_action = RECOMMENDATIONS[customer.risk_band];
  customer.reasons = reasonsFor(customer);
  customer.retention_plan = RETENTION_PLANS[customer.risk_band];
  return customer;
}

function loadCustomers() {
  customersPromise ||= fetchJson("/data/customers.json");
  return customersPromise;
}

function loadPredictor() {
  predictorPromise ||= fetchJson("/data/predictor.json");
  return predictorPromise;
}

function scoreTree(node, features) {
  if (typeof node === "number") return node;
  const [featureIndex, threshold, defaultLeft, left, right] = node;
  const value = features[featureIndex];
  const chooseLeft = Number.isNaN(value) ? Boolean(defaultLeft) : value <= threshold;
  return scoreTree(chooseLeft ? left : right, features);
}

function preprocess(customer) {
  const accountAge = customer.AccountAge;
  const monthlyCharges = customer.MonthlyCharges;
  const totalCharges = customer.TotalCharges;
  const viewingHours = customer.ViewingHoursPerWeek;
  const viewingDuration = customer.AverageViewingDuration;
  const downloads = customer.ContentDownloadsPerMonth;
  const rating = customer.UserRating;
  const tickets = customer.SupportTicketsPerMonth;
  const watchlist = customer.WatchlistSize;
  const subscriptions = { Basic: 0, Standard: 1, Premium: 2 };

  return [
    viewingHours + viewingDuration + downloads,
    accountAge,
    viewingDuration,
    totalCharges,
    viewingHours === 0 ? 0 : tickets / viewingHours,
    viewingHours,
    viewingHours === 0 ? 0 : accountAge / viewingHours,
    monthlyCharges,
    accountAge === 0 ? 0 : totalCharges / accountAge,
    rating,
    downloads,
    watchlist === 0 ? 0 : viewingHours / watchlist,
    watchlist,
    tickets,
    subscriptions[customer.SubscriptionType] ?? -1,
  ];
}

function validationWarnings(customer) {
  const warnings = [];
  const expectedTotal = customer.AccountAge * customer.MonthlyCharges;
  if (expectedTotal > 0) {
    const ratio = customer.TotalCharges / expectedTotal;
    if (ratio < 0.5 || ratio > 1.75) {
      warnings.push("Total Charges is unusual compared with Account Age and Monthly Charges.");
    }
  }
  if (customer.SupportTicketsPerMonth >= 5 && customer.UserRating >= 4.5) {
    warnings.push("High support tickets with a high rating may indicate inconsistent inputs.");
  }
  if (customer.ViewingHoursPerWeek === 0 && customer.AverageViewingDuration > 0) {
    warnings.push("Average Viewing Duration is above zero while weekly viewing hours are zero.");
  }
  return warnings;
}

export function scoreWithPredictor(predictor, customer) {
  const features = preprocess(customer);
  const rawScore = predictor.trees.reduce(
    (total, tree) => total + scoreTree(tree, features),
    0,
  );
  const probability = 1 / (1 + Math.exp(-rawScore));
  const band = riskBand(probability);
  return {
    prediction: probability >= 0.5 ? 1 : 0,
    probability,
    risk_band: band,
    recommendation: RECOMMENDATIONS[band],
    warnings: validationWarnings(customer),
    reasons: reasonsFor(customer),
    retention_plan: RETENTION_PLANS[band],
  };
}

async function predictLocally(customer) {
  const predictor = await loadPredictor();
  return scoreWithPredictor(predictor, customer);
}

async function queryCustomers(path) {
  const { rows } = await loadCustomers();
  const url = new URL(path, window.location.origin);
  const search = (url.searchParams.get("search") || "").toLowerCase();
  const risk = url.searchParams.get("risk") || "All";
  const subscription = url.searchParams.get("subscription") || "All";
  const sort = url.searchParams.get("sort") || "risk_desc";
  const page = Number(url.searchParams.get("page") || 1);
  const pageSize = Number(url.searchParams.get("page_size") || 20);

  const filtered = rows.filter((row) => (
    (!search || row[0].toLowerCase().includes(search))
    && (risk === "All" || row[12] === risk)
    && (subscription === "All" || row[1] === subscription)
  ));
  const sorters = {
    risk_desc: (a, b) => b[10] - a[10],
    risk_asc: (a, b) => a[10] - b[10],
    charges_desc: (a, b) => b[6] - a[6],
    tenure_desc: (a, b) => b[5] - a[5],
  };
  filtered.sort(sorters[sort]);
  const start = (page - 1) * pageSize;
  return {
    items: filtered.slice(start, start + pageSize).map(decodeCustomer),
    total: filtered.length,
    page,
    page_size: pageSize,
    pages: Math.max(1, Math.ceil(filtered.length / pageSize)),
  };
}

async function staticApiFetch(path, options) {
  if (path === "/portfolio/summary") return fetchJson("/data/summary.json");
  if (path === "/model") return fetchJson("/data/model-info.json");
  if (path.startsWith("/portfolio/customers?")) return queryCustomers(path);
  if (path === "/predict") return predictLocally(JSON.parse(options.body));
  throw new Error(`Static route is not available: ${path}`);
}

export async function apiFetch(path, options) {
  if (SERVER_MODE) return fetchJson(`${API}${path}`, options);
  return staticApiFetch(path, options);
}
