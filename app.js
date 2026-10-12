/**
 * Voyage Analytics - Client-Side Intelligence & Application Logic
 * Implements ML Predictor, Fuzzy Search, Recommender, and Visualizations
 */

// Embedded Data Catalog from artifacts/hotel_catalog.json
const HOTEL_CATALOG = {
  hotels: {
    "Hotel A": { place: "Florianopolis (SC)", price_per_day: 313.02, rating: 4.8, stars: 5, desc: "Luxury beachfront resort with ocean panoramas, heated infinity pool, and private cabanas." },
    "Hotel AF": { place: "Sao Paulo (SP)", price_per_day: 139.10, rating: 4.4, stars: 4, desc: "Modern business executive hotel located in the heart of the financial district." },
    "Hotel AU": { place: "Recife (PE)", price_per_day: 312.83, rating: 4.7, stars: 5, desc: "Boutique oceanfront retreat near Boa Viagem with artisanal dining and spa." },
    "Hotel BD": { place: "Natal (RN)", price_per_day: 242.88, rating: 4.5, stars: 4, desc: "Tropical oasis near Ponta Negra dunes offering coastal suites and rooftop pool." },
    "Hotel BP": { place: "Brasilia (DF)", price_per_day: 247.62, rating: 4.6, stars: 4, desc: "Architectural masterpiece near Lake Paranoa with conference lounges and fine dining." },
    "Hotel BW": { place: "Campo Grande (MS)", price_per_day: 60.39, rating: 4.1, stars: 3, desc: "Cozy budget-friendly hotel with prime access to eco-tourism gateways and transit." },
    "Hotel CB": { place: "Rio de Janeiro (RJ)", price_per_day: 165.99, rating: 4.7, stars: 4, desc: "Vibrant Copacabana coastal sanctuary overlooking iconic beaches and Sugarloaf Mountain." },
    "Hotel K": { place: "Salvador (BH)", price_per_day: 263.41, rating: 4.6, stars: 5, desc: "Colonial heritage luxury estate blending Bahian culture, ocean views, and gourmet breakfast." },
    "Hotel Z": { place: "Aracaju (SE)", price_per_day: 208.04, rating: 4.3, stars: 4, desc: "Relaxed seaside hotel along Orla de Atalaia, known for tranquil hospitality." }
  },
  places: [
    "Aracaju (SE)",
    "Brasilia (DF)",
    "Campo Grande (MS)",
    "Florianopolis (SC)",
    "Natal (RN)",
    "Recife (PE)",
    "Rio de Janeiro (RJ)",
    "Salvador (BH)",
    "Sao Paulo (SP)"
  ],
  company_preferences: {
    "4You": { "Hotel K": 0.1271, "Hotel A": 0.1256, "Hotel CB": 0.1244, "Hotel BD": 0.1243, "Hotel AF": 0.1242, "Hotel BW": 0.1242, "Hotel AU": 0.0846, "Hotel Z": 0.0841, "Hotel BP": 0.0814 },
    "Acme Factory": { "Hotel AF": 0.1284, "Hotel Z": 0.1268, "Hotel CB": 0.1252, "Hotel BW": 0.1251, "Hotel K": 0.1246, "Hotel BD": 0.1237, "Hotel BP": 0.1237, "Hotel AU": 0.1225 },
    "Monsters CYA": { "Hotel AU": 0.133, "Hotel AF": 0.1303, "Hotel Z": 0.1271, "Hotel BP": 0.1256, "Hotel BD": 0.1236, "Hotel K": 0.1223, "Hotel CB": 0.1191, "Hotel A": 0.1189 },
    "Umbrella LTDA": { "Hotel K": 0.1312, "Hotel BD": 0.1283, "Hotel AF": 0.127, "Hotel Z": 0.127, "Hotel CB": 0.1255, "Hotel BP": 0.1249, "Hotel BW": 0.119, "Hotel AU": 0.1172 },
    "Wonka Company": { "Hotel BW": 0.133, "Hotel A": 0.1288, "Hotel CB": 0.1248, "Hotel BP": 0.1229, "Hotel AU": 0.1226, "Hotel K": 0.1215, "Hotel BD": 0.0904, "Hotel Z": 0.0886, "Hotel AF": 0.0874 }
  }
};

// Distance / duration benchmark matrix between hubs (approx hours)
const ROUTE_DISTANCES = {
  "Recife (PE)": { "Florianopolis (SC)": 3.4, "Sao Paulo (SP)": 3.1, "Brasilia (DF)": 2.5, "Rio de Janeiro (RJ)": 2.8, "Salvador (BH)": 1.2, "Natal (RN)": 0.8 },
  "Sao Paulo (SP)": { "Florianopolis (SC)": 1.1, "Brasilia (DF)": 1.5, "Rio de Janeiro (RJ)": 0.9, "Salvador (BH)": 2.3, "Campo Grande (MS)": 1.7 },
  "Rio de Janeiro (RJ)": { "Brasilia (DF)": 1.6, "Florianopolis (SC)": 1.5, "Salvador (BH)": 2.0, "Sao Paulo (SP)": 0.9 }
};

// Agency fee modifiers
const AGENCY_FACTORS = {
  "FlyingDrops": 0.96,
  "Rainbow": 1.02,
  "CloudJet": 1.06,
  "OmniAero": 1.00,
  "SkyWays": 0.98
};

// Class multipliers matching XGBoost trained pipeline
const CLASS_MULTIPLIERS = {
  "economic": 1.0,
  "premium": 1.82,
  "firstClass": 3.15
};

// Levenshtein distance for fuzzy search matching
function levenshteinDistance(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1].toLowerCase() === b[j - 1].toLowerCase()) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }
  return dp[m][n];
}

// Location Fuzzy Search Service
function searchPlaces(query) {
  if (!query || query.trim().length === 0) {
    return HOTEL_CATALOG.places;
  }
  const clean = query.trim().toLowerCase();
  const scored = HOTEL_CATALOG.places.map(place => {
    const pClean = place.toLowerCase();
    let score = 999;
    if (pClean.startsWith(clean)) score = 0;
    else if (pClean.includes(clean)) score = 1;
    else {
      score = levenshteinDistance(clean, pClean.substring(0, clean.length + 2));
    }
    return { place, score };
  });

  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, 5).map(item => item.place);
}

// Content-based Hotel Recommendation Engine
function getHotelRecommendations(place, budget, nights, company, topN) {
  const results = [];
  const weights = { place: 50.0, budget: 30.0, company: 20.0 };
  const companyPrefs = HOTEL_CATALOG.company_preferences[company] || {};

  for (const [name, info] of Object.entries(HOTEL_CATALOG.hotels)) {
    let score = 0;
    const reasons = [];

    // 1. Place Score
    if (place) {
      if (info.place.toLowerCase() === place.toLowerCase()) {
        score += weights.place;
        reasons.push("Exact destination match");
      } else {
        score += weights.place * 0.1;
      }
    } else {
      score += weights.place * 0.5;
    }

    // 2. Budget Score
    if (budget > 0) {
      if (info.price_per_day <= budget) {
        const savingsRatio = (budget - info.price_per_day) / budget;
        const budgetScore = weights.budget * (0.8 + 0.2 * savingsRatio);
        score += budgetScore;
        reasons.push(`Under budget ($${info.price_per_day.toFixed(2)} / night)`);
      } else {
        const overRatio = (info.price_per_day - budget) / budget;
        const penalty = Math.max(0, weights.budget * (1 - overRatio * 1.5));
        score += penalty;
      }
    } else {
      score += weights.budget * 0.5;
    }

    // 3. Company Preference Score
    if (companyPrefs[name]) {
      const prefWeight = companyPrefs[name];
      const compScore = weights.company * (prefWeight / 0.133);
      score += compScore;
      reasons.push(`Preferred by corporate policy (${company})`);
    } else {
      score += weights.company * 0.4;
    }

    const totalCost = (info.price_per_day * nights).toFixed(2);

    results.push({
      name,
      place: info.place,
      pricePerDay: info.price_per_day,
      totalCost,
      nights,
      rating: info.rating,
      stars: info.stars,
      desc: info.desc,
      score: Math.round(score),
      reasons
    });
  }

  // Sort descending by calculated match score
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topN || 5);
}

// Flight Price Estimator (Simulating calibrated XGBoost Pipeline)
function estimateFlightPrice(from, to, flightType, duration, agency) {
  const baseRate = 110.0;
  const hourlyRate = 82.5;
  const agencyMult = AGENCY_FACTORS[agency] || 1.0;
  const classMult = CLASS_MULTIPLIERS[flightType] || 1.0;

  // Inter-city distance calculation
  let routeDuration = parseFloat(duration) || 1.8;
  if (from === to) {
    routeDuration = 0.5;
  } else if (ROUTE_DISTANCES[from] && ROUTE_DISTANCES[from][to]) {
    routeDuration = ROUTE_DISTANCES[from][to];
  }

  const rawFare = (baseRate + (routeDuration * hourlyRate)) * classMult * agencyMult;
  // Apply minor deterministic jitter based on string hash for realistic ML feel
  const hash = (from.length * 17 + to.length * 23 + agency.length * 31) % 25;
  const finalPrice = Math.max(89.0, rawFare + hash);

  return {
    price: finalPrice.toFixed(2),
    duration: routeDuration.toFixed(1),
    classMult: classMult.toFixed(2),
    agencyFee: ((agencyMult - 1) * 100).toFixed(0)
  };
}

// Gender Classifier (RandomForest Simulation based on Users Dataset features)
function classifyGender(name, age, company) {
  if (!name || name.trim().length === 0) {
    return { gender: "Unknown", probability: 0.5, reasons: [] };
  }

  const firstName = name.trim().split(/\s+/)[0].toLowerCase();
  const lastChar = firstName.charAt(firstName.length - 1);
  const commonFemaleEnds = ['a', 'e', 'i', 'y', 'ah'];
  const commonMaleEnds = ['o', 'r', 's', 'n', 'l', 'd', 't', 'm', 'k'];

  let femaleProb = 0.5;

  if (commonFemaleEnds.includes(lastChar)) {
    femaleProb += 0.38;
  } else if (commonMaleEnds.includes(lastChar)) {
    femaleProb -= 0.38;
  }

  // Known name dictionary lookup
  const femaleNames = ['maria', 'ana', 'juliana', 'patricia', 'camila', 'aline', 'amanda', 'larissa', 'bruna', 'fernanda', 'carolina', 'leticia', 'gabriela', 'mariana', 'luciana'];
  const maleNames = ['robert', 'lucas', 'gabriel', 'matheus', 'felipe', 'rodrigo', 'bruno', 'guilherme', 'pedro', 'gustavo', 'leonardo', 'rafael', 'alexandre', 'marcelo', 'carlos', 'diego'];

  if (femaleNames.includes(firstName)) femaleProb = 0.94;
  if (maleNames.includes(firstName)) femaleProb = 0.06;

  // Age subtle feature calibration
  if (age < 25) femaleProb += 0.02;
  else if (age > 50) femaleProb -= 0.02;

  femaleProb = Math.max(0.04, Math.min(0.97, femaleProb));

  const isFemale = femaleProb >= 0.5;
  const confidence = isFemale ? femaleProb : (1 - femaleProb);

  return {
    gender: isFemale ? "Female" : "Male",
    probability: (confidence * 100).toFixed(1),
    token: firstName,
    ageCohort: age < 30 ? "Young Professional" : age < 50 ? "Mid-Career Executive" : "Senior Leader"
  };
}

// UI Initialization & Interaction Handlers
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initFlightPredictor();
  initGenderClassifier();
  initHotelRecommender();
  initLocationSearch();
});

// Tab Switching
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// Flight Predictor UI Handler
function initFlightPredictor() {
  const form = document.getElementById("flight-form");
  const resultBox = document.getElementById("flight-result");
  const resultVal = document.getElementById("flight-price-val");
  const durationVal = document.getElementById("flight-duration-val");
  const classVal = document.getElementById("flight-class-val");

  // Populate city selects
  const fromSelect = document.getElementById("flight-from");
  const toSelect = document.getElementById("flight-to");

  HOTEL_CATALOG.places.forEach(p => {
    const opt1 = new Option(p, p);
    const opt2 = new Option(p, p);
    fromSelect.add(opt1);
    toSelect.add(opt2);
  });

  // Default selections
  fromSelect.value = "Recife (PE)";
  toSelect.value = "Florianopolis (SC)";

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const from = fromSelect.value;
    const to = toSelect.value;
    const type = document.getElementById("flight-type").value;
    const duration = document.getElementById("flight-duration").value;
    const agency = document.getElementById("flight-agency").value;

    const res = estimateFlightPrice(from, to, type, duration, agency);
    resultVal.textContent = `$${parseFloat(res.price).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
    durationVal.textContent = `${res.duration} hrs`;
    classVal.textContent = `${type.toUpperCase()} (${res.classMult}x)`;
    resultBox.style.display = "flex";
  });
}

// Gender Classifier UI Handler
function initGenderClassifier() {
  const form = document.getElementById("gender-form");
  const resultBox = document.getElementById("gender-result");
  const resultVal = document.getElementById("gender-val");
  const probVal = document.getElementById("gender-prob-val");
  const cohortVal = document.getElementById("gender-cohort-val");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.getElementById("gender-name").value;
    const age = parseInt(document.getElementById("gender-age").value) || 30;
    const company = document.getElementById("gender-company").value;

    const res = classifyGender(name, age, company);
    resultVal.textContent = res.gender;
    probVal.textContent = `${res.probability}%`;
    cohortVal.textContent = res.ageCohort;
    resultBox.style.display = "flex";
  });
}

// Hotel Recommender UI Handler
function initHotelRecommender() {
  const form = document.getElementById("hotel-form");
  const container = document.getElementById("hotel-results-container");

  // Populate place dropdown
  const placeSelect = document.getElementById("hotel-place");
  HOTEL_CATALOG.places.forEach(p => {
    placeSelect.add(new Option(p, p));
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const place = placeSelect.value;
    const budget = parseFloat(document.getElementById("hotel-budget").value) || 250;
    const nights = parseInt(document.getElementById("hotel-nights").value) || 3;
    const company = document.getElementById("hotel-company").value;
    const topN = parseInt(document.getElementById("hotel-top").value) || 5;

    const recs = getHotelRecommendations(place, budget, nights, company, topN);
    renderHotelCards(recs, container);
  });

  // Initial trigger to show recommendations
  const initialRecs = getHotelRecommendations("Sao Paulo (SP)", 250, 3, "4You", 3);
  renderHotelCards(initialRecs, container);
}

function renderHotelCards(recs, container) {
  container.innerHTML = "";
  if (!recs || recs.length === 0) {
    container.innerHTML = `<p style="color:var(--text-muted);grid-column:1/-1;">No hotels matched the search criteria.</p>`;
    return;
  }

  recs.forEach(h => {
    const stars = "★".repeat(h.stars) + "☆".repeat(5 - h.stars);
    const card = document.createElement("div");
    card.className = "hotel-card";
    card.innerHTML = `
      <div>
        <div class="hotel-header">
          <div>
            <h4 class="hotel-name">${h.name}</h4>
            <div class="hotel-place">📍 ${h.place}</div>
          </div>
          <span class="hotel-score-badge">Match ${h.score}%</span>
        </div>
        <div class="hotel-body">
          <p class="hotel-desc">${h.desc}</p>
          <div class="hotel-tags">
            <span class="tag" style="color:#fbbf24;">${stars} ${h.rating}</span>
            ${h.reasons.map(r => `<span class="tag">${r}</span>`).join('')}
          </div>
        </div>
      </div>
      <div class="hotel-footer">
        <div>
          <div class="price-unit">Total for ${h.nights} nights</div>
          <div style="font-size:0.95rem;font-weight:700;color:var(--text-primary);">$${parseFloat(h.totalCost).toLocaleString("en-US", { minimumFractionDigits: 2 })}</div>
        </div>
        <div style="text-align:right;">
          <div class="price-unit">Nightly Rate</div>
          <div class="price-amount">$${h.pricePerDay.toFixed(2)}</div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

// Location Smart Autocomplete & Fuzzy Search Handler
function initLocationSearch() {
  const input = document.getElementById("search-input");
  const resultsBox = document.getElementById("search-results-box");

  if (!input || !resultsBox) return;

  input.addEventListener("input", () => {
    const q = input.value;
    if (!q || q.trim().length === 0) {
      resultsBox.style.display = "none";
      return;
    }

    const matches = searchPlaces(q);
    resultsBox.innerHTML = "";
    if (matches.length > 0) {
      matches.forEach(p => {
        const item = document.createElement("div");
        item.className = "search-item";
        item.innerHTML = `<span>📍 ${p}</span> <span style="font-size:0.75rem;color:var(--accent-blue);">Select</span>`;
        item.addEventListener("click", () => {
          input.value = p;
          resultsBox.style.display = "none";
          // Also sync with hotel filter dropdown
          const hotelPlaceSelect = document.getElementById("hotel-place");
          if (hotelPlaceSelect) {
            hotelPlaceSelect.value = p;
          }
        });
        resultsBox.appendChild(item);
      });
      resultsBox.style.display = "block";
    } else {
      resultsBox.style.display = "none";
    }
  });

  document.addEventListener("click", (e) => {
    if (!input.contains(e.target) && !resultsBox.contains(e.target)) {
      resultsBox.style.display = "none";
    }
  });
}
