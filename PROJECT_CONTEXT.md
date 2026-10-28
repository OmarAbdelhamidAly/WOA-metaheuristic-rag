# WOA Metaheuristic RAG — Project Context & Handoff Guide

## 1. Project Overview & Objective
This repository (woa-metaheuristic-rag) is an isolated, production-ready, interactive simulation and optimization platform. It combines:
1. **Whale Optimization Algorithm (WOA)**: A nature-inspired metaheuristic mimicking the bubble-net hunting strategy of humpback whales (Mirjalili & Lewis, 2016).
2. **RAG Hyperparameter Optimization**: Using WOA to optimize continuous & discrete parameters in a Hybrid Retrieval-Augmented Generation pipeline (chunk size, chunk overlap, dense vs. sparse weighting alpha, top-k, and reranker threshold).
3. **Interactive Web Simulation UI**: A pure HTML5 Canvas + modern Vanilla CSS + JS dashboard for real-time 2D swarm visualization, parameter auto-tuning, benchmark comparisons (WOA vs. PSO, GA, GWO, DE), and step-by-step academic exam problem breakdowns.
4. **Cloud Containerization**: Production Docker setup (Dockerfile, docker-compose.yml, 
ender.yaml) configured for free/instant deployment on Render.com with dynamic $PORT.

---

## 2. Mathematical Foundation Implemented
The WOA algorithm implemented in core/woa.py executes three distinct hunting mechanisms:

### Phase 1: Encircling Prey (p < 0.5 and |A| < 1)
D = |C * X*(t) - X(t)|
X(t+1) = X*(t) - A * D
Where:
- a = 2 - 2 * (t / T_max) (linearly decreases from 2 to 0)
- A = 2a * r1 - a
- C = 2 * r2

### Phase 2: Bubble-Net Spiral Attack (p >= 0.5)
Whales swim along a 3D logarithmic spiral towards the prey:
D_prime = |X*(t) - X(t)|
X(t+1) = D_prime * exp(b * l) * cos(2 * pi * l) + X*(t)
Where:
- b is the spiral shape constant (default b = 1)
- l in [-1, 1] is a uniform random number

### Phase 3: Global Exploration / Prey Search (p < 0.5 and |A| >= 1)
When |A| >= 1, whales steer towards a randomly selected peer whale X_rand:
D = |C * X_rand - X(t)|
X(t+1) = X_rand - A * D

---

## 3. RAG 5D Search Space & Objective Function
Located in 
ag_pipeline/search_space.py and 
ag_pipeline/evaluator.py:
- x_0 in [128, 1024]: Chunk Size (tokens)
- x_1 in [20, 256]: Chunk Overlap (tokens, constrained <= 0.5 * Chunk Size)
- x_2 in [0.0, 1.0]: Hybrid Retrieval Weight alpha (Score = alpha * Dense + (1-alpha) * BM25)
- x_3 in [2, 20]: Top-k Retrieval Depth
- x_4 in [0.1, 0.9]: Cross-Encoder Reranker Score Threshold

### Composite Loss Function:
L_RAG = w1*(1 - HitRate@k) + w2*(1 - MRR) + w3*(1 - Faithfulness) + w4*LatencyPenalty + w5*CostPenalty
WOA minimizes L_RAG using the synthetic corpus and evaluation suite.

---

## 4. Repository Structure & Key Components
`
woa-metaheuristic-rag/
|-- core/
|   |-- woa.py                  # Full WOA engine with iteration callbacks & frame recording
|   |-- baseline_solvers.py     # PSO, GA, GWO, DE for multi-algorithm benchmarking
|   +-- test_functions.py       # Sphere, Rastrigin, Rosenbrock, Griewank, Ackley
|-- rag_pipeline/
|   |-- search_space.py         # 5D hyperparameter bounds & discrete decoders
|   |-- synthetic_corpus.py     # 100 enterprise QA documents + gold query set
|   +-- evaluator.py            # Hybrid BM25+Dense retrieval & RAG loss function
|-- exam_problems/
|   +-- exam_solver.py          # Step-by-step mathematical workout for university exam questions
|-- experiments/
|   |-- run_benchmark.py        # 30-run statistical comparison (Mean, Std, Wilcoxon p-value)
|   +-- visualizer.py           # Convergence plots & radar chart data generation
|-- public/                     # Frontend SPA (Vanilla JS + Modern CSS Glassmorphism)
|   |-- index.html              # 5-Tab interface (Swarm Viewport, RAG Tuner, Benchmarks, Exam Solver, Docs)
|   |-- style.css               # Dark theme, neon accents, responsive layout
|   +-- app.js                  # HTML5 Canvas real-time particle animation & REST API client
|-- tests/
|   +-- test_woa.py             # 7 unit tests covering WOA convergence, bounds, and RAG evaluator
|-- Dockerfile                  # Python 3.11-slim container with curl healthcheck
|-- docker-compose.yml          # Container configuration exposing port 5000:5000
|-- render.yaml                 # One-click Render.com deployment blueprint
|-- requirements.txt            # FastAPI, Uvicorn, NumPy, SciPy, Pytest
|-- run.py                      # Production entrypoint reading dynamic 
|-- server.py                   # FastAPI REST API exposing simulation, tuning, benchmark, exam endpoints
+-- README.md                   # Full Arabic & English documentation
`

---

## 5. API Endpoints (server.py)
- GET / -> Serves public/index.html
- POST /api/simulation/run -> Runs WOA on 2D benchmark functions, returning frame-by-frame whale coordinates for canvas animation.
- POST /api/rag/optimize -> Runs WOA on the 5D RAG space and returns optimal hyperparameters, fitness progression, and retrieval metrics.
- POST /api/benchmark/run -> Executes WOA, PSO, GA, GWO, and DE simultaneously and returns comparative stats.
- POST /api/exam/solve -> Returns step-by-step mathematical substitutions and manual vector calculations for exam questions.

---

## 6. How to Run Locally & In Docker
`ash
# Local Python:
pip install -r requirements.txt
python run.py
# Open: http://localhost:5000

# Docker:
docker build -t woa-metaheuristic-rag .
docker run -p 5000:5000 woa-metaheuristic-rag

# Run Tests:
pytest tests/
`

---

## 7. Next Immediate Steps for the New Chat
1. GitHub Setup: Push to a new GitHub repo YOUR_USERNAME/woa-metaheuristic-rag.
2. Render Deployment: Connect GitHub repo to Render as a Web Service (Docker runtime).
3. Feature Extensions:
   - Add real LLM evaluation connectors (e.g. LangChain, LlamaIndex, or Ragas).
   - Add 3D WebGL / Three.js whale trajectory rendering.
   - Export optimized RAG configurations as .env or YAML profiles.
