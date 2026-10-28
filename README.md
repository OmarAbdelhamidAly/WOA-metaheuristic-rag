# 🐋 Whale Optimization Algorithm (WOA) & RAG Simulator
### منصة محاكاة خوارزمية الحوت والاستمثال الآلي لأنظمة الـ RAG المتقدمة
> **جاهزة للتشغيل بالحاويات (Docker-Ready) والنشر المباشر على منصة Render (Render 1-Click Deploy)**

---

### 👥 فريق العمل (Team 6 &bull; Faculty of Computers & Artificial Intelligence)
- **Omar Abdelhamid**
- **Abdelmasih Ashraf**
- **Mohamed Maher**
- **Hussein Mohamed**
- **تاريخ العرض:** 09 / 11 / 2026

---

## 🌟 مميزات المشروع والمنصة التفاعلية (Key Features)

تم بناء هذا المشروع ليكون مستقلاً تماماً بريبو خاص به، ويحتوي على خادم متكامل (FastAPI) وواجهة تفاعلية بصرية حديثة (HTML5 / Vanilla CSS / JavaScript) بدون أي أدوات بناء معقدة، ويعمل داخل حاوية **Docker** خفيفة جاهزة للنشر على **Render**.

1. **🌊 محاكاة حية لسرب الحيتان (HTML5 Canvas 2D Swarm Simulation):**
   - رسم ديناميكي مباشر لتحركات الحيتان في الفضاء البحثي.
   - تجسيد بصري للحوت القائد ($X^*$) بنجم ذهبي متوهج.
   - رسم مسار الهجوم اللولبي بالفقاعات (**Logarithmic Spiral**) باللون البنفسجي، وحركة التشتت والاستكشاف (**Global Exploration**) باللون الوردي.
   - لوحة تحكم كاملة: تشغيل (Play)، إيقاف مؤقت (Pause)، خطوة بخطوة (Step)، إعادة تعيين (Reset)، وتحكم في سرعة العرض.
   - عدادات آنية (Telemetry) لقيم $a$ (من 2 إلى 0) والمتجه $|A|$ وقرار الاحتمال $p$.

2. **🤖 مختبر الضبط الآلي لمعاملات الـ RAG (5D RAG Auto-Tuner):**
   - ضبط تفاعلي لـ 5 معاملات حيوية: `Chunk Size`, `Chunk Overlap`, `Top-K`, `Hybrid Alpha`, `Rerank Threshold`.
   - زر لتشغيل استمثال WOA الفوري وعرض نسبة التحسن (+52.6% مكسب مقارنة بالإعدادات التقليدية).
   - رسم بياني لمنحنى التقارب (Convergence Curve) ومخطط راداري (Radar Chart) لأفضل توزيع للمعاملات.

3. **📊 مقارنة معيارية حية بين 5 خوارزميات (5-Algorithm Benchmark):**
   - مقارنة عادلة تحت نفس الميزانية الحسابية بين:
     - **Grid Search** (البحث الشبكي)
     - **Random Search** (البحث العشوائي)
     - **Genetic Algorithm - GA** (الخوارزمية الجينية)
     - **Particle Swarm - PSO** (سرب الجسيمات)
     - **Whale Optimization - WOA** (خوارزمية الحوت)
   - رسم بياني للأعمدة وجدول مقارن بالدقة والاستدعاء والزمن.

4. **📝 حاسبة مسائل الامتحان التفاعلية (Step-by-Step Exam Solver):**
   - حل فوري للمسائل الثلاثة المتوقعة في الامتحان بخطوات حسابية مفصلة:
     - **المسألة الأولى:** 1D Encircling ($|A| < 1$).
     - **المسألة الثانية:** 2D Spiral Attack ($p \ge 0.5$).
     - **المسألة الثالثة:** Global Exploration Rule ($|A| \ge 1$ مع حوت عشوائي $X_{rand}$).

---

## 🚀 طرق التشغيل (How to Run)

### الطريقة الأولى: التشغيل المحلي المباشر عبر Python (الأسرع للتجربة)
1. تثبيت المتطلبات:
   ```powershell
   pip install -r requirements.txt
   ```
2. تشغيل السيرفر:
   ```powershell
   python run.py
   ```
3. افتح المتصفح على الرابط:
   👉 **`http://localhost:5000`**

---

### الطريقة الثانية: التشغيل عبر Docker محلياً (Docker Compose)
1. تشغيل الحاوية:
   ```powershell
   docker-compose up --build
   ```
2. افتح المتصفح على:
   👉 **`http://localhost:5000`**

---

### الطريقة الثالثة: الرفع والنشر على منصة Render (Render Deployment)
المشروع مهيأ بالكامل للنشر التلقائي على **Render**:
1. أنشئ ريبو جديد على حسابك في GitHub (مثلاً: `woa-metaheuristic-rag`).
2. ارفع مجلد المشروع إليه (راجع قسم Git بالأسفل).
3. في لوحة تحكم Render:
   - اضغط **New +** ثم اختر **Web Service**.
   - اربط الريبو الخاص بك `woa-metaheuristic-rag`.
   - سيتعرف Render تلقائياً على ملف `Dockerfile` وملف `render.yaml`.
   - اضغط **Create Web Service**.
4. سيبدأ البناء تلقائياً وسيعطيك Render رابطاً عاماً سريعاً (مثال: `https://woa-metaheuristic-rag.onrender.com`).

---

## 📦 عزل المشروع في ريبو مستقل (Git Repository Setup)

تم تهيئة هذا المجلد ليكون **مستودع Git مستقل تماماً** ومنفصل عن مشروعك السابق:

```powershell
# 1. الدخول لمجلد المشروع
cd woa_rag_simulation

# 2. تهيئة الريبو وإضافة كافة الملفات
git init
git add .
git commit -m "Initial commit: Whale Optimization Algorithm (WOA) & RAG Web Simulator"

# 3. ربطه بمستودعك الجديد على GitHub ورفعه
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/woa-metaheuristic-rag.git
git push -u origin main
```

---

## 🗂️ هيكل الملفات وشرح الكود (Architecture Walkthrough)

```
woa_rag_simulation/
│
├── Dockerfile                     # ملف بناء إيميدج دوكر للإنتاج ولمنصة Render
├── docker-compose.yml             # ملف التشغيل المحلي بضغطة زر
├── render.yaml                    # إعدادات النشر الأوتوماتيكي على Render (Blueprint)
├── requirements.txt               # المكتبات المطلوبة (FastAPI, NumPy, Matplotlib, SciPy...)
├── run.py                         # سكريبت بدء التشغيل الذكي (يقرأ المنفذ $PORT تلقائياً)
├── server.py                      # خادم FastAPI الذي يوفر الـ REST APIs
│
├── public/                        # واجهة المستخدم التفاعلية (Frontend)
│   ├── index.html                 # الصفحة الرئيسية المبوبة (Single Page App)
│   ├── style.css                  # التصميم المظلم المحيطي الجذاب (Glassmorphism & Neon Glow)
│   └── app.js                     # محرك الرسوم على الكانفاس وربط الـ APIs
│
├── core/                          # المحرك الرياضي
│   ├── woa.py                     # كود خوارزمية الحوت الصافي بـ NumPy
│   ├── baseline_solvers.py        # خوارزميات المقارنة (Grid, Random, PSO, GA)
│   └── test_functions.py          # الدوال الرياضية القياسية (Sphere, Ackley, Rastrigin)
│
├── rag_pipeline/                  # محاكاة خط أنابيب الـ RAG
│   ├── search_space.py            # فك تشفير المتغيرات الخمسة والقيود الفيزيائية
│   ├── synthetic_corpus.py        # قاعدة البيانات المعرفية والأسئلة الاختبارية
│   └── evaluator.py               # التقطيع، البحث الهجين BM25+Dense، والتقييم بـ RAGAS
│
├── exam_problems/                 # حل مسائل الامتحان
│   └── exam_solver.py             # الحل التحليلي الرقمي خطوة بخطوة
│
├── experiments/                   # المقارنات المعيارية
│   ├── run_benchmark.py           # تنفيذ المقارنة الشاملة
│   └── visualizer.py              # رسم وحفظ منحنيات التقارب والمخطط الراداري
│
└── tests/                         # الاختبارات الآلية
    └── test_woa.py                # 7 اختبارات جودة تفحص النواة الرياضية ومسائل الامتحان
```

---

## 📡 واجهات الـ REST API المتوفرة في السيرفر

| المسار (Endpoint) | الطريقة (Method) | الوظيفة |
| :--- | :---: | :--- |
| `/health` | `GET` | فحص صحة الخادم (يستخدمه Render للـ Health Check). |
| `/` | `GET` | عرض واجهة المحاكاة الرسومية الكاملة. |
| `/api/simulation/run` | `POST` | محاكاة تحركات سرب الحيتان دورة بدورة وإرجاع إحداثيات كل كادر للكانفاس. |
| `/api/rag/optimize` | `POST` | تنفيذ استمثال WOA على الـ RAG وإرجاع أفضل قيم ومقاييس الأداء. |
| `/api/benchmark/run` | `POST` | تشغيل الخوارزميات الخمس وإرجاع جدول المقارنة والمنحنيات. |
| `/api/exam/solve` | `POST` | حل المسألة المحددة (1 أو 2 أو 3) وإرجاع خطوات التعويض الرقمي. |

---

## 📚 المراجع الأكاديمية (References)
1. **Mirjalili & Lewis (2016)** - *The Whale Optimization Algorithm*, Advances in Engineering Software.
2. **Kim et al. (AutoRAG, Oct 2024)** - *Automated Framework for Optimization of RAG Pipelines*, arXiv:2410.20878.
3. **Es et al. (RAGAS, 2023)** - *Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217.
4. **Kartal et al. (RAGSmith, Jan 2025)** - *A Framework for Holistic RAG Pipeline Optimization*, arXiv:2501.18950.
