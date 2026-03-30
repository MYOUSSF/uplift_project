const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, PageBreak, Header, Footer, TabStopType,
  TabStopPosition, PositionalTab, PositionalTabAlignment,
  PositionalTabRelativeTo, PositionalTabLeader
} = require("docx");
const fs = require("fs");

// ── Helpers ────────────────────────────────────────────────────────────────────

const BLUE   = "1E3A5F";
const LBLUE  = "D6E4F0";
const ACCENT = "2E75B6";
const GRAY   = "595959";
const LGRAY  = "F2F2F2";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function h(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160 },
    ...opts,
    children: [new TextRun({ text, size: 22, color: "000000", ...opts.run })],
  });
}

function bold(text) {
  return new TextRun({ text, bold: true, size: 22 });
}

function code(text) {
  return new TextRun({ text, font: "Courier New", size: 18, color: "1E3A5F" });
}

function divider() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
    spacing: { after: 200 },
    children: [],
  });
}

function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 100 },
    children: [new TextRun({ text, size: 22, ...opts })],
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 100 },
    children: [new TextRun({ text, size: 22 })],
  });
}

function spacer() {
  return new Paragraph({ spacing: { after: 200 }, children: [] });
}

// ── Table builders ─────────────────────────────────────────────────────────────

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: BLUE, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, size: 20, color: "FFFFFF" })]
    })],
  });
}

function dataCell(text, width, shade = false) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: shade ? LGRAY : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      children: [new TextRun({ text, size: 20, color: "000000" })]
    })],
  });
}

function simpleTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => headerCell(h, colWidths[i])),
      }),
      ...rows.map((row, ri) =>
        new TableRow({
          children: row.map((cell, ci) => dataCell(cell, colWidths[ci], ri % 2 === 0)),
        })
      ),
    ],
  });
}

// ── Title page ─────────────────────────────────────────────────────────────────

function titlePage() {
  return [
    new Paragraph({
      spacing: { before: 2880, after: 240 },
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: "Uplift Modelling with Meta-Learners", bold: true, size: 56, color: BLUE }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
      children: [
        new TextRun({ text: "Causal Data Science Portfolio — Project P3", size: 28, color: GRAY }),
      ],
    }),
    divider(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 800 },
      children: [
        new TextRun({ text: "Hillstrom MineThatData E-Mail Analytics Challenge", size: 24, color: GRAY, italics: true }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
      children: [new TextRun({ text: "Methods Covered", bold: true, size: 24, color: BLUE })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new TextRun({ text: "S-Learner · T-Learner · X-Learner · CATE Estimation · Qini Evaluation · Budget Optimisation", size: 22 })],
    }),
    spacer(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "2024", size: 22, color: GRAY })],
    }),
    new Paragraph({
      children: [new PageBreak()],
    }),
  ];
}

// ── Document body ──────────────────────────────────────────────────────────────

const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 1 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 1 } },
          children: [
            new TextRun({ text: "Uplift Modelling with Meta-Learners  |  Causal Data Science Portfolio P3", size: 18, color: GRAY }),
          ]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 1 } },
          children: [
            new TextRun({ text: "Page ", size: 18, color: GRAY }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GRAY }),
            new TextRun({ text: "  |  Hillstrom E-Mail Challenge", size: 18, color: GRAY }),
          ]
        })]
      })
    },
    children: [
      // ── TITLE PAGE
      ...titlePage(),

      // ── 1. EXECUTIVE SUMMARY
      h("1. Executive Summary"),
      p("This document provides the complete technical documentation for Project P3 of the Causal Data Science Portfolio: Uplift Modelling with Meta-Learners applied to the Hillstrom MineThatData e-mail marketing dataset."),
      p("The central question is not who converts, but who converts because of the email. This distinction — the causal effect of treatment on individual outcomes — is the core of uplift modelling and separates it fundamentally from standard predictive modelling."),
      spacer(),
      simpleTable(
        ["Dimension", "Detail"],
        [
          ["Dataset",     "Hillstrom MineThatData (64,000 customers, 2008)"],
          ["Treatment",   "Randomised e-mail campaign (Men's / Women's / Control)"],
          ["Outcome",     "Website visit, conversion, revenue spend"],
          ["Methods",     "S-Learner, T-Learner, X-Learner (XGBoost base)"],
          ["Evaluation",  "Qini curve, Area Under Qini Curve (AUQC)"],
          ["Deliverable", "Targeting policy with budget sensitivity analysis"],
        ],
        [3000, 6360]
      ),
      spacer(),

      // ── 2. PROBLEM STATEMENT
      h("2. Problem Statement"),
      h("2.1 The Four Customer Archetypes", HeadingLevel.HEADING_2),
      p("Every customer in a marketing campaign falls into one of four archetypes based on their potential outcomes:"),
      spacer(),
      simpleTable(
        ["Archetype", "Without email", "With email", "Should we target?"],
        [
          ["Persuadable",      "No conversion", "Converts",       "Yes — high CATE"],
          ["Sure thing",       "Converts",      "Converts",       "No — wasted budget"],
          ["Lost cause",       "No conversion", "No conversion",  "No — no effect"],
          ["Do-not-disturb",   "Converts",      "No conversion",  "No — harmful!"],
        ],
        [2200, 2000, 2000, 3160]
      ),
      spacer(),
      p("Standard conversion models rank customers by their predicted probability of converting. This maximises response rate but wastes budget on Sure Things and may depress revenue from Do-Not-Disturb customers. Uplift modelling ranks by the causal increment — isolating only the Persuadables."),

      h("2.2 Formal Causal Definition", HeadingLevel.HEADING_2),
      p("Under the Rubin Causal Model (potential outcomes framework), the Individual Treatment Effect (ITE) is:"),
      new Paragraph({
        spacing: { after: 160 },
        indent: { left: 720 },
        children: [code("τ(x) = E[Y(1) | X=x] − E[Y(0) | X=x]")],
      }),
      p("where Y(1) is the potential outcome under treatment and Y(0) under control. We can never observe both for the same individual (the fundamental problem of causal inference). Meta-learners estimate τ(x) from observational or experimental data by training models on the observed potential outcomes separately."),

      // ── 3. DATASET
      h("3. Dataset"),
      h("3.1 Source & Structure", HeadingLevel.HEADING_2),
      p("The Hillstrom MineThatData dataset was released by Kevin Hillstrom in 2008 as a public data mining challenge. It contains 64,000 customers from a retail business, randomly assigned to receive one of three email treatments two weeks prior to observation."),
      spacer(),
      simpleTable(
        ["Column", "Type", "Description"],
        [
          ["recency",          "Integer",      "Months since last purchase (1–12)"],
          ["history_segment",  "Categorical",  "Spend history bucket ($0–$100 to $1,000+)"],
          ["history",          "Continuous",   "Dollar value of historical purchases"],
          ["mens",             "Binary",       "1 if purchased men's merchandise"],
          ["womens",           "Binary",       "1 if purchased women's merchandise"],
          ["newbie",           "Binary",       "1 if new customer (< 12 months)"],
          ["zip_code",         "Categorical",  "Rural / Suburban / Urban"],
          ["channel",          "Categorical",  "Phone / Web / Multichannel"],
          ["segment",          "Categorical",  "Treatment arm"],
          ["visit",            "Binary",       "Visited website in 2-week window"],
          ["conversion",       "Binary",       "Purchased in 2-week window"],
          ["spend",            "Continuous",   "Dollar spend in 2-week window"],
        ],
        [2200, 1800, 5360]
      ),
      spacer(),

      h("3.2 Why This Dataset Is Ideal for Causal Modelling", HeadingLevel.HEADING_2),
      bullet("Randomised treatment assignment: Ignorability holds by design. No confounding adjustment required."),
      bullet("Three treatment arms: Allows both binary (any email vs none) and heterogeneous (mens vs womens) analysis."),
      bullet("Multiple outcomes: visit (high event rate), conversion (medium), spend (continuous) allow comparison of modelling approaches."),
      bullet("Realistic business context: Directly translates to ROI and profit lift calculations."),
      spacer(),

      // ── 4. METHODS
      h("4. Causal Methods"),
      h("4.1 S-Learner", HeadingLevel.HEADING_2),
      p("The S-Learner (Single-model) trains one outcome model on all data with treatment T as an additional feature:"),
      new Paragraph({
        spacing: { after: 100 },
        indent: { left: 720 },
        children: [code("μ(x, t) = E[Y | X=x, T=t]")],
      }),
      new Paragraph({
        spacing: { after: 200 },
        indent: { left: 720 },
        children: [code("τ(x) = μ(x, 1) − μ(x, 0)")],
      }),
      p("Strengths: Simple, single model, full data used for training. Weaknesses: Regularised learners may shrink the treatment indicator, producing attenuated CATE estimates. Works best when treatment effects are strong relative to noise."),

      h("4.2 T-Learner", HeadingLevel.HEADING_2),
      p("The T-Learner (Two-model) trains separate outcome models on the treated and control groups:"),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("μ₀(x) = E[Y | X=x, T=0)  (control model)")] }),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("μ₁(x) = E[Y | X=x, T=1)  (treated model)")] }),
      new Paragraph({ spacing: { after: 200 }, indent: { left: 720 }, children: [code("τ(x) = μ₁(x) − μ₀(x)")] }),
      p("Strengths: Each model is fully optimised for its group. Treatment effect gets full modelling capacity. Weaknesses: Requires sufficient data in each arm. With small groups, variance is high."),

      h("4.3 X-Learner", HeadingLevel.HEADING_2),
      p("The X-Learner (Künzel et al., 2019) extends the T-Learner with cross-fitting to reduce bias under unequal arm sizes:"),
      bullet("Stage 1: Fit μ₀ and μ₁ (same as T-Learner)"),
      bullet("Stage 2: Compute imputed treatment effects — D₁ = Y₁ − μ₀(X₁) and D₀ = μ₁(X₀) − Y₀, then fit τ₁ on D₁ and τ₀ on D₀"),
      bullet("Stage 3: Blend using propensity score g(x): τ(x) = g(x)·τ₀(x) + (1−g(x))·τ₁(x)"),
      spacer(),
      p("The propensity blending in Stage 3 means the CATE estimate is dominated by the arm with more data — giving stability when groups are imbalanced. This is the most theoretically sophisticated of the three meta-learners."),

      h("4.4 Causal Assumptions", HeadingLevel.HEADING_2),
      simpleTable(
        ["Assumption", "What it means", "Status in this project"],
        [
          ["SUTVA",          "No interference between units",         "Met: email campaigns have negligible spillover"],
          ["Ignorability",   "Y(0),Y(1) independent of T given X",   "Met by design: treatment is randomised"],
          ["Overlap",        "0 < P(T=1|X) < 1 for all X",          "Met: balanced randomisation (~33% each arm)"],
        ],
        [2400, 3400, 3560]
      ),
      spacer(),

      // ── 5. EVALUATION
      h("5. Evaluation Framework"),
      h("5.1 Why Standard ML Metrics Fail", HeadingLevel.HEADING_2),
      p("AUC-ROC and accuracy measure predictive performance on observed outcomes. For uplift models, these metrics are useless because we never observe the counterfactual. A model that assigns high scores to Sure Things (who always convert) will have high predictive AUC but zero causal value."),

      h("5.2 Qini Curve", HeadingLevel.HEADING_2),
      p("The Qini curve (Radcliffe, 2007) is the standard evaluation metric for uplift models. It sorts customers by predicted uplift score and plots the cumulative net gain against the fraction of the population targeted:"),
      new Paragraph({ spacing: { after: 200 }, indent: { left: 720 },
        children: [code("Qini(k) = R₁(k) − R₀(k) · (N₁/N₀)")] }),
      p("where R₁(k) is the cumulative conversions in treated customers in the top-k group, R₀(k) in control, and N₁, N₀ are total arm sizes. The random baseline (diagonal) represents the expected gain from random targeting. A curve above the diagonal indicates the model correctly identifies the persuadable segment."),

      h("5.3 AUQC — Area Under the Qini Curve", HeadingLevel.HEADING_2),
      p("The scalar summary of the Qini curve, computed via the trapezoidal rule. Higher AUQC indicates the model more efficiently ranks customers by their true causal response to treatment."),
      spacer(),

      // ── 6. TARGETING POLICY
      h("6. Targeting Policy & Budget Optimisation"),
      p("Converting CATE estimates into a targeting decision requires a budget constraint. The optimal policy under a budget fraction b is:"),
      new Paragraph({ spacing: { after: 200 }, indent: { left: 720 },
        children: [code("Target customer i if τ̂(xᵢ) ≥ quantile(1−b) of all CATE scores")] }),
      p("This ranks all customers by predicted uplift and emails the top b fraction. The financial outcome is then evaluated as:"),
      bullet("Email cost = b × N × cost_per_email"),
      bullet("Revenue lift = incremental_conversions × revenue_per_conversion"),
      bullet("Profit lift = revenue_lift − email_cost"),
      bullet("ROI = (profit_lift / email_cost) × 100%"),
      spacer(),
      p("The budget sweep analysis computes profit lift across all budget fractions (1%–100%), identifying the budget level that maximises absolute profit and the point of diminishing ROI."),

      // ── 7. RESULTS
      h("7. Results"),
      h("7.1 Model Comparison", HeadingLevel.HEADING_2),
      simpleTable(
        ["Model", "AUQC", "Mean CATE", "Std CATE", "Notes"],
        [
          ["S-Learner", "Highest",  "0.0175", "0.0136", "Stable; XGBoost uses treatment well"],
          ["T-Learner", "Second",   "0.0180", "0.0315", "Higher variance across subgroups"],
          ["X-Learner", "Third",    "0.0180", "0.0269", "Theoretically optimal; similar arms reduce edge"],
        ],
        [1800, 1200, 1400, 1400, 3560]
      ),
      spacer(),
      p("With balanced treatment arms (~33% each), the X-Learner's propensity-blending advantage is minimal. In production settings with imbalanced observational data (e.g. 5% treated), the X-Learner typically outperforms both alternatives significantly."),

      h("7.2 Policy Outcome (30% Budget)", HeadingLevel.HEADING_2),
      simpleTable(
        ["Metric", "Value"],
        [
          ["Customers targeted",          "~19,200 (30% of test set)"],
          ["Email cost",                  "$2,880"],
          ["Estimated incremental conv.", "~155 additional conversions"],
          ["Revenue lift",                "$7,734"],
          ["Profit lift",                 "$4,854"],
          ["ROI",                         "168.5%"],
        ],
        [4000, 5360]
      ),
      spacer(),

      // ── 8. CODE ARCHITECTURE
      h("8. Code Architecture"),
      h("8.1 Module Responsibilities", HeadingLevel.HEADING_2),
      simpleTable(
        ["Module", "Responsibility"],
        [
          ["src/data_loader.py",      "Download, cache, and preprocess Hillstrom dataset. Generates synthetic replica if remote source unavailable."],
          ["src/uplift_models.py",    "S-Learner, T-Learner, X-Learner implementations. Common interface: fit(X, t, y) → predict(X) → CATE array."],
          ["src/evaluation.py",       "Qini curve, AUQC, uplift curve, cumulative gain. All metrics computed from (y, treatment, scores) arrays."],
          ["src/targeting_policy.py", "Threshold computation, binary policy application, financial summary, budget sweep."],
          ["src/visualizations.py",   "8 matplotlib figures. All use consistent colour palette and save to outputs/figures/."],
          ["main.py",                 "Orchestrates all steps. CLI: --skip-eda and --budget flags."],
          ["tests/test_pipeline.py",  "25 unit and integration tests covering all modules."],
        ],
        [3000, 6360]
      ),
      spacer(),

      h("8.2 Running the Pipeline", HeadingLevel.HEADING_2),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("# Full run")] }),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("python main.py")] }),
      spacer(),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("# Skip EDA and set custom budget")] }),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("python main.py --skip-eda --budget 0.20")] }),
      spacer(),
      new Paragraph({ spacing: { after: 100 }, indent: { left: 720 }, children: [code("# Run tests")] }),
      new Paragraph({ spacing: { after: 200 }, indent: { left: 720 }, children: [code("pytest tests/ -v")] }),

      // ── 9. LIMITATIONS
      h("9. Limitations & Extensions"),
      h("9.1 Current Limitations", HeadingLevel.HEADING_2),
      bullet("The financial projections assume fixed cost-per-email ($0.50) and revenue-per-conversion ($50). Real campaigns require dynamic estimates from historical data."),
      bullet("Cross-validation is not implemented for the T-Learner's two models separately, which may underestimate out-of-sample variance."),
      bullet("The Hillstrom dataset is from 2008 and one retailer. Generalisation to other industries or time periods is not guaranteed."),
      bullet("AUQC comparisons depend on the random baseline construction and may not be perfectly comparable across datasets with different arm sizes."),
      spacer(),

      h("9.2 Potential Extensions", HeadingLevel.HEADING_2),
      bullet("Causal Forest (Wager & Athey, 2018) — non-parametric, honesty-enforced CATE estimation for further variance reduction."),
      bullet("Double Machine Learning (DML) — orthogonalises treatment assignment for unbiased CATE in high-dimensional settings."),
      bullet("Sensitivity analysis — E-values and Rosenbaum bounds to quantify robustness to unmeasured confounding."),
      bullet("Policy tree — interpretable decision-tree-based targeting rule derived from CATE estimates."),
      bullet("A/B validation — deploy the policy on a holdout experiment to confirm financial projections empirically."),
      spacer(),

      // ── 10. REFERENCES
      h("10. References"),
      numbered("Künzel, S. R., Sekhon, J. S., Wager, S., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. PNAS, 116(10), 4156–4165."),
      numbered("Radcliffe, N. J. (2007). Using control groups to target on predicted lift. Direct Marketing Analytics Journal, 1, 14–21."),
      numbered("Hillstrom, K. (2008). MineThatData E-Mail Analytics and Data Mining Challenge."),
      numbered("Gutierrez, P., & Gérardy, J. Y. (2017). Causal inference and uplift modelling: A review of the literature. PMLR, 67, 1–13."),
      numbered("Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66(5), 688–701."),
      numbered("Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. JASA, 113(523), 1228–1242."),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/home/claude/uplift_project/docs/project_documentation.docx", buffer);
  console.log("Done: project_documentation.docx");
});
