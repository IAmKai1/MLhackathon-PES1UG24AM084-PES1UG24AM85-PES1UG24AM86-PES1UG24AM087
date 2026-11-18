"""Build the PathSnap technical report PDF (structure from Deliverables.pdf, sections 2-7).

Tables and figures are read from results/ at build time, so every number in the PDF is
verifiable from the repository. Written explanations come from report/text/<key>.md:
if a file exists its text is used, otherwise a highlighted box shows what that section
must cover. Rebuild after editing the text files:

    python report/build_report.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TEXT = ROOT / "report" / "text"
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "PathSnap_Technical_Report_PT-021.pdf"
REPO = "https://github.com/IAmKai1/MLhackathon-PES1UG24AM084-PES1UG24AM85-PES1UG24AM86-PES1UG24AM087"
TEAM = "PT-021"
MEMBERS = [("Dhareen G", "PES1UG24AM084"), ("Dhriti Kamani", "PES1UG24AM085"),
           ("Dhruv Rajagopalan", "PES1UG24AM086"), ("Dhruv Talavat", "PES1UG24AM087")]

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=6, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13.5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, textColor=colors.HexColor("#52514e"), alignment=TA_CENTER)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.2, leading=9)
TODO = ParagraphStyle("todo", parent=BODY, fontSize=8.5, leading=12, textColor=colors.HexColor("#7a4b00"),
                      backColor=colors.HexColor("#fff4d6"), borderColor=colors.HexColor("#eda100"),
                      borderWidth=0.6, borderPadding=6, spaceBefore=4, spaceAfter=8)
W = A4[0] - 36 * mm


def text(key, guidance):
    """Team-written text for a section, or a highlighted placeholder describing what to write."""
    f = TEXT / f"{key}.md"
    if f.exists() and f.read_text(encoding="utf-8").strip():
        out = []
        for para in re.split(r"\n\s*\n", f.read_text(encoding="utf-8").strip()):
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            para = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", para).replace("\n", " ")
            out.append(Paragraph(para, BODY))
            out.append(Spacer(1, 4))
        return out
    return [Paragraph(f"<b>TEAM TO WRITE</b> (report/text/{key}.md): {guidance}", TODO)]


def table(df, widths=None, fmt=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}f}")
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in df.columns]]
    data += [[Paragraph(str(v), CELL) for v in row] for row in df.to_numpy()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
    ]))
    return [t, Spacer(1, 6)]


def fig(path, caption, width=W):
    path = Path(path)
    if not path.exists():
        return [Paragraph(f"[missing figure: {path.relative_to(ROOT)}]", TODO)]
    img = Image(str(path))
    scale = width / img.imageWidth
    img.drawWidth, img.drawHeight = width, img.imageHeight * scale
    if img.drawHeight > 170 * mm:
        s = 170 * mm / img.drawHeight
        img.drawWidth, img.drawHeight = img.drawWidth * s, img.drawHeight * s
    return [KeepTogether([img, Paragraph(f"{caption}<br/><i>Source: {path.relative_to(ROOT).as_posix()}</i>", CAP)]),
            Spacer(1, 8)]


def src(*paths):
    return Paragraph("<i>Data: " + ", ".join(p for p in paths) + "</i>", CAP)


def main():
    comp = pd.read_csv(R / "models" / "model_comparison.csv")
    eda = pd.read_csv(R / "eda" / "summary_by_split.csv", index_col=0)
    inv = pd.read_csv(R / "eda" / "trip_inventory.csv")
    miss = pd.read_csv(R / "features" / "candidate_miss_analysis_train.csv")
    tuned = pd.read_csv(R / "bayes_opt" / "tuned_vs_untuned.csv")
    best = json.load(open(R / "bayes_opt" / "best_params.json"))
    trials = pd.read_csv(R / "bayes_opt" / "optuna_trials.csv")
    grid = pd.read_csv(R / "hmm" / "decoder_tuning_train_oof.csv")
    hst = pd.read_csv(R / "hmm" / "student_test_classifier_vs_hmm.csv")
    hsum = json.load(open(R / "hmm" / "student_test_summary.json"))
    ev = pd.read_csv(R / "evaluator" / "evaluator_classifier_vs_hmm.csv")
    esum = json.load(open(R / "evaluator" / "evaluator_summary.json"))
    diag = R / "diagnostics"
    ovf = pd.read_csv(diag / "train_vs_cv_vs_test.csv") if (diag / "train_vs_cv_vs_test.csv").exists() else None
    fixed = (pd.read_csv(diag / "hmm_fixed_vs_broke_student_test.csv")
             if (diag / "hmm_fixed_vs_broke_student_test.csv").exists() else None)

    s = []
    # ---------------- Title ----------------
    s += [Spacer(1, 40 * mm), Paragraph("PathSnap: Distributed Map-Matching &amp; Route Reconstruction",
                                        ParagraphStyle("t", parent=H1, fontSize=20, alignment=TA_CENTER)),
          Spacer(1, 6), Paragraph("UE24CS352A Machine Learning - Technical Report",
                                  ParagraphStyle("st", parent=BODY, fontSize=12, alignment=TA_CENTER)),
          Spacer(1, 14 * mm), Paragraph(f"<b>Team {TEAM}</b>", ParagraphStyle("tm", parent=BODY, fontSize=12,
                                                                           alignment=TA_CENTER)), Spacer(1, 6)]
    s += table(pd.DataFrame(MEMBERS, columns=["Member", "SRN"]), widths=[60 * mm, 50 * mm])
    s += [Spacer(1, 10 * mm), Paragraph(f"Repository: <link href='{REPO}' color='blue'>{REPO}</link>",
                                         ParagraphStyle("r", parent=CAP, fontSize=8.5)),
          Spacer(1, 4), Paragraph("All tables and figures are generated from files in the repository's results/ "
                                  "folder by report/build_report.py.", CAP), PageBreak()]

    # ---------------- 1. Overview ----------------
    s += [Paragraph("1. Overview", H1)]
    s += text("1_overview", "Problem in 3-4 sentences, the pipeline (candidates -> classifier -> HMM/Viterbi -> "
                            "FastAPI -> UI) and the headline results (student-test and evaluator accuracy).")
    s += table(pd.DataFrame([
        {"Stage": "Best classifier (HistGradientBoosting, tuned)", "Student-test": f"{tuned.test_top1_acc.iloc[1]:.4f}",
         "Evaluator": f"{esum['classifier_obs_acc_weighted']:.4f}"},
        {"Stage": "Classifier + HMM/Viterbi", "Student-test": f"{hsum['hmm_obs_acc_weighted']:.4f}",
         "Evaluator": f"{esum['hmm_obs_acc_weighted']:.4f}"},
        {"Stage": "Route continuity with HMM", "Student-test": f"{hsum['hmm_continuity_mean']:.4f}",
         "Evaluator": f"{esum['hmm_continuity_mean']:.4f}"}]), widths=[90 * mm, 40 * mm, 40 * mm])
    s += [Paragraph("Observation-level accuracy on trusted labels. Evaluator trips were used for inference only; "
                    "no model or decoder setting was changed after receiving them.", CAP)]

    # ---------------- 2. Dataset & EDA ----------------
    s += [PageBreak(), Paragraph("2. Dataset Understanding and Exploratory Data Analysis", H1),
          Paragraph("Partitions used: training = 80 trips, student-testing = 10 trips (evaluator trips not used "
                    "in any analysis below).", BODY), Spacer(1, 4),
          Paragraph("2.1 How the data was preprocessed", H2)]
    s += text("2_1_preprocessing", "Answer Q1: file meanings and relationships; invalid-row and timestamp checks; "
                                   "projection to metres and why; per-observation label alignment from the route; "
                                   "30 m trusted-label rule and how many observations it removes.")
    s += [Paragraph("Dataset summary by partition (missing-value and ordering analysis included)", H3)]
    s += table(eda.reset_index().rename(columns={"index": "statistic"}), widths=[60 * mm, 50 * mm, 50 * mm], fmt=3)
    s += fig(R / "eda" / "trip_distributions.png", "Observations per trip, largest time gap per trip, and share of "
                                                   "GPS points within 30 m of the reference route (train + student-test).")
    s += fig(R / "eda" / "all_trajectories.png", "All GPS trajectories (lon/lat) by partition.", width=W * 0.7)
    s += fig(R / "eda" / "sampling_intervals.png", "Sampling interval distribution (all partitions used for EDA).",
             width=W * 0.75)
    s += [Paragraph("2.2 Patterns observed", H2)]
    s += text("2_2_patterns", "Answer Q2: 1 Hz sampling, gaps, network size and components, trips 33/54 with "
                              "unreliable reference routes, what that means for evaluation.")
    poor = inv[inv.gps_within_30m_of_route < 0.8][["split", "trip", "n_obs", "path_km", "route_km",
                                                   "gps_within_30m_of_route", "gps_to_route_median_m"]]
    s += [Paragraph("Trips whose GPS disagrees with the supplied reference route", H3)]
    s += table(poor, fmt=2)
    s += fig(FIG / "map_eda_trip54_bad_reference.png", "Folium map, training trip 54: GPS (blue) follows the "
                                                       "reference route (green) only for the first part of the drive.")
    s += fig(FIG / "map_eda_trip03.png", "Folium map, training trip 3: GPS, nearby road network and reference route.")
    s += [Paragraph("2.3 Features extracted and why", H2)]
    s += text("2_3_features", "List the 19 features (pathsnap/features.py FEATURES) grouped by purpose and justify "
                              "each group. Explain candidate generation (KD-tree, R=60 m, K=10) using the miss-rate table.")
    s += [Paragraph("Candidate miss rate on training trips by radius and K", H3)]
    s += table(miss.pivot(index="radius_m", columns="K", values="candidate_miss_rate").reset_index())
    s += fig(R / "features" / "candidate_miss_analysis.png", "Candidate miss rate vs search radius (training trips).",
             width=W * 0.75)

    # ---------------- 3. Models ----------------
    s += [PageBreak(), Paragraph("3. Model Experimentation and Comparison", H1),
          Paragraph("Seven models from seven algorithm families were trained on the 80 training trips and evaluated "
                    "with 5-fold GroupKFold (groups = trips), then refit on all training trips and evaluated on the 10 "
                    "student-testing trips.", BODY),
          Paragraph("3.1 Overall model comparison", H2)]
    t31 = comp[["model", "family", "config", "cv_macro_f1", "test_top1_acc", "test_route_continuity",
                "test_macro_f1", "test_log_loss", "inference_ms_per_obs"]].rename(columns={
        "model": "Model", "family": "Family", "config": "Key configuration", "cv_macro_f1": "CV macro F1",
        "test_top1_acc": "Top-1 acc.", "test_route_continuity": "Route contin.",
        "test_macro_f1": "Test macro F1", "test_log_loss": "Log loss", "inference_ms_per_obs": "Infer. ms/obs"})
    t31["Model"] = t31["Model"].str.replace("HistGradientBoosting", "HistGradient Boosting").str.replace(
        "LogisticRegression", "Logistic Regression")
    t31["Family"] = t31["Family"].str.replace("Instance-based", "Instance based")
    s += table(t31, widths=[24 * mm, 20 * mm, 34 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm])
    s += [src("results/models/model_comparison.csv")]
    s += text("3_1_selection", "Which model you selected and why (primary metric, CV stability, probability quality "
                               "for the HMM, inference speed). Mention RandomForest's CV score is within one fold std.")
    if ovf is not None:
        s += [Paragraph("Training vs cross-validation vs student-test (overfitting check)", H3)]
        s += table(ovf[["model", "train_macro_f1", "cv_macro_f1", "test_macro_f1", "train_top1", "cv_top1",
                        "test_top1", "train_minus_cv_f1"]])
        s += [src("results/diagnostics/train_vs_cv_vs_test.csv")]
    s += [Paragraph("3.2 Per-model analysis", H2)]
    for _, row in comp.iterrows():
        m = row.model
        s += [Paragraph(f"3.2.{_ + 1} {m} ({row.family})", H3),
              Paragraph(f"Configuration: {row.config}. CV macro F1 {row.cv_macro_f1:.4f} (std {row.cv_macro_f1_std:.4f}); "
                        f"student-test Top-1 {row.test_top1_acc:.4f}, macro F1 {row.test_macro_f1:.4f}, "
                        f"log loss {row.test_log_loss:.4f}; training {row.train_time_s:.1f} s on {int(row.n_train_rows):,} rows.",
                        BODY)]
        s += text(f"3_2_{m}", "A1 why suitable; A2 strengths and limitations for this task; B5 important "
                              "hyperparameters and initial values; B6 loss / objective; C7-C8 student-test results "
                              "(above); C9 over/underfitting with evidence from the tables. Tie to your results.")
    s += [Paragraph("3.3 Model visualisations", H2)]
    s += fig(R / "models" / "cv_macro_f1.png", "Cross-validation macro F1 across all models (error bar = fold std).")
    s += fig(R / "models" / "student_test_top1_f1.png", "Student-testing Top-1 accuracy and macro F1.")
    s += fig(R / "models" / "student_test_log_loss.png", "Student-testing log loss.")
    s += fig(diag / "train_vs_cv_vs_test.png", "Training vs CV vs student-test macro F1.")
    s += fig(diag / "training_validation_loss.png", "Training and validation log loss for the iterative models "
                                                     "(fit on CV fold 0's training trips, validated on its held-out trips).")
    s += fig(R / "models" / "mlp_training_history.png", "MLP training loss and internal validation accuracy "
                                                        "(early-stopping run used in the comparison).")
    s += fig(R / "models" / "confusion_matrix_HistGradientBoosting.png",
             "Confusion matrix, HistGradientBoosting, student-test candidates.", width=W * 0.45)

    # ---------------- 4. Bayesian optimisation ----------------
    s += [PageBreak(), Paragraph("4. Bayesian Hyperparameter Optimisation (HistGradientBoosting)", H1)]
    bp = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in best["best_params"].items())
    s += table(pd.DataFrame([
        ["Hyperparameters searched", "learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization, max_features"],
        ["Ranges / distributions", "learning_rate 0.02-0.3 (log); max_iter 100-600 step 50; max_leaf_nodes 15-127; "
                                   "min_samples_leaf 10-300 (log); l2_regularization 1e-6-10 (log); max_features 0.4-1.0"],
        ["Objective", f"{best['objective']} ({best['cv']})"],
        ["Search strategy", best["sampler"]],
        ["Trials", f"{best['n_trials']} (trial 0 = untuned defaults, CV macro F1 {trials.value.iloc[0]:.4f})"],
        ["Best values", f"{bp} (CV macro F1 {best['best_cv_macro_f1']:.4f})"],
    ], columns=["Question", "Answer"]), widths=[40 * mm, W - 40 * mm])
    s += [Paragraph("Tuned vs untuned (same training data, folds and metrics)", H3)]
    s += table(tuned.rename(columns={"model": "Model"}))
    s += [src("results/bayes_opt/tuned_vs_untuned.csv", "results/bayes_opt/optuna_trials.csv")]
    s += text("4_bayes", "Q6: did tuning help and by how much; cost (training and inference time); why TPE rather "
                         "than grid search; what the parameter-importance plot shows.")
    s += fig(R / "bayes_opt" / "optimisation_history.png", "Optimisation history (3-fold group CV macro F1).")
    s += fig(R / "bayes_opt" / "param_importance.png", "Hyperparameter importance (fANOVA).", width=W * 0.7)

    # ---------------- 5. HMM ----------------
    s += [PageBreak(), Paragraph("5. HMM and Viterbi Route Reconstruction", H1)]
    s += text("5_hmm", "Q1 hidden states; Q2 emissions; Q3 transitions (formula); Q4 missing candidates and GPS gaps; "
                       "Q5 disconnected segments (chain restarts); Viterbi recurrence. Implementation: pathsnap/hmm.py.")
    s += [Paragraph("Decoder tuning on training trips (out-of-fold probabilities; only beta x weight matters)", H3)]
    s += table(grid.assign(product=grid.beta * grid.emission_weight).sort_values(["product", "beta"])[
        ["beta", "emission_weight", "product", "classifier_top1_acc", "hmm_top1_acc", "hmm_continuity"]])
    s += [Paragraph("Classifier only vs classifier + HMM, student-testing trips", H3)]
    s += table(hst[["trip", "n_obs", "chain_breaks", "clf_obs_acc", "hmm_obs_acc", "clf_continuity", "hmm_continuity",
                    "clf_route_f1", "hmm_route_f1"]])
    s += table(pd.DataFrame([
        {"Pipeline": "Classifier only", "Observation accuracy": hsum["classifier_obs_acc_weighted"],
         "Route continuity": hsum["classifier_continuity_mean"], "Route set F1": hsum["classifier_route_f1_mean"]},
        {"Pipeline": "Classifier + HMM", "Observation accuracy": hsum["hmm_obs_acc_weighted"],
         "Route continuity": hsum["hmm_continuity_mean"], "Route set F1": hsum["hmm_route_f1_mean"]}]))
    if fixed is not None:
        s += [Paragraph("Where the HMM helped and where it did not (student-test, scored observations)", H3)]
        s += table(fixed)
        s += [src("results/diagnostics/hmm_fixed_vs_broke_student_test.csv")]
    s += text("5_hmm_results", "Q6: did HMM improve the route; best and smallest gains; the 'hmm_broke' cases "
                               "(where it made a correct point wrong) and why that can happen.")
    s += fig(R / "hmm" / "classifier_vs_hmm.png", "Observation accuracy and route continuity per student-test trip.")
    s += fig(R / "hmm" / "decoder_tuning.png", "Decoder tuning on training trips.", width=W * 0.6)
    s += fig(ROOT / "report" / "route_comparison_00000077.png", "Student-test trip 77: classifier only vs "
                                                                "classifier + HMM against the reference route.")
    s += [Paragraph("Final evaluation on the evaluator-testing trips (inference only)", H3)]
    s += table(ev[["trip", "n_obs", "chain_breaks", "clf_obs_acc", "hmm_obs_acc", "clf_continuity", "hmm_continuity",
                   "hmm_route_f1"]])
    s += [Paragraph(f"Weighted observation accuracy: classifier only {esum['classifier_obs_acc_weighted']:.4f}, "
                    f"classifier + HMM {esum['hmm_obs_acc_weighted']:.4f}.", CAP)]

    # ---------------- 6. Folium ----------------
    s += [PageBreak(), Paragraph("6. Geographical Visualisation Using Folium", H1)]
    s += text("6_folium", "What the maps show (layers, colours, legend, layer control) and what they reveal. "
                          "Interactive maps: results/maps/*.html, results/evaluator/*.html; code: pathsnap/maps.py.")
    s += fig(FIG / "map_student_test_trip77.png", "Student-test trip 77: raw GPS, road network, reference route and "
                                                  "predicted route (layer control top right, legend bottom left).")
    s += fig(FIG / "map_evaluator_trip30.png", "Evaluator trip 30: raw GPS, road network, reference and predicted route.")

    # ---------------- 7. API / UI ----------------
    s += [PageBreak(), Paragraph("7. API, UI and Deployment", H1)]
    s += text("7_deployment", "Two-laptop architecture, endpoints (/trips, /predict), how ground truth stays "
                              "server-side, run commands (README).")
    for name, cap in [("ui_screenshot.png", "Working UI on Laptop 2 showing evaluator trip 15: predicted route and results."),
                      ("ui_server_laptop2.png", "Laptop 2 serving the UI (python -m http.server 5500)."),
                      ("api_logs_screenshot.png", "API logs on Laptop 1 (receiving laptop): GET /trips and POST /predict "
                                                  "requests arriving from Laptop 2 (10.8.121.59).")]:
        p = FIG / name
        s += fig(p, cap) if p.exists() else [Paragraph(f"<b>ADD SCREENSHOT</b>: save it as report/figures/{name} - {cap}",
                                                       TODO)]

    # ---------------- 8. Integrity ----------------
    s += [Paragraph("8. Academic Integrity and AI Use", H1)]
    s += text("8_ai_use", "Disclose AI assistance as the event requires, consistent with the README.")

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#52514e"))
        c.drawString(18 * mm, 10 * mm, f"PathSnap - Team {TEAM}")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
        c.restoreState()

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm, title=f"PathSnap Technical Report - Team {TEAM}").build(
        s, onFirstPage=footer, onLaterPages=footer)
    print("written", OUT)


if __name__ == "__main__":
    main()
