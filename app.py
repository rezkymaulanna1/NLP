import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

st.set_page_config(
    page_title="PIHPS Food Price Classifier",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2rem; font-weight: 700;
    color: #1a5276; text-align: center; padding: 1rem 0 0.2rem;
}
.sub-header {
    font-size: 0.95rem; color: #5d6d7e;
    text-align: center; margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    return re.sub(r"[^a-z\s]", "", str(text).lower()).strip()

@st.cache_data(show_spinner="Memuat dataset...")
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("komoditas_bawang_merah_2022_2026.csv")

    df = df.rename(columns={
        "Commodity_Name": "nama_komoditas",
        "Province_Name":  "provinsi",
        "Price":          "harga",
        "Date_Param":     "tanggal"
    })

    df = df.dropna(subset=["nama_komoditas", "provinsi", "harga"])
    df = df[df["harga"] > 0]
    df = df.drop_duplicates(subset=["tanggal", "nama_komoditas", "provinsi"])
    df["nama_komoditas"] = df["nama_komoditas"].apply(clean_text)
    df["provinsi"]       = df["provinsi"].apply(clean_text)

    mean_h = df["harga"].mean()
    std_h  = df["harga"].std()
    df["price_label"] = df["harga"].apply(
        lambda p: "Mahal" if p > mean_h + std_h
                  else ("Murah" if p < mean_h - std_h else "Normal")
    )
    df["text_input"] = df["nama_komoditas"] + " " + df["provinsi"]
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df, mean_h, std_h

@st.cache_resource(show_spinner="Training model...")
def train_models(_df):
    X = _df["text_input"]
    y = _df["price_label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    experiments = {
        "DT + BoW":    CountVectorizer(),
        "DT + N-Gram": CountVectorizer(ngram_range=(2, 2)),
        "DT + TF-IDF": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
    }
    results = {}
    for name, vec in experiments.items():
        Xtr = vec.fit_transform(X_tr)
        Xte = vec.transform(X_te)
        clf = DecisionTreeClassifier(criterion="gini", random_state=42)
        clf.fit(Xtr, y_tr)
        yp  = clf.predict(Xte)
        results[name] = {
            "vec": vec, "clf": clf,
            "accuracy":  accuracy_score(y_te, yp),
            "precision": precision_score(y_te, yp, average="weighted", zero_division=0),
            "recall":    recall_score(y_te, yp, average="weighted", zero_division=0),
            "f1":        f1_score(y_te, yp, average="weighted", zero_division=0),
            "report":    classification_report(y_te, yp, output_dict=True),
            "cm":        confusion_matrix(y_te, yp, labels=["Mahal", "Normal", "Murah"]),
        }
    return results

COLOR = {"Mahal": "#e74c3c", "Normal": "#f39c12", "Murah": "#27ae60"}

with st.sidebar:
    st.title("🧅 PIHPS NLP")
    st.caption("Klasifikasi Status Harga Pangan")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV lain (opsional)", type=["csv"])
    st.markdown("---")
    page = st.radio("Navigasi", [
        "🏠 Overview",
        "📊 Eksplorasi Data",
        "🤖 Training & Evaluasi",
        "🔍 Uji Model",
    ])

df, mean_h, std_h = load_data(uploaded)
results = train_models(df)
PROVINCES = sorted(df["provinsi"].unique().tolist())

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🧅 PIHPS Food Price Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistem Klasifikasi Status Harga Komoditas Pangan · NLP Mini Project</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Data",      f"{len(df):,}")
    c2.metric("Komoditas",       df["nama_komoditas"].nunique())
    c3.metric("Provinsi",        df["provinsi"].nunique())
    c4.metric("Periode",         f"{df['tanggal'].dt.year.min()}–{df['tanggal'].dt.year.max()}")
    c5.metric("Rata-rata Harga", f"Rp {mean_h:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Business Objective")
        st.info(
            "Sistem ini mengklasifikasikan status harga **Bawang Merah** menjadi "
            "**Mahal**, **Normal**, atau **Murah** berdasarkan input teks "
            "`nama_komoditas + provinsi` menggunakan algoritma **Decision Tree**."
        )
        st.subheader("🏷️ Aturan Labeling")
        st.markdown(f"""
        | Label | Kondisi |
        |-------|---------|
        | 🔴 Mahal  | Harga > Rp {mean_h + std_h:,.0f} |
        | 🟡 Normal | Rp {mean_h - std_h:,.0f} – Rp {mean_h + std_h:,.0f} |
        | 🟢 Murah  | Harga < Rp {mean_h - std_h:,.0f} |
        """)
    with col2:
        st.subheader("🔬 Pipeline NLP")
        st.markdown("""
        | Tahap | Proses |
        |-------|--------|
        | Input       | `nama_komoditas` + `provinsi` |
        | Preprocessing | Lowercase, hapus non-alfanumerik |
        | Labeling    | mean ± 1 std keseluruhan dataset |
        | Fitur       | BoW / N-Gram / TF-IDF |
        | Model       | Decision Tree (Gini, seed=42) |
        | Split       | 80% Train / 20% Test |
        """)

# ── Eksplorasi Data ───────────────────────────────────────────────────────────
elif page == "📊 Eksplorasi Data":
    st.header("📊 Eksplorasi Data")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribusi Label", "Tren Harga", "Per Provinsi", "Sample Data"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        label_counts = df["price_label"].value_counts()
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(label_counts.index, label_counts.values,
                          color=[COLOR[k] for k in label_counts.index],
                          edgecolor="white", width=0.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 50,
                        f"{bar.get_height():,}", ha="center",
                        fontsize=10, fontweight="bold")
            ax.set_title("Jumlah Per Kelas Label", fontweight="bold")
            ax.set_xlabel("Status Harga"); ax.set_ylabel("Jumlah")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(label_counts.values,
                    labels=label_counts.index,
                    colors=[COLOR[k] for k in label_counts.index],
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 2})
            ax2.set_title("Proporsi Label", fontweight="bold")
            st.pyplot(fig2)

    with tab2:
        df_m = df.copy()
        df_m["bulan"] = df_m["tanggal"].dt.to_period("M").dt.to_timestamp()
        monthly_avg = df_m.groupby("bulan")["harga"].mean().reset_index()
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(monthly_avg["bulan"], monthly_avg["harga"],
                 color="#2e86c1", linewidth=2)
        ax3.axhline(mean_h + std_h, color="#e74c3c", linestyle="--",
                    alpha=0.7, label="Threshold Mahal")
        ax3.axhline(mean_h - std_h, color="#27ae60", linestyle="--",
                    alpha=0.7, label="Threshold Murah")
        ax3.fill_between(monthly_avg["bulan"],
                         mean_h - std_h, mean_h + std_h,
                         alpha=0.1, color="#f39c12", label="Zona Normal")
        ax3.set_title("Tren Harga Rata-rata Bulanan Bawang Merah (2022–2026)",
                      fontweight="bold")
        ax3.set_xlabel("Bulan"); ax3.set_ylabel("Harga (Rp)")
        ax3.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
        ax3.legend(); ax3.grid(alpha=0.3)
        st.pyplot(fig3)

    with tab3:
        prov_avg = df.groupby("provinsi")["harga"].mean().sort_values(ascending=False)
        bar_colors = [
            COLOR["Mahal"] if v > mean_h + std_h
            else (COLOR["Murah"] if v < mean_h - std_h else COLOR["Normal"])
            for v in prov_avg.values
        ]
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        ax4.barh(prov_avg.index[::-1], prov_avg.values[::-1],
                 color=bar_colors[::-1])
        ax4.axvline(mean_h + std_h, color="#e74c3c", linestyle="--",
                    alpha=0.8, label="Threshold Mahal")
        ax4.axvline(mean_h - std_h, color="#27ae60", linestyle="--",
                    alpha=0.8, label="Threshold Murah")
        ax4.set_title("Rata-rata Harga per Provinsi", fontweight="bold")
        ax4.set_xlabel("Harga (Rp)")
        ax4.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
        ax4.legend(); ax4.grid(axis="x", alpha=0.3)
        st.pyplot(fig4)

    with tab4:
        st.dataframe(
            df[["tanggal", "nama_komoditas", "provinsi", "harga", "price_label"]]
            .sort_values("tanggal", ascending=False).head(500),
            use_container_width=True, height=400
        )

# ── Training & Evaluasi ───────────────────────────────────────────────────────
elif page == "🤖 Training & Evaluasi":
    st.header("🤖 Training & Evaluasi Model")

    summary = pd.DataFrame([
        {"Eksperimen": k,
         "Accuracy":  round(v["accuracy"],  4),
         "Precision": round(v["precision"], 4),
         "Recall":    round(v["recall"],    4),
         "F1-Score":  round(v["f1"],        4)}
        for k, v in results.items()
    ])
    st.subheader("Perbandingan Performa Keseluruhan")
    st.dataframe(summary.set_index("Eksperimen"), use_container_width=True)

    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_k = ["accuracy", "precision", "recall", "f1"]
    x     = np.arange(len(metrics))
    width = 0.25
    clrs  = ["#2e86c1", "#e74c3c", "#27ae60"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (exp, res) in enumerate(results.items()):
        vals = [res[mk] for mk in metric_k]
        bars = ax.bar(x + i * width, vals, width, label=exp,
                      color=clrs[i], zorder=3, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Perbandingan Metrik (3 Eksperimen)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3, zorder=0)
    st.pyplot(fig)

    st.subheader("Detail Per Eksperimen")
    exp_sel = st.selectbox("Pilih Eksperimen:", list(results.keys()))
    r = results[exp_sel]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Classification Report**")
        st.dataframe(
            pd.DataFrame(r["report"]).transpose().round(3),
            use_container_width=True
        )
    with col2:
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Mahal", "Normal", "Murah"],
                    yticklabels=["Mahal", "Normal", "Murah"],
                    ax=ax_cm, linewidths=0.5, annot_kws={"size": 13})
        ax_cm.set_title(f"Confusion Matrix — {exp_sel}", fontweight="bold")
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

# ── Uji Model ─────────────────────────────────────────────────────────────────
elif page == "🔍 Uji Model":
    st.header("🔍 Uji Model — Prediksi Status Harga")

    col1, col2, col3 = st.columns(3)

    with col1:
        model_sel = st.selectbox("Model:", list(results.keys()))

    with col2:
        komoditas = st.text_input(
            "Nama Komoditas:",
            value="bawang merah"
        )

    with col3:
        provinsi = st.selectbox(
            "Provinsi:",
            PROVINCES
        )

    # =========================================================
    # SINGLE PREDICTION
    # =========================================================

    if st.button(
        "🔮 Prediksi",
        type="primary",
        use_container_width=True
    ):

        text = clean_text(f"{komoditas} {provinsi}")

        vec = results[model_sel]["vec"]
        clf = results[model_sel]["clf"]

        pred = clf.predict(vec.transform([text]))[0]
        proba = clf.predict_proba(vec.transform([text]))[0]

        st.markdown(
            f"""
            <div style="
                background:{COLOR[pred]};
                color:white;
                padding:1.5rem;
                border-radius:12px;
                text-align:center;
                margin:1rem 0
            ">
                <h2 style="margin:0">
                    Status Harga: {pred}
                </h2>

                <p style="margin:0.3rem 0 0">
                    {komoditas.title()} · {provinsi.title()}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        prob_df = pd.DataFrame({
            "Kelas": clf.classes_,
            "Probabilitas": proba
        }).sort_values(
            "Probabilitas",
            ascending=True
        )

        fig_p, ax_p = plt.subplots(figsize=(6, 3))

        ax_p.barh(
            prob_df["Kelas"],
            prob_df["Probabilitas"],
            color=[COLOR[k] for k in prob_df["Kelas"]]
        )

        for i, v in enumerate(prob_df["Probabilitas"]):
            ax_p.text(
                v + 0.005,
                i,
                f"{v:.3f}",
                va="center",
                fontsize=11,
                fontweight="bold"
            )

        ax_p.set_xlim(0, 1.15)
        ax_p.set_title(
            "Probabilitas per Kelas",
            fontweight="bold"
        )

        ax_p.grid(axis="x", alpha=0.3)

        st.pyplot(fig_p)

    # =========================================================
    # BATCH PREDICTION
    # =========================================================

    st.markdown("---")

    st.subheader("📂 Prediksi Batch (Upload CSV)")

    st.caption("""
    CSV minimal harus memiliki kolom:
    - nama_komoditas
    - provinsi

    Opsional:
    - label_asli (untuk evaluasi model)
    """)

    batch_file = st.file_uploader(
        "Upload CSV batch",
        type=["csv"],
        key="batch"
    )

    if batch_file:

        bdf = pd.read_csv(batch_file)

        # =====================================================
        # VALIDASI KOLOM
        # =====================================================

        if {"nama_komoditas", "provinsi"}.issubset(bdf.columns):

            # =================================================
            # PREPROCESSING
            # =================================================

            bdf["text_input"] = bdf.apply(
                lambda r: clean_text(
                    f"{r['nama_komoditas']} {r['provinsi']}"
                ),
                axis=1
            )

            vec = results[model_sel]["vec"]
            clf = results[model_sel]["clf"]

            X_batch = vec.transform(
                bdf["text_input"]
            )

            # =================================================
            # PREDIKSI
            # =================================================

            bdf["prediksi"] = clf.predict(X_batch)

            # =================================================
            # PROBABILITAS / CONFIDENCE
            # =================================================

            proba = clf.predict_proba(X_batch)

            bdf["confidence"] = np.max(
                proba,
                axis=1
            )

            # =================================================
            # TAMPILKAN HASIL
            # =================================================

            st.subheader("📋 Hasil Prediksi Batch")

            st.dataframe(
                bdf,
                use_container_width=True
            )

            # =================================================
            # EVALUASI MODEL
            # =================================================

            if "label_asli" in bdf.columns:

                st.markdown("---")

                st.subheader(
                    "📊 Evaluasi Batch Prediction"
                )

                y_true = bdf["label_asli"]
                y_pred = bdf["prediksi"]

                # =============================================
                # METRICS
                # =============================================

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Accuracy",
                    f"{accuracy_score(y_true, y_pred):.3f}"
                )

                col2.metric(
                    "Precision",
                    f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.3f}"
                )

                col3.metric(
                    "Recall",
                    f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.3f}"
                )

                col4.metric(
                    "F1-Score",
                    f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.3f}"
                )

                # =============================================
                # CLASSIFICATION REPORT
                # =============================================

                st.write("### 📋 Classification Report")

                report_df = pd.DataFrame(
                    classification_report(
                        y_true,
                        y_pred,
                        output_dict=True,
                        zero_division=0
                    )
                ).transpose()

                st.dataframe(
                    report_df.round(3),
                    use_container_width=True
                )

                # =============================================
                # CONFUSION MATRIX
                # =============================================

                st.write("### 🔥 Confusion Matrix")

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=["Mahal", "Normal", "Murah"]
                )

                fig_cm, ax_cm = plt.subplots(
                    figsize=(6, 5)
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=[
                        "Mahal",
                        "Normal",
                        "Murah"
                    ],
                    yticklabels=[
                        "Mahal",
                        "Normal",
                        "Murah"
                    ],
                    linewidths=0.5,
                    annot_kws={"size": 13},
                    ax=ax_cm
                )

                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")

                ax_cm.set_title(
                    "Confusion Matrix Batch",
                    fontweight="bold"
                )

                st.pyplot(fig_cm)

                # =============================================
                # ERROR ANALYSIS
                # =============================================

                st.write("### ❌ Error Analysis")

                error_df = bdf[
                    y_true != y_pred
                ]

                st.metric(
                    "Jumlah Salah Prediksi",
                    len(error_df)
                )

                if len(error_df) > 0:

                    st.dataframe(
                        error_df[
                            [
                                "nama_komoditas",
                                "provinsi",
                                "label_asli",
                                "prediksi",
                                "confidence"
                            ]
                        ],
                        use_container_width=True
                    )

                else:
                    st.success(
                        "Tidak ada kesalahan prediksi 🎉"
                    )

            # =================================================
            # VISUALISASI HASIL PREDIKSI
            # =================================================

            st.markdown("---")

            st.subheader(
                "📈 Visualisasi Hasil Prediksi"
            )

            col1, col2 = st.columns(2)

            # =============================================
            # PIE CHART PROPORSI PREDIKSI
            # =============================================

            with col1:

                pred_counts = (
                    bdf["prediksi"]
                    .value_counts()
                )

                fig1, ax1 = plt.subplots(
                    figsize=(5, 4)
                )

                ax1.pie(
                    pred_counts.values,
                    labels=pred_counts.index,
                    autopct="%1.1f%%",
                    colors=[
                        COLOR[k]
                        for k in pred_counts.index
                    ],
                    startangle=90,
                    wedgeprops={
                        "edgecolor": "white",
                        "linewidth": 2
                    }
                )

                ax1.set_title(
                    "Proporsi Prediksi"
                )

                st.pyplot(fig1)

            # =============================================
            # DISTRIBUSI CONFIDENCE
            # =============================================

            with col2:

                fig2, ax2 = plt.subplots(
                    figsize=(6, 4)
                )

                ax2.hist(
                    bdf["confidence"],
                    bins=10,
                    edgecolor="black"
                )

                ax2.set_title(
                    "Distribusi Keyakinan Model"
                )

                ax2.set_xlabel(
                    "Confidence Score"
                )

                ax2.set_ylabel(
                    "Jumlah Data"
                )

                ax2.grid(alpha=0.3)

                st.pyplot(fig2)

            # =================================================
            # DOWNLOAD BUTTON
            # =================================================

            st.markdown("---")

            st.download_button(
                "⬇️ Download Hasil Prediksi",
                data=bdf.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_batch.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.error("""
            CSV harus memiliki kolom:
            - nama_komoditas
            - provinsi

            Opsional:
            - label_asli
            """)
