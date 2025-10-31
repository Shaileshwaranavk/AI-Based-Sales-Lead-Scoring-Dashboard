import pandas as pd
import numpy as np
import requests, json, time, re, os
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from io import StringIO
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# === FUNCTION: MACHINE LEARNING TRAINING & PREDICTION ====
# =========================================================
def train_and_predict_with_ml(df_train, df_new, product_name, description, features, top_n=20):
    """
    Train a RandomForest model on labeled leads and predict conversion scores.
    """
    result = {"status": "", "r2_score": None, "top_leads": [], "lime_explanation": []}
    target_col = "Conversion_Rate"

    if target_col not in df_train.columns:
        result["status"] = f"❌ Target column '{target_col}' missing in labeled dataset!"
        return result

    if "Product_Interest_Level" in df_train.columns:
        df_train = df_train.drop(columns=["Product_Interest_Level"])

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    # Filter relevant rows
    keywords = [kw.lower() for kw in [product_name] + description.split() + features.split(",")]
    keyword_hits = df_train["Industry"].astype(str).apply(
        lambda x: any(kw in x.lower() for kw in keywords if isinstance(x, str))
    )
    filtered_df = df_train[keyword_hits]

    X_train_data = filtered_df.drop(columns=[target_col], errors="ignore") if len(filtered_df) >= 10 else X
    y_train_data = filtered_df[target_col] if len(filtered_df) >= 10 else y

    # Align columns
    missing_cols = [c for c in X_train_data.columns if c not in df_new.columns]
    extra_cols = [c for c in df_new.columns if c not in X_train_data.columns]

    for col in missing_cols:
        df_new[col] = 0
    if extra_cols:
        df_new = df_new.drop(columns=extra_cols)

    # Label Encoding
    for col in X_train_data.select_dtypes(include=["object"]).columns:
        if col not in df_new.columns:
            df_new[col] = "Unknown"
        le = LabelEncoder()
        all_values = list(X_train_data[col].astype(str)) + list(df_new[col].astype(str))
        le.fit(all_values)
        X_train_data[col] = le.transform(X_train_data[col].astype(str))
        df_new[col] = le.transform(df_new[col].astype(str))

    # Train & predict
    X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    result["r2_score"] = round(r2, 3)

    predictions = model.predict(df_new[X_train_data.columns].to_numpy())
    df_new["Predicted_Conversion_Score_ML"] = predictions
    top_leads = df_new.sort_values(by="Predicted_Conversion_Score_ML", ascending=False).head(top_n)
    result["top_leads"] = top_leads[["Lead_ID", "Predicted_Conversion_Score_ML"]].to_dict(orient="records")

    # LIME explainability
    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train_data),
            feature_names=X_train_data.columns.tolist(),
            mode="regression"
        )
        instance = np.array(top_leads[X_train_data.columns].iloc[0])
        exp = explainer.explain_instance(instance, model.predict, num_features=8)
        explanations = []
        for feature, weight in exp.as_list():
            direction = "increases" if weight > 0 else "decreases"
            explanations.append({
                "feature": feature,
                "direction": direction,
                "weight": round(abs(weight), 3)
            })
        result["lime_explanation"] = explanations
    except Exception as e:
        result["lime_explanation"] = [{"error": str(e)}]

    result["status"] = "✅ ML prediction successful"
    return result


# =========================================================
# === FUNCTION: AI-BASED LEAD SCORING (Groq LLM) ==========
# =========================================================
def analyze_with_ai(product_name, description, features, df, top_n=20):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GROQ_API_KEY in .env")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    all_results = []

    BATCH_SIZE = 30
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(num_batches):
        batch = df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        leads_text = "\n".join(
            f"Lead_ID={row.get('Lead_ID','')}, Source={row.get('Lead_Source','')}, "
            f"Country={row.get('Country','')}, Revenue={row.get('Annual_Revenue','')}, "
            f"Employees={row.get('Employee_Count','')}, Interactions={row.get('Website_Visits','')}/"
            f"{row.get('Email_Opens','')}"
            for _, row in batch.iterrows()
        )

        prompt = f"""
        You are an expert AI sales analyst.
        Product: {product_name}
        Description: {description}
        Features: {features}

        For each lead below, predict conversion likelihood (0–100)
        and give a short 1–2 line explanation.

        Return CSV with:
        Lead_ID,Predicted_Conversion_Score,Explanation

        Leads:
        {leads_text}
        """

        data = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}]}

        while True:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            try:
                result_json = response.json()
            except Exception:
                time.sleep(5)
                continue

            if "error" in result_json:
                time.sleep(10)
                continue

            if "choices" in result_json:
                content = result_json["choices"][0]["message"]["content"]
                csv_block = re.search(r"(?s)Lead_ID.*", content)
                if not csv_block:
                    break
                csv_text = csv_block.group(0)
                csv_text = re.sub(r"```csv|```", "", csv_text).strip()
                if not csv_text.lower().startswith("lead_id"):
                    csv_text = "Lead_ID,Predicted_Conversion_Score,Explanation\n" + csv_text
                all_results.append(csv_text)
                break

    combined_csv = "\n".join(all_results)
    df_ai = pd.read_csv(StringIO(combined_csv), on_bad_lines='skip')

    df_ai["Predicted_Conversion_Score_AI"] = (
        df_ai["Predicted_Conversion_Score"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    df_top = df_ai.sort_values(by="Predicted_Conversion_Score_AI", ascending=False).head(top_n)
    return {
        "status": "✅ AI scoring complete",
        "top_leads": df_top[["Lead_ID", "Predicted_Conversion_Score_AI", "Explanation"]].to_dict(orient="records"),
        "summary": f"Analyzed {len(df)} leads using AI. Top {top_n} leads predicted."
    }


# =========================================================
# === MAIN HYBRID PIPELINE (for Django endpoint) ==========
# =========================================================
def run_hybrid_pipeline(new_data_path, labeled_data_path, product_name, description, features, top_n=20):
    df_new = pd.read_csv(new_data_path)
    df_ml_result = None

    if labeled_data_path:
        df_train = pd.read_csv(labeled_data_path)
        df_ml_result = train_and_predict_with_ml(df_train, df_new, product_name, description, features, top_n)

    df_ai_result = analyze_with_ai(product_name, description, features, df_new, top_n)

    result = {"ml": df_ml_result, "ai": df_ai_result, "hybrid": None}

    # Combine ML + AI (Hybrid)
    if df_ml_result and df_ml_result.get("top_leads"):
        df_ml_top = pd.DataFrame(df_ml_result["top_leads"])
        df_ai_top = pd.DataFrame(df_ai_result["top_leads"])
        df_ai_top["Lead_ID"] = df_ai_top["Lead_ID"].astype(str)
        df_ml_top["Lead_ID"] = df_ml_top["Lead_ID"].astype(str)

        merged = pd.merge(df_ai_top, df_ml_top, on="Lead_ID", how="inner")
        merged["Hybrid_Score"] = (
            0.6 * merged["Predicted_Conversion_Score_ML"] +
            0.4 * merged["Predicted_Conversion_Score_AI"]
        )
        merged = merged.sort_values(by="Hybrid_Score", ascending=False).head(top_n)
        result["hybrid"] = merged.to_dict(orient="records")

    return result
