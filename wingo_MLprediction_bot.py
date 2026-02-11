import requests
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)

# ================= CONFIG =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
#BOT_TOKEN = "8320293848:AAH7jnQttKNgwoayL1bcQwpNoww1AtyogJ8"
CHAT_ID = "1150824427"
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts=1770652682680"
MODEL_FILE = "ai_model.pkl"
MIN_HISTORY = 10        # minimum data required

# ================= DATA =================
def fetch_history():
    r = requests.get(API_URL, timeout=15)
    data = r.json()["data"]["list"]

    df = pd.DataFrame(data)
    df["number"] = df["number"].astype(int)
    df["result"] = (df["number"] >= 5).astype(int)  # 1=BIG, 0=SMALL

    df = df.sort_values("issueNumber")
    df.reset_index(drop=True, inplace=True)
    return df

# ================= FEATURES =================
def build_features(df):
    rows = []

    if len(df) < 20:
        return pd.DataFrame()   # not enough data

    for i in range(10, len(df)):
        row = {}

        row["issueNumber"] = df.loc[i, "issueNumber"]

        row["last1"] = df.loc[i-1, "result"]
        row["last2"] = df.loc[i-2, "result"]
        row["last3"] = df.loc[i-3, "result"]

        row["last5_mean"] = df.loc[i-5:i-1, "result"].mean()
        row["last10_mean"] = df.loc[i-10:i-1, "result"].mean()

        row["volatility"] = df.loc[i-10:i-1, "result"].std()

        # streak
        streak = 1
        for j in range(i-1, 0, -1):
            if df.loc[j, "result"] == df.loc[j-1, "result"]:
                streak += 1
            else:
                break
        row["streak"] = streak

        row["target"] = df.loc[i, "result"]
        rows.append(row)

    return pd.DataFrame(rows)

# ================= TRAIN =================
def train_model(df):
    feature_df = build_features(df)

    if feature_df.empty:
        print("⚠️ Not enough data for ML training")
        return None

    X = feature_df.drop(columns=["target", "issueNumber"], errors="ignore")
    y = feature_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_FILE)
    return round(acc * 100, 2)

def fallback_prediction(df):
    last = df["result"].tolist()

    big = sum(last)
    small = len(last) - big

    # trend reversal logic
    if big > small:
        pred = 0
    else:
        pred = 1

    return {
        "issue": int(df.iloc[-1]["issueNumber"]) + 1,
        "prediction": "BIG 📈" if pred == 1 else "SMALL 📉",
        "confidence": round(abs(big - small) / len(last) * 100 + 50, 2),
        "mode": "Fallback (Low Data)"
    }


# ================= PREDICT =================
def predict_next(df):
    # if not enough data → fallback
    if len(df) < 20 or not os.path.exists(MODEL_FILE):
        return fallback_prediction(df)

    model = joblib.load(MODEL_FILE)

    latest = {
        "last1": df.iloc[-1]["result"],
        "last2": df.iloc[-2]["result"],
        "last3": df.iloc[-3]["result"],
        "last5_mean": df.iloc[-5:]["result"].mean(),
        "last10_mean": df.iloc[-10:]["result"].mean(),
        "volatility": df.iloc[-10:]["result"].std(),
        "streak": sum(
            df.iloc[-i]["result"] == df.iloc[-1]["result"]
            for i in range(1, min(8, len(df)))
        )
    }

    X = pd.DataFrame([latest])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max() * 100

    return {
        "issue": int(df.iloc[-1]["issueNumber"]) + 1,
        "prediction": "BIG 📈" if pred == 1 else "SMALL 📉",
        "confidence": round(prob, 2),
        "mode": "ML"
    }

# ================= TELEGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Big/Small Bot\n"
        "/predict – AI prediction\n"
        "/train – retrain model"
    )

async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history()
    res = predict_next(df)

    await update.message.reply_text(
        f"🔮 AI Prediction\n\n"
        f"Issue: {res['issue']}\n"
        f"Result: {res['prediction']}\n"
        f"Confidence: {res['confidence']}%"
    )

async def train_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history()
    acc = train_model(df)

    await update.message.reply_text(
        f"✅ Model trained on FULL HISTORY\nAccuracy: {acc}%"
    )

# ================= AUTO JOB =================
async def auto_predict(context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history()
    res = predict_next(df)

    await context.bot.send_message(
        chat_id=CHAT_ID,
        text=(
            f"⏰ AUTO AI PREDICTION\n\n"
            f"Issue: {res['issue']}\n"
            f"Result: {res['prediction']}\n"
            f"Confidence: {res['confidence']}%"
        )
    )

# ================= MAIN =================
if __name__ == "__main__":
    print("🚀 Starting AI Telegram Bot...")

    df = fetch_history()
    print(f"📊 History rows: {len(df)}")

    if len(df) < MIN_HISTORY:
        raise Exception("❌ Not enough history data")

    if not os.path.exists(MODEL_FILE):
       acc = train_model(df)
if acc:
    print(f"✅ Model trained | Accuracy: {acc}%")
else:
    print("⚠️ ML disabled – using fallback logic")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("train", train_cmd))

    app.job_queue.run_repeating(auto_predict, interval=60, first=10)

    app.run_polling()


