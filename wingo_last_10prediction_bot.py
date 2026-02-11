import requests
import pandas as pd
import joblib
import schedule
import time
import threading

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ======================
# CONFIG
# ======================
BOT_TOKEN = "7846952902:AAGX1okxNpy4c4i7LwCtIpf6AEeAV4ZaVjE"
CHAT_ID = "1150824427"
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts=1770652682680"

MODEL_FILE = "big_small_model.pkl"

# ======================
# DATA FETCH
# ======================
def fetch_history(limit=200):
    r = requests.get(API_URL, timeout=10)
    raw = r.json()["data"]["list"][:limit]

    df = pd.DataFrame(raw)
    df["number"] = df["number"].astype(int)
    df["result"] = (df["number"] >= 5).astype(int)  # 1=BIG, 0=SMALL

    df = df.sort_values("issueNumber")
    df.reset_index(drop=True, inplace=True)
    return df

# ======================
# FEATURE ENGINEERING
# ======================
def create_features(df):
    rows = []

    for i in range(5, len(df)):
        row = {
            "issueNumber": df.loc[i, "issueNumber"],
            "last_1": df.loc[i-1, "result"],
            "last_2": df.loc[i-2, "result"],
            "last_3": df.loc[i-3, "result"],
            "big_ratio_3": df.loc[i-3:i-1, "result"].mean(),
            "big_ratio_5": df.loc[i-5:i-1, "result"].mean(),
        }

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

# ======================
# TRAIN MODEL
# ======================
def train_model():
    df = fetch_history()
    feature_df = create_features(df)

    X = feature_df.drop(columns=["target", "issueNumber"])
    y = feature_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_FILE)
    return round(acc * 100, 2)

# ======================
# PREDICTION
# ======================

    df = fetch_history()
    model = joblib.load(MODEL_FILE)

    latest = {
        "last_1": df.iloc[-1]["result"],
        "last_2": df.iloc[-2]["result"],
        "last_3": df.iloc[-3]["result"],
        "big_ratio_3": df.iloc[-3:]["result"].mean(),
        "big_ratio_5": df.iloc[-5:]["result"].mean(),
        "streak": sum(
            df.iloc[-i]["result"] == df.iloc[-1]["result"]
            for i in range(1, 6)
        )
    }

    X = pd.DataFrame([latest])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max() * 100

    return {
        "issue": int(df.iloc[-1]["issueNumber"]) + 1,
        "result": "BIG 📈" if pred == 1 else "SMALL 📉",
        "confidence": round(prob, 2)
    }
def predict_next():
    df = fetch_history()
    model = joblib.load(MODEL_FILE)

    # ===== ML FEATURES =====
    features = {
        "last_1": df.iloc[-1]["result"],
        "last_2": df.iloc[-2]["result"],
        "last_3": df.iloc[-3]["result"],
        "big_ratio_3": df.iloc[-3:]["result"].mean(),
        "big_ratio_5": df.iloc[-5:]["result"].mean(),
        "streak": sum(
            df.iloc[-i]["result"] == df.iloc[-1]["result"]
            for i in range(1, 6)
        )
    }

    X = pd.DataFrame([features])
    ml_pred = model.predict(X)[0]
    ml_conf = model.predict_proba(X).max()

    # ===== LAST 10 ANALYSIS =====
    last10 = analyze_last_10(df)

    # ===== VOTING SYSTEM =====
    votes = []

    votes += [ml_pred] * 5            # ML weight
    votes += [last10["trend_pred"]] * 3

    if last10["streak"] >= 3:         # streak-break logic
        votes += [1 - df.iloc[-1]["result"]] * 2

    final_pred = 1 if votes.count(1) > votes.count(0) else 0

    confidence = round(
        (votes.count(final_pred) / len(votes)) * 100 * ml_conf,
        2
    )

    return {
        "issue": int(df.iloc[-1]["issueNumber"]) + 1,
        "result": "BIG 📈" if final_pred == 1 else "SMALL 📉",
        "confidence": min(confidence, 99),
        "details": {
            "ml": "BIG" if ml_pred else "SMALL",
            "last10_big": last10["big_count"],
            "last10_small": last10["small_count"],
            "streak": last10["streak"]
        }
    }

def analyze_last_10(df):
    last10 = df.tail(10)

    big_count = last10["result"].sum()
    small_count = 10 - big_count

    trend_pred = 1 if big_count < small_count else 0  # contrarian logic

    streak = 1
    for i in range(len(last10)-1, 0, -1):
        if last10.iloc[i]["result"] == last10.iloc[i-1]["result"]:
            streak += 1
        else:
            break

    return {
        "trend_pred": trend_pred,   # 1=BIG, 0=SMALL
        "big_count": big_count,
        "small_count": small_count,
        "streak": streak
    }


# ======================
# TELEGRAM COMMANDS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Big / Small Prediction Bot\n\n"
        "/predict - Get prediction\n"
        "/train - Retrain AI model"
    )

async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = predict_next()
        msg = (
    f"🤖 AI / ML Prediction\n\n"
    f"Issue: {res['issue']}\n"
    f"Result: {res['result']}\n"
    f"Confidence: {res['confidence']}%\n\n"
    f"📊 Last 10 Rounds\n"
    f"BIG: {res['details']['last10_big']} | "
    f"SMALL: {res['details']['last10_small']}\n"
    f"Streak: {res['details']['streak']}"
)

        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def train_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    acc = train_model()
    await update.message.reply_text(
        f"✅ Model retrained successfully\nAccuracy: {acc}%"
    )

# ======================
# AUTO PREDICTION (OPTIONAL)
# ======================
def auto_predict(bot):
    if not CHAT_ID:
        return
    res = predict_next()
    bot.send_message(
        chat_id=CHAT_ID,
        text = (
    f"🤖 AI / ML Prediction\n\n"
    f"Issue: {res['issue']}\n"
    f"Result: {res['result']}\n"
    f"Confidence: {res['confidence']}%\n\n"
    f"📊 Last 10 Rounds\n"
    f"BIG: {res['details']['last10_big']} | "
    f"SMALL: {res['details']['last10_small']}\n"
    f"Streak: {res['details']['streak']}"
)

    )

def scheduler_thread(bot):
    schedule.every(1).minutes.do(auto_predict, bot)
    while True:
        schedule.run_pending()
        time.sleep(1)

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print("🚀 Starting AI Telegram Bot...")

    try:
        acc = train_model()
        print(f"✅ Model trained | Accuracy: {acc}%")
    except:
        print("⚠️ Using existing model")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("train", train_cmd))

    if CHAT_ID:
        threading.Thread(
            target=scheduler_thread,
            args=(app.bot,),
            daemon=True
        ).start()

    app.run_polling()
