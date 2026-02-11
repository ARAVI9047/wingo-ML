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

BOT_TOKEN = "8277352183:AAGKpp16D0kOsVy7bkQR7_c6UBKWc4Ys4YE"
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
def predict_next():
    df = fetch_history(limit=10)

    results = df["result"].tolist()  # 1 = BIG, 0 = SMALL

    big_count = sum(results)
    small_count = len(results) - big_count

    big_pct = (big_count / 10) * 100
    small_pct = (small_count / 10) * 100

    last_result = results[-1]

    # =====================
    # DECISION LOGIC
    # =====================
    if big_pct > small_pct:
        prediction = "BIG 📈"
    elif small_pct > big_pct:
        prediction = "SMALL 📉"
    else:
        prediction = "BIG 📈" if last_result == 1 else "SMALL 📉"

    return {
        "issue": int(df.iloc[-1]["issueNumber"]) + 1,
        "prediction": prediction,
        "big_pct": round(big_pct, 2),
        "small_pct": round(small_pct, 2)
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
    f"🔮 AI Prediction (Last 10 Rounds)\n\n"
    f"Issue: {res['issue']}\n"
    f"Prediction: {res['prediction']}\n\n"
    f"📊 Percentages:\n"
    f"BIG: {res['big_percent']}%\n"
    f"SMALL: {res['small_percent']}%\n\n"
    f"Confidence: {res['confidence']}%"
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
        text  = (
    f"🔮 BIG / SMALL PREDICTION\n\n"
    f"Issue: {res['issue']}\n"
    f"Prediction: {res['prediction']}\n\n"
    f"📊 Last 10 Results Stats\n"
    f"BIG: {res['big_pct']}%\n"
    f"SMALL: {res['small_pct']}%"
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
