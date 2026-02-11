
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = "8277352183:AAGKpp16D0kOsVy7bkQR7_c6UBKWc4Ys4YE"
CHAT_ID = "1150824427"
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts=1770652682680"

def get_prediction():
    response = requests.get(API_URL, timeout=10)
    data = response.json()["data"]["list"][:10]  # last 10 rounds

    big = 0
    small = 0

    for item in data:
        number = int(item["number"])
        if number >= 5:
            big += 1
        else:
            small += 1

    if big > small:
        return "SMALL 📉"
    else:
        return "BIG 📈"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Big / Small Prediction Bot\n\nUse /predict"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prediction = get_prediction()
    await update.message.reply_text(
        f"📊 Next Prediction:\n\n➡️ {prediction}"
    )

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))
app.run_polling()


import schedule
import time
from telegram import Bot

CHAT_ID = "YOUR_CHANNEL_OR_USER_ID"
bot = Bot(BOT_TOKEN)

def auto_prediction():
    prediction = get_prediction()
    bot.send_message(
        chat_id=CHAT_ID,
        text=f"⏰ Auto Prediction\n➡️ {prediction}"
    )

schedule.every(1).minutes.do(auto_prediction)

while True:
    schedule.run_pending()
    time.sleep(1)
