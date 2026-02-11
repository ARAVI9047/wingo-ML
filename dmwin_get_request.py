import requests

url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts=1770647323997"
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
data = response.json()

print(data)

results = []

for item in data["data"]:
    results.append(item["color"])   # example field

print(results)

from bs4 import BeautifulSoup

url = "https://example.com/results"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

rows = soup.find_all("div", class_="result-row")

results = []
for row in rows:
    results.append(row.text.strip())

print(results)

import pandas as pd

df = pd.DataFrame(results, columns=["color"])
df.to_csv("data.csv", index=False)

from collections import Counter

counter = Counter(results)
prediction = counter.most_common(1)[0][0]

print("Next predicted color:", prediction)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encoded = le.fit_transform(results)

from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.arange(len(encoded)).reshape(-1, 1)
y = encoded

model = LogisticRegression()
model.fit(X, y)

next_index = [[len(encoded) + 1]]
predicted = model.predict(next_index)

print("Predicted:", le.inverse_transform(predicted))

import time

while True:
    # fetch data
    response = requests.get(url, headers=headers)
    data = response.json()

    # predict
    results = [x["color"] for x in data["data"]]
    prediction = Counter(results).most_common(1)[0][0]

    print("Prediction:", prediction)

    time.sleep(60)



