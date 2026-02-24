import requests, time

key = "34d03d1d1382e36010bdb817d2512a4bfa5585f3"
headers = {"Content-Type": "application/json", "Authorization": f"Token {key}"}

# Test 1: Daily endpoint
print("Test 1: Daily endpoint...")
r = requests.get("https://api.tiingo.com/tiingo/daily/tsla/prices",
                  headers=headers,
                  params={"startDate": "2025-02-01", "endDate": "2025-02-07"},
                  timeout=15)
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    print(f"  Rows: {len(d)}, keys: {list(d[0].keys()) if d else 'empty'}")
elif r.status_code == 429:
    print(f"  Rate limited: {r.text[:200]}")
else:
    print(f"  Error: {r.text[:300]}")

time.sleep(2)

# Test 2: IEX intraday
print("Test 2: IEX intraday  (1 day only)...")
r2 = requests.get("https://api.tiingo.com/iex/tsla/prices",
                   headers=headers,
                   params={"startDate": "2025-02-20", "endDate": "2025-02-21",
                           "resampleFreq": "5min"},
                   timeout=15)
print(f"  Status: {r2.status_code}")
if r2.status_code == 200:
    d2 = r2.json()
    print(f"  Rows: {len(d2)}")
    if d2:
        print(f"  Keys: {list(d2[0].keys())}")
        print(f"  Sample: {d2[0]}")
else:
    print(f"  Response: {r2.text[:300]}")

time.sleep(2)

# Test 3: Crypto
print("Test 3: Crypto endpoint...")
r3 = requests.get("https://api.tiingo.com/tiingo/crypto/prices",
                   headers=headers,
                   params={"tickers": "solusd",
                           "startDate": "2025-02-20",
                           "endDate": "2025-02-21",
                           "resampleFreq": "5min"},
                   timeout=15)
print(f"  Status: {r3.status_code}")
if r3.status_code == 200:
    d3 = r3.json()
    print(f"  Entries: {len(d3)}")
    if d3 and "priceData" in d3[0]:
        pd3 = d3[0]["priceData"]
        print(f"  Price rows: {len(pd3)}, keys: {list(pd3[0].keys()) if pd3 else 'empty'}")
        if pd3:
            print(f"  Sample: {pd3[0]}")
else:
    print(f"  Response: {r3.text[:300]}")

time.sleep(2)

# Test 4: How far back does IEX go?
print("Test 4: IEX historical depth (2021 data)...")
r4 = requests.get("https://api.tiingo.com/iex/tsla/prices",
                   headers=headers,
                   params={"startDate": "2021-06-01", "endDate": "2021-06-02",
                           "resampleFreq": "5min"},
                   timeout=15)
print(f"  Status: {r4.status_code}")
if r4.status_code == 200:
    d4 = r4.json()
    print(f"  Rows: {len(d4)}")
    if d4:
        print(f"  First: {d4[0].get('date', 'no date')}")
else:
    print(f"  Response: {r4.text[:300]}")
