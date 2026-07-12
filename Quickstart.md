<div align="center">

# ⚡ Quickstart

### Zero to running dashboard in under 5 minutes

[![Time](https://img.shields.io/badge/Setup%20time-~5%20min-brightgreen)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Easy-blue)]()
[![Python](https://img.shields.io/badge/Requires-Python%203.10%2B-yellow?logo=python&logoColor=white)]()

</div>

<br/>

## 🗺️ The path

```mermaid
flowchart LR
    A[📥 Clone] --> B[📦 Install]
    B --> C[🔑 Get API key]
    C --> D[🔐 Set key]
    D --> E[▶️ Run]
    E --> F[🎉 Open localhost]

    style A fill:#1f2937,stroke:#60a5fa,color:#fff
    style B fill:#1f2937,stroke:#60a5fa,color:#fff
    style C fill:#1f2937,stroke:#fbbf24,color:#fff
    style D fill:#1f2937,stroke:#fbbf24,color:#fff
    style E fill:#1f2937,stroke:#34d399,color:#fff
    style F fill:#1f2937,stroke:#34d399,color:#fff
```

<br/>

## ① Clone the repo

```bash
git clone https://github.com/AkshPaliwal/FIFA-WORLD-CUP-PREDICTION-ORACLE-AI-.git
cd FIFA-WORLD-CUP-PREDICTION-ORACLE-AI-
```

<br/>

## ② Install dependencies

> 💡 **Tip:** use a virtual environment so this project's packages don't collide with anything else on your machine.

<details>
<summary><strong>Show me how to set up a virtual environment</strong> (click to expand)</summary>

<br/>

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

You'll know it worked if your terminal prompt now starts with `(.venv)`.

</details>

<br/>

```bash
pip install -r requirements.txt
```

<br/>

## ③ Get a free API key

Live results are pulled from **SportAPI7** on RapidAPI — it's free.

| Step | Action |
|---|---|
| 1 | Go to [rapidapi.com](https://rapidapi.com) and sign up / log in |
| 2 | Search **`SportAPI7`**, published by **rapidsportapi** |
| 3 | Click **Subscribe** → choose the **free/basic** plan |
| 4 | Open any endpoint's **Code Snippets** panel → copy the value after `x-rapidapi-key:` |

> ⚠️ **One key, many APIs.** RapidAPI ties one key to your whole account — you do **not** need a separate key per API. If you've subscribed to anything else on RapidAPI before, that same key already works here.

<br/>

## ④ Set the key

Pick **one** of these — Option A is faster to try right now, Option B means you never think about it again.

<table>
<tr>
<td width="50%" valign="top">

**Option A — quick, this session only**

```bash
export RAPIDAPI_KEY="paste-your-key-here"
```

Confirm it took:
```bash
echo $RAPIDAPI_KEY
```

⚠️ Only lasts until you close this terminal.

</td>
<td width="50%" valign="top">

**Option B — permanent, local**

Create `.streamlit/secrets.toml`:
```toml
RAPIDAPI_KEY = "paste-your-key-here"
```

✅ Also required for Streamlit Cloud deployment later.

</td>
</tr>
</table>

<br/>

## ⑤ Run it

```bash
streamlit run app.py
```

You should see:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

<br/>

## ⑥ Open it

<div align="center">

### 👉 [http://localhost:8501](http://localhost:8501) 👈

It should open automatically. If not, paste that link into your browser.

</div>

<br/>

---

<br/>

## 🚑 Troubleshooting

<details>
<summary><strong>❌ <code>RuntimeError: No RapidAPI key found</code></strong></summary>

<br/>

Your key isn't set in this terminal session. Run:
```bash
echo $RAPIDAPI_KEY
```
If it prints nothing, go back to Step ④. Remember: `export` only applies to the terminal window you ran it in — a new tab or a restarted terminal needs it set again (or use Option B above to skip this forever).

</details>

<details>
<summary><strong>❌ <code>ModuleNotFoundError: No module named 'streamlit'</code></strong> (or requests/pandas/etc.)</summary>

<br/>

A dependency didn't install. Re-run:
```bash
pip install -r requirements.txt
```
If you're using a virtual environment, make sure it's activated first (`source .venv/bin/activate`) — a common trap is installing into the wrong Python.

</details>

<details>
<summary><strong>❌ Page loads but Quarterfinal/Semifinal results never appear as "finished"</strong></summary>

<br/>

This is expected until:
1. The match has actually finished in real life, **and**
2. The app was open (even briefly) *before* kickoff, so it had a chance to freeze its prediction

If a match already finished before you ever ran the app, there's no frozen prediction to compare against — it'll silently fall back to whatever's in the manual `QUARTERFINAL_RESULTS_FALLBACK` list in `app.py`.

</details>

<details>
<summary><strong>❌ <code>404</code> or empty response from the API</strong></summary>

<br/>

Double check `WORLD_CATEGORY_ID` in `live_results.py` is still `1468` — this was confirmed against the SportAPI7 `/category/list` endpoint but IDs can occasionally shift on the provider's end. Re-verify with:
```bash
curl --request GET \
  --url 'https://sportapi7.p.rapidapi.com/api/v1/category/list' \
  --header "x-rapidapi-key: $RAPIDAPI_KEY" \
  --header 'x-rapidapi-host: sportapi7.p.rapidapi.com'
```

</details>

<br/>

## 🛑 Stopping the app

Click back into the terminal running it, press:

```
Ctrl + C
```

<br/>

<div align="center">

<sub>Full architecture, model details, and live-tracking design → see <a href="README.md">README.md</a></sub>

</div>
