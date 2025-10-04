# 📊 MLB Rolling Plate Appearance Correlation Analysis

This project analyzes how predictive a player's recent plate appearances are of their next outcome using rolling stats like AVG, OBP, and SLG. It uses data from the past 10 MLB seasons via Baseball Savant's Statcast.

---

## ⚙️ Features

- Rolling window analysis for 1–250 previous plate appearances
- Calculates rolling AVG, OBP, SLG
- Measures correlation with the outcome of the **next** plate appearance
- Outputs:
  - CSV table of correlations
  - Three PNG plots:
    - Full range (1–250)
    - Zoomed (1–10)
    - Zoomed (11–250)

---

## 🧱 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
