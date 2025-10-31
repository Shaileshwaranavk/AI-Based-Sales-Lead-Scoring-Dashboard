import csv, random

products = [
    ("SmartCRM Pro", "AI-powered CRM software for sales teams.", ["Lead tracking", "Email automation", "Pipeline analytics"]),
    ("FitTrack Wear", "Smart fitness tracker for health monitoring.", ["Heart rate", "Sleep tracking", "Water resistance"]),
    ("EcoClean Home", "Eco-friendly cleaning spray for multipurpose use.", ["Non-toxic", "Biodegradable", "Fresh scent"]),
    ("FoodieBox", "Subscription box delivering gourmet snacks monthly.", ["Global snacks", "Vegan options", "Personalized picks"]),
    ("StudyGenius", "AI learning assistant for students.", ["Note generation", "Quiz creation", "Progress tracking"]),
]

audiences = ["B2B sales teams", "Health-conscious individuals", "Eco-friendly families", "Snack lovers", "Students"]
strategies = ["LinkedIn ads", "Instagram influencers", "Email campaigns", "YouTube reviews", "Campus promotions"]
highlights = ["automation", "health tracking", "eco-safety", "variety", "AI personalization"]

rows = []
for _ in range(300):
    p = random.choice(products)
    rows.append({
        "product_name": p[0],
        "description": p[1],
        "features": ", ".join(p[2]),
        "target_audience": random.choice(audiences),
        "highlight_features": random.choice(highlights),
        "sales_strategy": random.choice(strategies),
    })

with open("synthetic_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("✅ Synthetic dataset generated: synthetic_dataset.csv")
