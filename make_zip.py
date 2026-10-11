import pandas as pd, zipfile

df = pd.read_csv("data/Reviews.csv", usecols=["Text", "Score"])
neg = df[df.Score.isin([1, 2])].sample(12000, random_state=42)
pos = df[df.Score.isin([4, 5])].sample(12000, random_state=42)
pd.concat([neg, pos]).to_csv("small_reviews.csv", index=False)

with zipfile.ZipFile("project.zip", "w") as z:
    for f in ["main.py", "model.py", "utils.py"]:
        z.write(f"src/{f}", f"src/{f}")
    z.write("small_reviews.csv", "data/Reviews.csv")
print("done")