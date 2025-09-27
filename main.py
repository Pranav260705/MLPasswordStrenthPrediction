# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, numpy as np
from scipy import sparse
import uvicorn  # <- add this

app = FastAPI(title="Password Strength API")

artifact = joblib.load("password_strength_xgb.joblib")
model = artifact["model"]
vectorizer = artifact["vectorizer"]
label_map = artifact.get("label_mapping", {0: "weak", 1: "medium", 2: "strong"})

class PasswordIn(BaseModel):
    password: str = Field(min_length=1, max_length=128)

def extract_features(p: str):
    L = len(p)
    d = sum(c.isdigit() for c in p)
    lo = sum(c.islower() for c in p)
    up = sum(c.isupper() for c in p)
    sp = sum(not c.isalnum() for c in p)
    has_lo, has_up, has_d, has_sp = int(lo > 0), int(up > 0), int(d > 0), int(sp > 0)
    variety = has_lo + has_up + has_d + has_sp
    safe = max(L, 1)
    digit_ratio, special_ratio = d / safe, sp / safe
    has_consecutive = int(any(p[i] == p[i+1] == p[i+2] for i in range(max(0, L-2))))
    has_sequential = int(any(k in p.lower() for k in ["123","234","abc","bcd"]))
    unique_ratio = len(set(p)) / safe
    stat = np.array([[L, has_lo, has_up, has_d, has_sp, variety,
                      digit_ratio, special_ratio, has_consecutive, has_sequential, unique_ratio]],
                    dtype=np.float32)
    return sparse.csr_matrix(stat)

@app.post("/predict")
def predict(inp: PasswordIn):
    try:
        stat = extract_features(inp.password)
        tfidf = vectorizer.transform([inp.password])
        X = sparse.hstack([stat, tfidf], format="csr")
        idx = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()
        return {
            "index": idx,
            "label": label_map[idx],
            "probabilities": {"weak": proba[0], "medium": proba[1], "strong": proba[2]}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- start server automatically ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

