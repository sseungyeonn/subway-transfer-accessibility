"""논문 표 3·4·6 재현 + 표본 크기 한계 점검.

실행: python scripts/analysis.py
필요 패키지: pandas, numpy, statsmodels, scikit-learn
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "evaluation_scores.csv")
X = df.drop(columns=["route", "measured_m"]).astype(float)
y = df["measured_m"]
n, p = X.shape

# 1. 원변수 다중회귀: 표본(10) < 계수(12+1)라 해가 유일하지 않다
Xc = sm.add_constant(X)
print(f"[1] 표본 {n}개, 변수 {p}개, 설계행렬 rank = {np.linalg.matrix_rank(Xc.values)} (필요: {p + 1})")
vif = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
print("    VIF:", "모두 무한대에 가까움" if min(vif) > 1e6 else np.round(vif, 1))

# 2. 표준화 후 PCA (논문 표 4)
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
ratio = pca.explained_variance_ratio_
print("\n[2] 주성분 설명력 (PC1~4):", np.round(ratio[:4], 4), "누적", round(ratio[:4].sum(), 4))
loadings = pd.DataFrame(pca.components_[:4].T, index=X.columns, columns=["PC1", "PC2", "PC3", "PC4"])
print(loadings.round(3))

# 3. 주성분 4개로 회귀 (논문 표 6)
pcs = pca.transform(Z)[:, :4]
pcr = sm.OLS(y, sm.add_constant(pcs)).fit()
print("\n[3] 주성분 회귀")
print(pd.DataFrame({"coef": pcr.params.values, "p": pcr.pvalues.values},
                   index=["const", "PC1", "PC2", "PC3", "PC4"]).round(4))
print("    조정 R²:", round(pcr.rsquared_adj, 3))

# 4. 한계 점검 ① 한 경로씩 빼고 예측 (LOOCV): 표본 밖에서도 맞는가
pred = np.empty(n)
for i in range(n):
    tr = np.arange(n) != i
    sc = StandardScaler().fit(X[tr])
    pc = PCA(4).fit(sc.transform(X[tr]))
    m = sm.OLS(y[tr], sm.add_constant(pc.transform(sc.transform(X[tr])))).fit()
    pred[i] = m.predict(np.r_[1, pc.transform(sc.transform(X[~tr]))[0]].reshape(1, -1))[0]
cv_r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"\n[4] LOOCV R² = {cv_r2:.3f}, 평균 절대오차 = {np.abs(y - pred).mean():.1f}m  (조정 R² {pcr.rsquared_adj:.3f}와 비교)")

# 5. 한계 점검 ② 모형 없이 순위상관만으로도 같은 결론이 나오는가
rho = {c: spearmanr(X[c], y).correlation for c in X.columns}
print("\n[5] 항목별 실측 거리와의 스피어만 순위상관 (상위 5개)")
print(pd.Series(rho).sort_values(ascending=False).head(5).round(2))
