# QAOA 量子最佳化「可行性診斷」框架（v4）

> 目的：在投入量子硬體之前，用**可重現、對照古典基線、允許得出「不值得」結論**的方式，評估 CVaR-QAOA、Warm-start QAOA、R-QAOA 在製造排程類問題上的價值。

## 這一版和 v3 差在哪（對照 v3 的主要缺陷）
| v3 的問題 | v4 的處理 |
|---|---|
| 只在 J1=J2=1（基態簡併 6/16）測試；隨機亂猜的 CVaR 也等於 −2.000，測試等於空轉 | 掃描 J2；所有指標與**均勻隨機基線**比較（`lift`）；`tests/` 內有重現該缺陷的測試 |
| CVaR 能量與期望值能量並排比較 | 統一指標：基態機率、lift、期望擊中次數、近似比、單答案命中 |
| Warm-start 的 c* 是寫死、且剛好指向基態 | c* 由**真正的古典啟發式**產生（含 ε 正則化）；另設「刻意較差起點」對照；並證明純 ZZ 問題的凸鬆弛恆為 c=0.5（無資訊） |
| R-QAOA 只做第一步，卻宣稱已實作 | 完整遞迴（降維→暴力解殘餘→回推），並測試降維的**精確性** |
| 無古典基線、無重複、無 seed | 1-flip 局部搜尋、SA 基線；3 個獨立主種子 × 多實例 × 多初值；同種子逐位元可重現 |
| 提案宣稱「有測試／梯度正確／完全正確」但 `tests/` 不存在 | 15 個單元測試（含獨立稠密矩陣模擬器交叉比對） |
| 4-qubit「製造情境」沒有映射 | 改用**兩產線指派**（負載平衡 + 換線衝突）的 Ising 展開，並有直接成本函數的對照測試 |

## 快速開始（只需 numpy / scipy / matplotlib）
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v          # 15 個測試，約 2 秒
python scripts/j1j2_sweep.py                      # 2x2 J1-J2 掃描
python scripts/benchmark.py --seed 0 --sizes 8 10 12 --instances 8 --inits 4   # 約 5 分鐘
bash scripts/run_all_seeds.sh && python scripts/aggregate.py                    # 3 種子重複驗證 + 彙整圖表
python scripts/repro_check.py                     # 可重現性檢查
python scripts/crosscheck_qiskit.py               # 選配：與 Qiskit 交叉比對（需自行安裝 qiskit）
```

## 主要結果（3 個獨立種子 × 3 種規模，詳見 `results/summary_all_seeds.md`）
- 標準 QAOA(p=1) 對隨機取樣的基態機率增益 **約 2.2–2.4 倍，且不隨 n 增長**；n=12 時基態機率仍只有約 0.1%。
- CVaR-QAOA(α=0.1) 的 lift 在 9/9 組都高於標準 QAOA（約 2.5–2.8 倍），但「機率最高的位元串」是基態的比率只有 0–0.25（標準 QAOA 為 0.12–0.75）——它優化的是尾端質量，不是峰值。
- Warm-start 的好壞**幾乎完全由古典起點決定**：一般起點平均 lift 2.5–5.9 倍（跨種子變異大），刻意較差的起點只有 1.3–1.6 倍。
- R-QAOA（完整遞迴）單答案命中率 0.29–0.67，**高於單次局部搜尋，但在 7/9 組低於一條 1024 步的古典 SA 鏈**（0.35–0.98）。
- 結論：在此規模，古典方法全面勝出；本框架的價值是**提供一套能誠實回答「值不值得上量子」的診斷流程**，而不是宣稱量子優勢。

## 已知限制（請在簡報/報告中主動說明）
- 全部為**精確、無雜訊**的狀態向量模擬，n ≤ 12，p=1（J1-J2 掃描為 p=2）。沒有硬體、沒有雜訊結果。
- 問題為隨機產生的「兩產線指派」，不是真實工廠資料；也不是完整的 job-shop。
- 實例數有限（每規模 24 個，分在 3 個種子），個別比較（例如 R-QAOA vs SA）在不同種子間會翻轉，詳見穩定性表。
- `scripts/crosscheck_qiskit.py` 尚未在撰寫者環境執行（環境無法安裝 qiskit），請在你的機器上跑一次並保留輸出。

## 結構
```
src/qdiag/   ising.py（問題）sim.py（模擬器）algos.py（四種演算法+指標）classical.py（古典基線）
tests/       test_core.py
scripts/     benchmark.py  aggregate.py  j1j2_sweep.py  repro_check.py  run_all_seeds.sh  crosscheck_qiskit.py
results/     各種子 JSON/MD、summary_all_seeds.md、summary.png、j1j2_sweep.md
docs/        proposal.md（企劃書）  verification_report.md（驗證報告）
```

## 參考文獻
- Barkoutsos et al., *Improving Variational Quantum Optimization using CVaR*, Quantum 4, 256 (2020).
- Egger, Mareček, Woerner, *Warm-starting quantum optimization*, Quantum 5, 479 (2021).
- Bravyi, Kliesch, Koenig, Tang, *Obstacles to variational quantum optimization from symmetry protection*, PRL 125, 260505 (2020)（R-QAOA 的提出）。
- Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
（引用以作者所知為準，投稿前請逐筆核對卷期頁碼。）
