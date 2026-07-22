# Multi-Station Graph 擴充：架構設計與開發規劃

> Branch: `feature/multi-station-graph`
> Status: design draft（尚未實作）
> 目標：在現有單站 Graph-TCN-VAE 上，引入多個一般測站（單模態）資料，以提升 super-station 的 Chem/PSD 缺值填補（後續可延伸至預測）。

---

## 1. 問題陳述

現況：模型以**單一超級測站**的多模態資料（Chem ×31 + PSD ×230 bins）做 joint imputation，核心輸出是 calibrated uncertainty（指標優先序 `PICP > CRPS > R²`）。

動機：超級站儀器與維護成本高；台灣環境部有大量**一般測站**，但只有單模態資料（PM2.5、SOx、NOx、O3、CO 與氣象參數）。先前單站實驗顯示「化學成分能幫助 PSD 填補」，因此假設：**多個一般站的單模態資料，若帶有污染物 local 傳輸（advection）訊息，可作為額外輸入提升填補性能。**

核心難點：現有 `forward(x, cond, mask)` 三個主輸入皆為 `[B, W, D]`，**天生沒有站點軸**。多站資料天生是 `[B, S, W, D_s]`，且一般站特徵維度 `D_s` 比 super-station 小（異質）。直接 concat 進 `cond` 會：(a) 抹除站點身份 → 重演 distribution-shift 失敗；(b) 把 block-missing 的缺站當「恆觀測」→ 違反 mask invariant。

---

## 2. 參考文獻定位

**Wu et al. 2025（ES&A, Estimating PSD using deep learning）— 科學動機 + 反面教材。**
證明用 routine air-quality 資料（trace gas + met + Ntot）可估 PSD。但其多站做法是把多站直接 pool（"AllTrain"）、**不標站點身份**：對相似環境可行、對不同環境（SMEAR I / Qvidja）退化。教訓：(1) 必須標記站點與環境，不能盲 pool；(2) 一個高資訊量協變數（Ntot）勝過一堆弱協變數 → 要選帶 transport 訊號的站，不是堆站數。

**Wang et al. 2025（KBS, MI-EL, Matrix-Informed Ensemble Learning）— 架構 idea 來源。**
核心是把空間-時間權重矩陣 `W[N×T]` 低秩分解為**空間因子 `SFM[N×r]` × 時間因子 `TFM[r×T]`**；`SFM` 來自兩個鄰接矩陣（距離 `A^dist` + 數值相似度 `A^sim`）的 Laplacian Eigenmaps。

> 注意（重要修正）：MI-EL 分解的是 **ensemble 權重**，不是 graph node representation。可遷移的只有「**低秩因子乘積 + 因子共享**」這個原則，**不是** ensemble 機制本身。

---

## 3. 核心設計原則

現有模型只有一條圖軸：**feature 圖**（節點 = 物種 / PSD bins，單站）。多站要新增第二條軸：**station 軸**。

不採用「節點 =（station, feature）乘積後做單一全域 attention」的暴力做法（節點爆炸、難訓練、且違反專案紅線「不要直接跳進 full feature-time token Transformer」）。改採**因子化 + axial（軸向）attention**：

1. **節點 =（station, species）**，但表示用低秩因子化，讓「同一物種跨站共享因子」。
2. **Axial attention 交替兩軸**：跨站（同物種軸）↔ 跨物種（站內軸），聯合學習，取消人為的「先後順序」。
3. **沿用既有 `local_context` 旁路 gated 注入**：主幹（`x/cond/mask` → encoder → feature graph → UQ）一個字不動。

這同時解決三個顧慮：
- 異質維度：缺的（station, species）= 圖上不存在的 node，免 padding、免投到共同維度。
- 順序粗糙：axial 交替取代序列式 graph A→B。
- 過擬合：低秩因子化是讓「高解析節點圖」可訓練的**必要** regularizer，不是選配。

---

## 4. 模型架構

### 4.1 節點與因子化嵌入

對節點（station `s`, variable `f`）的視窗觀測 `x_{s,f,·}`：

```
h_{s,f} = φ(  value_enc(x_{s,f,·}, mask_{s,f,·})     # 該站該變數的觀測編碼
            + E_species[f]                            # 物種因子：跨所有測到 f 的站「共享」
            + E_station[s] )                          # 站點因子：含空間 + 風向 embedding
```

- `E_species[f]`：跨站共享 → 用所有測該變數的站一起估，well-regularized、可遷移；super-station-only 的物種優雅退化為「單站 + 共享因子」。**這是跨站物種傳遞的橋樑。**
- `E_station[s]`：來自雙鄰接 Laplacian Eigenmaps（`A^dist` ⊕ `A^sim`）＋（Stage 3）風向 embedding。

### 4.2 Axial Attention（交替 ×L）

- **Step A — 跨站（同物種軸）**：對每個共享變數 `f`，在站點集合上 attention：
  ```
  α_{t→s} = softmax_s [ (W_q h_t)·(W_k h_s)/√d  +  b_geo(dist)  +  b_wind(風向·地理向量, lag) ]
  ```
  作用：把多站同一變數的時空傳輸模式聚合成空間上更豐富的表示。
- **Step B — 跨物種（站內軸）**：即既有 feature graph（`InputGraphLayer`），super-station 物種彼此 attention，並吸收 Step A 精煉後的共享變數。

交替 L 次。物種填補**永遠由 Step B（feature graph）完成**；Step A 只提供「平流 context」。

### 4.3 注入與 backbone

```
station_context = StationGraphEncoder(stations, cond, mask)   # [B, C, W]
local_context  = local_context + gate ⊙ station_context        # gate init ≈ -2
```

注入點 = forward 中既有的 `local_context / history_context` 縫（與 `ExternalHistoryContext` 同一條 rail）。`stations=None` 時行為完全等於現行 `26e` baseline。

### 4.4 forward 介面變更（最小 diff）

```python
def forward(self, x, cond, mask, history=None, stations=None, sample_latent=True):
    # stations = {
    #   'feats':      [B, S, W, D_s],   # 一般站單模態特徵
    #   'mask':       [B, S, W],        # 每站每時刻是否在線（block missing → first-class mask）
    #   'station_id': [B, S],           # 站點索引 → learnable embedding
    #   'geo':        預算好的 A_dist / A_sim（+ Stage3 風向量）
    # }
```

---

## 5. 必守約束（來自既有 project memory）

- 缺站是 **block missing → first-class mask**，不可 zero-fill；`mask_embed` 在 instance norm 之後。
- 小心 **variance-head magnitude-proxy bug**：加站不可讓 input magnitude 變大而假性變自信；Stage 3 讓「transport 明確→收緊、缺站→放寬」顯式接 support→logvar。
- **held-out / observed 指標分開**；以 official PICP 為區間指標、CRPS 為 model-selection。
- 不碰 PSD mode 邊界（station 軸與 bins 正交，安全）。
- 不一步到位做 full（station×species×time）token 圖 → 靠 staged evidence 逼近。

---

## 6. 誠實的限制（解析度不對稱）

- 一般站**無 PSD** → 跨站軸（Step A）實際只作用在共享子集（PM2.5 / gases）。
- PSD 的增益是**間接**的（靠 chem→PSD 的 feature graph），**不要期待對稱**。
- 多站 PM2.5 能傳的物種訊息上限 = PM2.5 與物種協變的程度：對 accumulation mode、二次無機鹽（SO4²⁻/NO3⁻/NH4⁺）幫助最大；對 nucleation mode、痕量金屬、與 PM2.5 解耦的物種幾乎傳不過去。
- 高解析只有在「足夠時間對齊的並存資料」下有意義 → 必須先做 Stage 0。

---

## 7. 分階段開發規劃

| Stage | 內容 | 目的 / 通過條件 |
| --- | --- | --- |
| **S0** | 可行性驗證：RF / 相關性 + lagged 多站特徵重要性（複刻 Paper 1 但用自己的資料） | 確認哪些站、哪個 lag 真有 transport 訊號；資料是否時間對齊可得。**無訊號則停。** |
| **S1** | 粗版：station = 1 node，經 `CrossModalGraphLayer` 注入 `local_context` | 隔離驗證「多站到底有沒有用」；held-out CRPS/PICP 不退於 `26e` |
| **S2** | factorized（station×species）node + axial attention（對稱鄰接 `A^dist`+`A^sim`） | species-resolved 傳輸；held-out CRPS/PICP 改善且 mean 不顯著退步 |
| **S3** | 風向條件化非對稱鄰接 `b_wind` + lagged upwind retrieval（複用 `ExternalHistoryContext`） | 顯式平流；UQ 隨 transport 訊號自適應；延伸至 `PredictionVAE_Graph` 預測 |

每階段以 `stations=None`（= `26e`）為 A/B 對照，observed-point 與 held-out-point 指標嚴格分開。

---

## 8. 模組與檔案規劃

- 新增 `graph_tcn_vae/station_graph.py`：`StationGraphEncoder`、`StationFactor`（Eigenmaps）、`SpeciesFactor`、axial attention block。
- `model_graph_uq.py`：`ImputationVAE_Graph.__init__` 增 `use_station_graph` 等旗標；`forward` 增 `stations` 參數與注入縫（§4.3）。
- `dataset.py` / `masking_utils.py`：多站張量打包與 station-level block masking。
- `tests/`：`stations=None` 等價性測試（必須 bit-for-bit 等於現行 forward）、shape 測試、缺站 mask 行為測試。

---

## 9. 評估協定

- 主指標：held-out PICP（區間有效性 gate）、標準化 CRPS（model selection）；R² 僅 sanity。
- 對照：同 mask-seed 下 `stations=None` vs `stations=各 Stage`。
- 拆解：分別報 Chem / PSD、observed-point / held-out-point。
- 可解釋性產出：station-factor / species-factor scores（指出哪個站、哪個物種驅動填補）— 對應論文 interpretability 框架。

---

## 10. 非目標 / 紅線

- 不把多站特徵直接 concat 進 `x` 或 `cond`。
- 不一次蓋 full token transformer。
- 不為了加站而觸發 magnitude-proxy 假自信。
- 不改動已驗證有效的 feature graph 主幹（只透過 gated 旁路）。
- 不引入跨 PSD mode 邊界平滑。
