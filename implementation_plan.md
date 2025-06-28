# 実装方針

## 全体アーキテクチャ

### ベースモデル
- **Transformer**: BERT-baseをベースとした修正版
- **フレームワーク**: PyTorch + Transformers library
- **計算環境**: GPU (CUDA対応)

### 幾何学的拡張

#### 計量テンソルモジュール
```python
class MetricTensor(nn.Module):
    def __init__(self, hidden_size, rank=None):
        # rank=None: 対角計量, rank=r: 低ランク近似
        self.hidden_size = hidden_size
        self.rank = rank or hidden_size
        
    def forward(self, context_embeddings):
        # 文脈から計量テンソルを生成
        return metric_tensor
```

#### 測地線アテンション
```python
class GeometricAttention(nn.Module):
    def __init__(self, config):
        self.metric_layer = MetricTensor(config.hidden_size)
        
    def forward(self, query, key, value, context):
        metric = self.metric_layer(context)
        # G^(-1) の効率的計算
        attention_scores = torch.matmul(query, torch.solve(key.T, metric)[0])
        return attention_scores
```

## データ処理パイプライン

### データセット
1. **訓練データ**: Wikipedia + BookCorpus (BERT標準)
2. **評価データ**: 
   - WordSim-353, SimLex-999 (静的類似性)
   - CoSimLex, SCWS (文脈的類似性)
   - GLUE tasks (下流タスク)

### 前処理
```python
def preprocess_similarity_data(dataset_path):
    # 類似性データセットの読み込み
    # 文脈付きペアの抽出
    # トークナイゼーション
    return processed_data
```

## 実験設定

### ハイパーパラメータ
```yaml
model:
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 12
  metric_rank: 64  # 対角計量の場合は768

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 1000

evaluation:
  similarity_datasets: ["wordsim353", "simlex999", "cosimlx", "scws"]
  glue_tasks: ["cola", "sst2", "mrpc", "qqp"]
```

### 実験手順
1. **事前訓練**: 修正されたアテンション機構での言語モデル訓練
2. **ファインチューニング**: 各下流タスクでの微調整
3. **類似性評価**: 測地線距離による類似性計算
4. **比較分析**: ベースライン（BERT）との性能比較

## 評価メトリクス実装

### 測地線距離計算
```python
def geodesic_distance(emb1, emb2, metric_tensor):
    # Riemannian距離の近似計算
    diff = emb1 - emb2
    distance = torch.sqrt(torch.matmul(diff.T, torch.matmul(metric_tensor, diff)))
    return distance

def evaluate_similarity(model, dataset):
    # 人間評価との相関計算
    predicted_similarities = []
    human_similarities = []
    
    for word1, word2, context1, context2, human_score in dataset:
        # 文脈埋め込みの取得
        emb1 = model.get_contextual_embedding(word1, context1)
        emb2 = model.get_contextual_embedding(word2, context2)
        
        # 測地線距離による類似性
        similarity = 1 / (1 + geodesic_distance(emb1, emb2, metric))
        predicted_similarities.append(similarity)
        human_similarities.append(human_score)
    
    correlation = spearman_correlation(predicted_similarities, human_similarities)
    return correlation
```

## 結果出力形式

### 自動生成される結果ファイル
```
results/
├── similarity_results.json      # 類似性タスクの結果
├── glue_results.json           # GLUEタスクの結果
├── computational_metrics.json   # 計算効率性の結果
├── visualizations/             # 可視化ファイル
│   ├── attention_heatmaps.png
│   ├── meaning_space_tsne.png
│   └── curvature_analysis.png
└── model_checkpoints/          # 訓練済みモデル
```

### 結果JSONの形式
```json
{
  "similarity_results": {
    "wordsim353": {"spearman": 0.75, "pearson": 0.73},
    "simlex999": {"spearman": 0.68, "pearson": 0.66},
    "cosimlx": {"spearman": 0.82, "pearson": 0.79},
    "scws": {"spearman": 0.71, "pearson": 0.69}
  },
  "baseline_comparison": {
    "bert_base": {"cosimlx": 0.74, "scws": 0.65},
    "improvement": {"cosimlx": "+0.08", "scws": "+0.06"}
  },
  "computational_metrics": {
    "training_time": "4.2 hours",
    "inference_speed": "124 samples/sec",
    "memory_usage": "12.4 GB"
  }
}
```

## デバッグとモニタリング

### ログ出力
- 訓練中の損失関数の推移
- 計量テンソルの固有値分布
- アテンション重みの統計
- メモリ使用量の監視

### 可視化
- t-SNEによる意味空間の可視化
- 計量テンソルのヒートマップ
- 測地線の軌道描画
- 文脈による意味変化の可視化

## エラーハンドリング
- GPU メモリ不足時の自動バッチサイズ調整
- 数値不安定性（行列の逆行列計算）の回避
- 収束しない場合の学習率自動調整