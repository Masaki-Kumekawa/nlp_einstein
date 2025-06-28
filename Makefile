# 幾何学的言語モデル研究：完全自動化Makefile
# 使用方法: make do

.PHONY: do setup data train evaluate paper clean help real

# メインターゲット：実際の研究実行
do: real

# 実際の研究実行（ダミーデータなし）
real: setup real-data real-experiment paper-real
	@echo "🎉 実際の研究パイプライン完了！"
	@echo "📄 論文: output/paper.tex"
	@echo "📊 実験結果: results/real_experiment_results.json"
	@echo "📈 詳細レポート: results/experiment_report.txt"

# 環境セットアップ
setup:
	@echo "🔧 環境セットアップ中..."
	@pip install torch transformers scipy matplotlib seaborn pandas datasets pytest scikit-learn tqdm requests
	@mkdir -p data results output src checkpoints models
	@echo "✅ 環境セットアップ完了"

# 実際のデータセット取得
real-data:
	@echo "📊 実際のデータセット準備中..."
	@python download_real_datasets.py
	@python download_full_datasets.py
	@echo "✅ 実データ準備完了"

# 実際の実験実行
real-experiment:
	@echo "🚀 実際の実験実行中..."
	@echo "⚠️ 注意: これは実際の機械学習実験です。完了まで数時間かかる可能性があります。"
	@python real_experiment.py
	@echo "✅ 実験完了"

# 実験結果に基づく論文生成
paper-real:
	@echo "📝 実験結果を基に論文生成中..."
	@python generate_paper_from_real_results.py
	@cd output && pdflatex paper.tex || echo "⚠️ PDF生成失敗：手動でpdflatex実行してください"
	@echo "✅ 論文生成完了"

# チェックポイントから評価して論文生成
paper-checkpoint:
	@echo "🔍 最新のチェックポイントから評価実行中..."
	@python evaluate_from_checkpoint.py
	@echo "📝 評価結果を基に論文生成中..."
	@python generate_paper_from_real_results.py
	@cd output && pdflatex paper.tex || echo "⚠️ PDF生成失敗：手動でpdflatex実行してください"
	@echo "✅ チェックポイントからの論文生成完了"

# 旧システム（デモ用）
demo: setup demo-data demo-train demo-evaluate demo-paper
	@echo "🎉 デモパイプライン完了！"
	@echo "📄 論文: output/paper.tex"
	@echo "📊 結果: results/experiment_results.json"

# デモ用データ準備
demo-data:
	@echo "📊 デモ用データ準備中..."
	@python data_pipeline.py
	@echo "✅ デモデータ準備完了"

# デモ用訓練
demo-train:
	@echo "🚀 デモ用モデル訓練中..."
	@python -c "import experiment_code; experiment_code.run_training()"
	@echo "✅ デモ訓練完了"

# デモ用評価
demo-evaluate:
	@echo "📈 デモ用評価実行中..."
	@python -c "import experiment_code; experiment_code.run_evaluation()"
	@echo "✅ デモ評価完了"

# デモ用論文生成
demo-paper:
	@echo "📝 デモ用論文生成中..."
	@python -c "import experiment_code; experiment_code.generate_paper()"
	@cd output && pdflatex paper.tex || echo "⚠️ PDF生成失敗：手動でpdflatex実行してください"
	@echo "✅ デモ論文生成完了"

# テスト実行
test:
	@echo "🧪 テスト実行中..."
	@python -m pytest src/ -v
	@echo "✅ テスト完了"

# 完全クリーンアップ（毎回同じ状態に戻す）
clean:
	@echo "🧹 完全クリーンアップ中..."
	@echo "⚠️ 以下のファイル・フォルダを削除します:"
	@echo "   - 実験結果 (results/)"
	@echo "   - 生成論文 (output/)"
	@echo "   - モデルチェックポイント (checkpoints/, models/)"
	@echo "   - ログファイル"
	@echo "   - 一時ファイル"
	@rm -rf results/ output/ checkpoints/ models/
	@rm -f experiment.log experiment_output.log experiment_progress.log experiment_status.json
	@rm -f *.pyc
	@rm -rf __pycache__/ src/__pycache__/
	@rm -rf .pytest_cache/
	@echo "✅ クリーンアップ完了 - 初期状態に戻りました"

# データも含む完全リセット
clean-all: clean
	@echo "🧹 データを含む完全リセット中..."
	@rm -rf data/
	@echo "✅ データを含む完全リセット完了"

# 軽量クリーンアップ（データセットは保持）
clean-light:
	@echo "🧹 軽量クリーンアップ中（データセット保持）..."
	@rm -rf results/ output/ checkpoints/ models/
	@rm -f experiment.log experiment_output.log experiment_progress.log experiment_status.json
	@rm -f *.pyc
	@rm -rf __pycache__/ src/__pycache__/
	@echo "✅ 軽量クリーンアップ完了"

# ヘルプ
help:
	@echo "幾何学的言語モデル研究自動化システム"
	@echo ""
	@echo "主要コマンド:"
	@echo "  make do          - 実際の研究実行（完全版）"
	@echo "  make demo        - デモ版実行（高速）"
	@echo "  make paper-checkpoint - 最新チェックポイントから論文生成"
	@echo "  make clean       - 完全クリーンアップ（初期状態に戻す）"
	@echo "  make clean-all   - データを含む完全リセット"
	@echo "  make clean-light - 軽量クリーンアップ（データ保持）"
	@echo ""
	@echo "段階実行:"
	@echo "  make setup       - 環境セットアップ"
	@echo "  make real-data   - 実データセット取得"
	@echo "  make real-experiment - 実験実行"
	@echo "  make paper-real  - 実験結果から論文生成"
	@echo "  make paper-checkpoint - チェックポイントから評価&論文生成"
	@echo ""
	@echo "その他:"
	@echo "  make test        - テスト実行"
	@echo "  make help        - このヘルプ"
	@echo ""
	@echo "🔄 毎回同じ条件で実行するには:"
	@echo "  make clean && make do"