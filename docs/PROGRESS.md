# PROGRESS.md

## 現在のフェーズ
フェーズ0：計画と環境構築【承認】— 残作業あり（候補銘柄数の確認、GitHub の設定）

## 完了したこと
- ユーザーへの質問と回答の記録（Windows 11、J-Quants 未契約で Standard 予定、楽天証券、朝9:00前と日中に発注可、GitHub アカウントあり）
- J-Quants V2 の公式仕様の確認（使うデータ、プラン別の期間・遅延・レート制限、更新時刻、利用規約の解約後の扱い）→ `docs/PLAN.md` 第1・2章
- 楽天証券の注文方法の確認（寄付指値・引成・逆指値は可、IFD の売り側に逆指値は不可、**引成は前場中に入れると前場引けで約定**）→ `docs/PLAN.md` 第3章
- 仮想環境 `.venv`（Python 3.14.5）と主要ライブラリの導入・動作確認、`requirements.txt`・`requirements.lock.txt`
- プロジェクトの構成、`config/base.yaml`・`config/production.yaml`、`.gitignore`、`.env.example`
- J-Quants クライアント（`src/data/jquants.py`：ページング、レート制限、429 の待機、キーを表示しない）
- 候補銘柄数の確認スクリプト（`src/analysis/phase0_candidates.py`）
- テスト 8件（`pytest`：すべて成功）
- git の初期化（ローカルのみ。GitHub は未接続）
- `docs/PLAN.md`（ドラフト）、`docs/DECISIONS.md`、`docs/EXPERIMENTS.md`、`docs/RELEASES.md`、`reports/phase0.md`（ドラフト）

## 次にやること
1. `python -m src.analysis.phase0_candidates` の結果（実行中、2026-10-04 開始）を PLAN.md 第4章に記入し、N を提案
2. `reports/phase0.md` を完成させ、フェーズ0の承認を依頼する

## 未解決の問題・ユーザーへの確認事項
- `git push` と J-Quants API への接続はユーザーが許可（2026-10-04）。GitHub への初回送信は完了（コミットのメールは noreply に変更）
- CLAUDE.md に第12章（会話の圧縮への備え）が追記されている（作業ツリーの変更・未コミット）。ユーザーの追記として扱い従う。コミットしてよいか確認中
- `.env` は作成済み（キー設定済み）。`.env.example` が `.env` に名前変更されていたため、ひな形を git から復元した。API キーがコミット対象・履歴に含まれないことを確認済み
- GitHub：`origin` = https://github.com/keisuke-sunahara/small-cap-stock.git（リモートは空であることを確認済み）
- 評価役：同じPCの `evaluator/` で、ユーザーご本人が起動・管理。データは私的利用の範囲で外部共有なし（2026-10-04 回答）→ 利用規約第8条の懸念は解消
- J-Quants への問い合わせ：Standard → Light に下げたとき、5年より古いデータを保持してよいか
- PLAN.md 第7.1節（ルールの細部4点）、第9章（合格基準の追加3点）、第10章（CLAUDE.md の改善提案5点）への判断

## 最終更新日時
2026-10-04
