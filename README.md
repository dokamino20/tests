# 動画投稿シミュレーションアプリ

登録者数・合計視聴回数を設定し、動画タイトルと内容を入力して投稿すると、AI APIを使ってコメント欄を生成するシンプルなWebアプリです。

## 機能

- チャンネルの登録者数と合計視聴回数を設定
- 「動画を追加」ボタンからタイトルと内容を入力して投稿
- 投稿ごとに推定視聴回数を計算
- OpenAI APIを使った自然なコメント欄の生成
- `OPENAI_API_KEY` が未設定でも動作確認できるローカルデモコメント生成

## セットアップ

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python server.py
```

ブラウザで `http://localhost:8000` を開いてください。

## 環境変数

- `OPENAI_API_KEY`: AIコメント生成に使うOpenAI APIキーです。未設定の場合はデモコメントを返します。
- `OPENAI_MODEL`: 使用するモデル名です。未設定の場合は `gpt-4o-mini` を使います。
- `PORT`: 起動ポートです。未設定の場合は `8000` です。
