import json
import os
import random
from typing import List, Literal

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="動画投稿シミュレーション API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CommentReq(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=4000)
    subscribers: int = Field(ge=0, le=999_999_999)
    total_views: int = Field(ge=0, le=999_999_999_999)
    video_views: int = Field(ge=0, le=999_999_999_999)
    comment_count: int = Field(default=8, ge=3, le=20)


class Comment(BaseModel):
    author: str
    body: str
    likes: int
    minutes_ago: int
    sentiment: Literal["positive", "neutral", "critical"]


class CommentResp(BaseModel):
    comments: List[Comment]
    source: str


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.post("/api/comments", response_model=CommentResp)
async def generate_comments(req: CommentReq):
    if client:
        comments = await generate_ai_comments(req)
        return CommentResp(comments=comments, source="openai")

    return CommentResp(comments=generate_local_comments(req), source="local-demo")


async def generate_ai_comments(req: CommentReq) -> List[Comment]:
    prompt = f"""
あなたは動画投稿サイトの自然な日本語コメント欄を作るAIです。
以下の動画に対して、視聴者コメントを{req.comment_count}件生成してください。

チャンネル登録者数: {req.subscribers:,}
チャンネル合計視聴回数: {req.total_views:,}
今回の動画視聴回数: {req.video_views:,}
タイトル: {req.title}
内容: {req.content}

条件:
- コメントは日本語中心で、たまに短い英語や絵文字を混ぜてもよい
- 反応は称賛、質問、ツッコミ、改善提案、常連感などをバランスよく含める
- 誹謗中傷、個人情報、危険行為の助長は避ける
- {{"comments": [...]}} 形式のJSONオブジェクトのみを返す
- comments の各要素は author, body, likes, minutes_ago, sentiment を持つ
- sentiment は positive, neutral, critical のいずれか
""".strip()

    completion = await client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.9,
        messages=[
            {"role": "system", "content": "You generate safe, realistic simulated video comments as JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw_content = completion.choices[0].message.content or "{}"
    parsed = json.loads(raw_content)
    raw_comments = parsed.get("comments", parsed if isinstance(parsed, list) else [])
    return [Comment(**item) for item in raw_comments[: req.comment_count]]


def generate_local_comments(req: CommentReq) -> List[Comment]:
    positive = [
        "このテーマ待ってました！タイトルからもうワクワクする。",
        "内容がわかりやすいし、最後まで見たくなる構成ですね。",
        "登録者数もっと伸びそう。次回も楽しみにしてます！",
        "編集テンポが良い想定で脳内再生できる😂",
    ]
    neutral = [
        "この動画の続編を出すなら、どの部分を深掘りしますか？",
        "初見です。チャンネルの他の動画も見てみます。",
        "サムネはどんな感じにする予定なんだろう。",
    ]
    critical = [
        "面白いけど、冒頭でもう少し結論が見えるとさらに見やすそう。",
        "内容は良いので、具体例がもう1つあるともっと刺さると思います。",
    ]
    pool = [(text, "positive") for text in positive]
    pool += [(text, "neutral") for text in neutral]
    pool += [(text, "critical") for text in critical]
    random.shuffle(pool)
    selected = [pool[index % len(pool)] for index in range(req.comment_count)]

    max_likes = max(5, min(req.video_views // 20 + req.subscribers // 100, 50_000))
    comments = []
    for body, sentiment in selected:
        comments.append(
            Comment(
                author=f"視聴者{random.randint(1000, 9999)}",
                body=body,
                likes=random.randint(0, max_likes),
                minutes_ago=random.randint(1, 60 * 24 * 14),
                sentiment=sentiment,
            )
        )
    return comments


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
