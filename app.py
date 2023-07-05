import os
import pandas as pd
from catboost import CatBoostClassifier
from datetime import datetime
from typing import List
from sqlalchemy import func, desc, create_engine, Column, Integer, Text
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, HTTPException
from database import SessionLocal, Base
from schema import UserGet, PostGet, FeedGet
from table_feed import Feed
from table_post import Post
from table_user import User


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


post_text = batch_load_sql("SELECT * FROM public.cocu_xyu_base")
user_data = batch_load_sql("SELECT * FROM public.cocu_xyu_user")
feed = batch_load_sql("SELECT * FROM public.cocu_xyu_feed")


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model.cbm")
    # LOAD MODEL HERE PLS :)
    cbm = CatBoostClassifier()
    return cbm.load_model(model_path)


pipe = load_models()


cbm = load_models()


app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db


class Post(Base):
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    topic = Column(Text)


def get_top_post(user_id: int):
    user_info = user_data[user_data['user_id'] == user_id]
    score_topic = {
        'movie': 0,
        'covid': 0,
        'tech': 0,
        'politics': 0,
        'business': 0,
        'sport': 0,
        'entertainment': 0
    }
    for i, post_info in post_text.iterrows():
        predict_data = pd.DataFrame(data={
            'gender': user_info['gender'],
            'timestamp': user_info['timestamp'],
            'topic': post_info['topic'],
        })
        pred = pipe.predict_proba(predict_data)[0][1]
        score_topic[post_info['topic']] += pred

    top_topic = 'movie'
    top_score = score_topic['movie']
    for topic, score in score_topic.items():
        if score > top_score:
            top_topic = topic
            top_score = score

    answer = []
    for _, post in post_text[post_text['topic'] == top_topic].iterrows():
        js_post = {
            "id": post['post_id'],
            "text": post['text'],
            "topic": post['topic']
        }
        answer.append(js_post)
    return answer


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int,
                      time: datetime,
                      limit: int = 5,
                      db: Session = Depends(get_db)) -> List[PostGet]:
    result = get_top_post(id)
    if result:
        return result
    raise HTTPException(404, 'post not found')



@app.get("/user/{id}", response_model=UserGet)
def find_user(id, db: Session = Depends(get_db)):
    result = db.query(User)\
        .filter(User.id == id).one_or_none()

    if result:
        return result
    else:
        raise HTTPException(404, "User not found")


@app.get("/post/{id}", response_model=PostGet)
def find_post(id, db: Session = Depends(get_db)):
    result = db.query(Post)\
        .filter(Post.id == id).one_or_none()

    if result:
        return result
    else:
        raise HTTPException(404, "Post not found")


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def user_feed(id, limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Feed) \
        .filter(Feed.user_id == id)\
        .order_by(Feed.time.desc())\
        .limit(limit).all()

    if result:
        return result
    else:
        raise HTTPException(404, "User not found")


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def post_feed(id, limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Feed) \
        .filter(Feed.post_id == id) \
        .order_by(Feed.time.desc()) \
        .limit(limit).all()

    if result:
        return result
    else:
        raise HTTPException(404, "Post not found")


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_recommended_feed(limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Post)\
        .select_from(Feed)\
        .filter(Feed.action == "like")\
        .join(Post)\
        .group_by(Post.id)\
        .order_by(desc(func.count(Post.id)))\
        .limit(limit).all()

    return result