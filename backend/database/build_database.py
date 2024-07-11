from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Text
from globals import *

engine = create_engine(conn_str)
connection = engine.connect()
metadata = MetaData()

t1 = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("login", Text, nullable=True),
    Column("password", Text, nullable=True),
    Column("name", Text, nullable=True),
    Column("surname", Text, nullable=True),
    Column("patronymic", Text, nullable=True)
)


def create_database():
    global metadata

    metadata.bind = engine
    metadata.drop_all()
    metadata.create_all(engine)
