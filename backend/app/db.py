"""Base de datos: SQLAlchemy (Postgres en prod, SQLite en local) + modelos."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (JSON, DateTime, ForeignKey, String, Text, create_engine, func)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker)

from .config import settings

_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    threads: Mapped[list["Thread"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Thread(Base):
    __tablename__ = "threads"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # uuid
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="Nueva consulta")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[User] = relationship(back_populates="threads")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="thread", cascade="all, delete-orphan", order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))           # "medico" | "asistente"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, default=list)  # [{n,guia,seccion,pagina}]
    model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    veredicto: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    thread: Mapped[Thread] = relationship(back_populates="messages")


def init_db() -> None:
    Base.metadata.create_all(engine)
