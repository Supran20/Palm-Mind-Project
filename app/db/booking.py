import datetime as dt
import uuid

from sqlalchemy import Date, DateTime, String, Time
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255))
    interview_date: Mapped[dt.date] = mapped_column(Date)
    interview_time: Mapped[dt.time] = mapped_column(Time)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )