import datetime as dt
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, EmailStr, ValidationError, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.booking import Booking


class BookingInput(BaseModel):
    name: str
    email: EmailStr
    date: dt.date
    time: dt.time

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("name is too short")
        return v

    @model_validator(mode="after")
    def not_in_past(self) -> "BookingInput":
        if dt.datetime.combine(self.date, self.time) < dt.datetime.now():
            raise ValueError("the interview date and time must be in the future")
        return self


class BookingStore(Protocol):
    async def add(self, data: BookingInput) -> None: ...


class SqlBookingStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add(self, data: BookingInput) -> None:
        async with self._session_factory() as session:
            session.add(
                Booking(
                    name=data.name,
                    email=str(data.email),
                    interview_date=data.date,
                    interview_time=data.time,
                )
            )
            await session.commit()


BOOK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "book_interview",
        "description": (
            "Book an interview. Call ONLY when the user has provided all four "
            "details: name, email, date and time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "email": {"type": "string", "description": "Email address"},
                "date": {"type": "string", "description": "Date as YYYY-MM-DD"},
                "time": {"type": "string", "description": "Time as HH:MM, 24-hour"},
            },
            "required": ["name", "email", "date", "time"],
        },
    },
}


class BookingService:
    """Validates tool arguments from the LLM and saves valid bookings.

    Always returns a plain string, which is fed back to the LLM as the tool result.
    """

    def __init__(self, store: BookingStore) -> None:
        self._store = store

    async def book(self, arguments: dict[str, Any]) -> str:
        try:
            data = BookingInput(**arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'booking'}: "
                f"{err['msg'].removeprefix('Value error, ')}"
                for err in exc.errors()
            )
            return (
                f"NOT BOOKED. Invalid input: {problems}. "
                "Ask the user to correct this. Do not tell them it was booked."
            )
        await self._store.add(data)
        return (
            f"BOOKED. Interview for {data.name} ({data.email}) on "
            f"{data.date.isoformat()} at {data.time.strftime('%H:%M')}."
        )