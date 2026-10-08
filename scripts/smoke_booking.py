import asyncio
from datetime import date, timedelta

from sqlalchemy import select

from app.core.config import get_settings
from app.db.booking import Booking
from app.db.session import Base, create_engine, create_session_factory
from app.services.booking import BookingService, SqlBookingStore


async def main() -> None:
    s = get_settings()
    engine = create_engine(s.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    service = BookingService(SqlBookingStore(factory))

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    cases = {
        "valid": {"name": "Test User", "email": "test@example.com", "date": tomorrow, "time": "14:30"},
        "bad email": {"name": "Test User", "email": "not-an-email", "date": tomorrow, "time": "14:30"},
        "past date": {"name": "Test User", "email": "test@example.com", "date": yesterday, "time": "14:30"},
        "missing fields": {"name": "Test User"},
        "bad time": {"name": "Test User", "email": "test@example.com", "date": tomorrow, "time": "25:99"},
    }
    for label, args in cases.items():
        print(f"\n[{label}]\n  {await service.book(args)}")

    async with factory() as session:
        rows = (await session.execute(select(Booking))).scalars().all()
        print(f"\nRows in bookings table: {len(rows)}")
        for r in rows:
            print(f"  {r.name} {r.email} {r.interview_date} {r.interview_time}")

    await engine.dispose()


asyncio.run(main())