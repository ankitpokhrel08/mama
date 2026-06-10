from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    plate_text: Mapped[str | None] = mapped_column(String, nullable=True)
    speed_kmh: Mapped[float] = mapped_column(Float)
    speed_limit_kmh: Mapped[int] = mapped_column(Integer)
    camera_id: Mapped[str] = mapped_column(String)
    location: Mapped[str] = mapped_column(String)
    vehicle_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    plate_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String, nullable=True)
    vehicle_color: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending/approved/rejected
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ticket_number: Mapped[str | None] = mapped_column(String, nullable=True)
