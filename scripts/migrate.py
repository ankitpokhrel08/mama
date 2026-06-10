"""Run once to apply schema changes to an existing database."""
import sys
sys.path.insert(0, ".")

from sqlalchemy import text
from app.database import engine

migrations = [
    "ALTER TABLE violations ADD COLUMN IF NOT EXISTS vehicle_color VARCHAR",
    "ALTER TABLE violations ADD COLUMN IF NOT EXISTS vehicle_image_path VARCHAR",
    # screenshot_path no longer used but kept nullable so old rows stay valid
]

with engine.connect() as conn:
    for sql in migrations:
        conn.execute(text(sql))
        print(f"OK: {sql}")
    conn.commit()

print("Migration complete.")
