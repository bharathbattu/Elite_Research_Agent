import json
import sqlite3
from pathlib import Path

from .models import ReportSummary, ResearchReport


class ReportRepository:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS reports_created_at ON reports(created_at DESC)"
            )

    def save(self, report: ResearchReport) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO reports(id, query, topic, model, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.query,
                    report.topic,
                    report.model,
                    report.created_at.isoformat(),
                    report.model_dump_json(),
                ),
            )

    def get(self, report_id: str) -> ResearchReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM reports WHERE id = ?", (report_id,)
            ).fetchone()
        return ResearchReport.model_validate_json(row["payload"]) if row else None

    def list(self, limit: int = 50) -> list[ReportSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, query, topic, model, created_at
                FROM reports ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [ReportSummary.model_validate(dict(row)) for row in rows]

    def delete(self, report_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        return cursor.rowcount > 0

    def export_json(self, report_id: str) -> str | None:
        report = self.get(report_id)
        return json.dumps(report.model_dump(mode="json"), indent=2) if report else None
