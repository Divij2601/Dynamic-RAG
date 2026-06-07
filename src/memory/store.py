from datetime import datetime

from src.database.mongo_client import (
    mongo_client
)

from src.observability.logger import (
    app_logger
)


class ConversationStore:
    """
    Mongo conversation store
    """

    def __init__(self):

        self.db = (
            mongo_client
            .get_database()
        )

        self.collection = (
            self.db[
                "conversation_memory"
            ]
        )

    def save_interaction(
        self,
        session_id: str,
        query: str,
        answer: str,
        route: str,
        confidence: float,
        query_id: str = None
    ):
        """
        Save one conversation turn.
        query_id links this turn to the trace record
        so sources can be retrieved later.
        """

        document = {
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "route": route,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }

        if query_id:
            document["query_id"] = query_id

        self.collection.insert_one(document)

        # Auto-name the session from its first message
        # (only if no custom name exists yet).
        self._auto_name_session(session_id, query)

        # Update last-active timestamp on the session.
        self._upsert_session_metadata(session_id)

        app_logger.success("Interaction saved")

    def _auto_name_session(
        self,
        session_id: str,
        query: str
    ):
        """
        Set the session name to the first user query
        (truncated) if no name has been set yet.
        """

        db = mongo_client.get_database()
        existing = db["session_metadata"].find_one(
            {"session_id": session_id},
            {"_id": 0, "name": 1}
        )

        if not existing:
            auto_name = (
                query[:50].strip() + "…"
                if len(query) > 50
                else query.strip()
            )
            db["session_metadata"].insert_one({
                "session_id": session_id,
                "name": auto_name,
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow()
            })

    def _upsert_session_metadata(
        self,
        session_id: str
    ):
        """
        Update the last_active timestamp for a session.
        """

        db = mongo_client.get_database()
        db["session_metadata"].update_one(
            {"session_id": session_id},
            {"$set": {"last_active": datetime.utcnow()}},
            upsert=True
        )

    def rename_session(
        self,
        session_id: str,
        name: str
    ):
        """
        Set or update a custom name for a session.
        """

        db = mongo_client.get_database()
        db["session_metadata"].update_one(
            {"session_id": session_id},
            {"$set": {
                "name": name.strip(),
                "last_active": datetime.utcnow()
            }},
            upsert=True
        )

        app_logger.info(
            f"Session {session_id} renamed: {name!r}"
        )

    def get_all_sessions(self, limit: int = 50) -> list:
        """
        Return all sessions ordered by most recent
        activity, newest first.

        Each item:
        {
          session_id, name,
          last_active, created_at,
          message_count, preview
        }
        """

        db = mongo_client.get_database()

        # Aggregate turns to get count + preview per session.
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$session_id",
                "last_active": {"$first": "$timestamp"},
                # last in reverse-sorted = chronologically first
                "first_query": {"$last": "$query"},
                "message_count": {"$sum": 1}
            }},
            {"$sort": {"last_active": -1}},
            {"$limit": limit}
        ]

        rows = list(
            db["conversation_memory"].aggregate(pipeline)
        )

        # Join with session_metadata for custom names.
        meta_by_id = {
            m["session_id"]: m
            for m in db["session_metadata"].find(
                {"session_id": {
                    "$in": [r["_id"] for r in rows]
                }},
                {"_id": 0}
            )
        }

        sessions = []
        for row in rows:
            sid = row["_id"]
            meta = meta_by_id.get(sid, {})
            preview = row.get("first_query", "")
            if len(preview) > 45:
                preview = preview[:45] + "…"
            sessions.append({
                "session_id": sid,
                "name": meta.get("name", sid),
                "last_active": row.get("last_active"),
                "created_at": meta.get("created_at"),
                "message_count": row.get(
                    "message_count", 0
                ),
                "preview": preview
            })

        return sessions

    def load_session_messages(
        self,
        session_id: str,
        limit: int = 100
    ) -> list:
        """
        Reconstruct the Streamlit messages list from
        stored conversation turns for a session.
        Sources are recovered from the traces collection
        via the stored query_id.
        """

        db = mongo_client.get_database()

        turns = list(
            self.collection
            .find({"session_id": session_id})
            .sort("timestamp", 1)
            .limit(limit)
        )

        # Pre-fetch all relevant traces in one query.
        query_ids = [
            t["query_id"] for t in turns
            if t.get("query_id")
        ]

        traces_by_qid = {}
        if query_ids:
            for tr in db["traces"].find(
                {"request_id": {"$in": query_ids}},
                {"_id": 0}
            ):
                traces_by_qid[tr["request_id"]] = tr

        messages = []
        for turn in turns:
            messages.append({
                "role": "user",
                "content": turn.get("query", "")
            })

            qid = turn.get("query_id")
            trace = traces_by_qid.get(qid, {})

            messages.append({
                "role": "assistant",
                "content": turn.get("answer", ""),
                "meta": {
                    "route": turn.get("route", "unknown"),
                    "status": "success",
                    "confidence": turn.get("confidence"),
                    "faithfulness_score":
                        trace.get("faithfulness_score"),
                    "grounded": trace.get("grounded"),
                    "latency_ms": (
                        (trace.get("retrieval_latency_ms") or 0)
                        + (trace.get("generation_latency_ms") or 0)
                    ) or None,
                    "query_id": qid,
                    "sources": trace.get("sources", [])
                }
            })

        return messages

    def get_recent_context(
        self,
        session_id: str,
        limit: int = 5
    ):
        """
        Get recent conversation turns (chronological).
        """

        results = list(
            self.collection
            .find({
                "session_id": session_id
            })
            .sort("timestamp", -1)
            .limit(limit)
        )

        results.reverse()
        return results

    def count_turns(
        self,
        session_id: str
    ) -> int:
        """
        Count total turns in a session.
        """

        return self.collection.count_documents(
            {"session_id": session_id}
        )

    def save_summary(
        self,
        session_id: str,
        summary: str,
        turn_count: int
    ):
        """
        Upsert the rolling summary for a session.
        """

        db = mongo_client.get_database()
        db["session_summaries"].update_one(
            {"session_id": session_id},
            {"$set": {
                "summary": summary,
                "turn_count": turn_count,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )

    def get_summary(
        self,
        session_id: str
    ) -> str:
        """
        Return the stored session summary or empty string.
        """

        db = mongo_client.get_database()
        doc = db["session_summaries"].find_one(
            {"session_id": session_id},
            {"_id": 0, "summary": 1}
        )

        return doc["summary"] if doc else ""


conversation_store = (
    ConversationStore()
)