"""
Conversation summariser.

When a session grows beyond MAX_HISTORY_TURNS the
older turns are compressed into a rolling summary
that is prepended to the planner / generator context.
Only the most recent SUMMARY_KEEP_RECENT turns are
kept verbatim; everything older is represented by
the summary text.

The summary is persisted to MongoDB (session_summaries)
so it survives process restarts.
"""

from typing import List, Dict

from src.config import settings
from src.models.groq_provider import groq_provider
from src.memory.store import conversation_store
from src.observability.logger import app_logger


class ConversationSummarizer:

    def should_summarise(
        self,
        session_id: str
    ) -> bool:
        """
        Return True when the session has grown past
        the configured threshold.
        """

        total = conversation_store.count_turns(
            session_id
        )

        return total > settings.MAX_HISTORY_TURNS

    def get_context_for_session(
        self,
        session_id: str
    ) -> Dict:
        """
        Return the best available context dict for
        the session:

            {
              "summary":       str,   # "" if none yet
              "recent_turns":  [...]  # last N turns
            }

        If the session is short, summary is empty and
        recent_turns holds all turns.
        If the session is long, summary holds the
        compressed history and recent_turns holds
        the last SUMMARY_KEEP_RECENT turns.
        """

        total = conversation_store.count_turns(
            session_id
        )

        if total <= settings.MAX_HISTORY_TURNS:

            recent = conversation_store.get_recent_context(
                session_id,
                limit=settings.MAX_HISTORY_TURNS
            )

            return {"summary": "", "recent_turns": recent}

        # Long session — fetch summary + recent turns.
        summary = conversation_store.get_summary(
            session_id
        )

        recent = conversation_store.get_recent_context(
            session_id,
            limit=settings.SUMMARY_KEEP_RECENT
        )

        # Rebuild the summary if it covers fewer turns
        # than the current session (i.e. it's stale).
        turns_in_summary = (
            total - settings.SUMMARY_KEEP_RECENT
        )

        if not summary:
            summary = self._build_summary(
                session_id,
                turns_to_summarise=turns_in_summary
            )

        return {"summary": summary, "recent_turns": recent}

    def _build_summary(
        self,
        session_id: str,
        turns_to_summarise: int
    ) -> str:
        """
        Fetch the older turns, ask the LLM to
        summarise them, and persist the summary.
        """

        # Fetch older turns (all minus the recent ones
        # that will be kept verbatim).
        all_turns = conversation_store.get_recent_context(
            session_id,
            limit=turns_to_summarise + settings.SUMMARY_KEEP_RECENT
        )

        # Exclude the SUMMARY_KEEP_RECENT most recent
        # (they'll be shown verbatim).
        older_turns = all_turns[:-settings.SUMMARY_KEEP_RECENT] if (
            len(all_turns) > settings.SUMMARY_KEEP_RECENT
        ) else all_turns

        if not older_turns:
            return ""

        history_text = "\n".join(
            f"User: {t.get('query', '')}\n"
            f"Assistant: {t.get('answer', '')}"
            for t in older_turns
        )

        prompt = (
            "You are summarising a conversation for "
            "a RAG assistant. Produce a concise "
            "summary (max 200 words) that captures "
            "the key topics discussed, decisions made, "
            "and any important facts the user referenced "
            "or asked about. This summary will be used "
            "as context for future turns.\n\n"
            "CONVERSATION:\n"
            f"{history_text}\n\n"
            "SUMMARY:"
        )

        try:
            summary = groq_provider.complete(
                prompt=prompt,
                model=settings.FAST_MODEL,
                temperature=0.0,
                max_tokens=300
            )

            conversation_store.save_summary(
                session_id=session_id,
                summary=summary.strip(),
                turn_count=len(older_turns)
            )

            app_logger.success(
                f"Session summary built for "
                f"{session_id} "
                f"({len(older_turns)} turns)"
            )

            return summary.strip()

        except Exception as exc:

            app_logger.error(
                f"Summariser failed for "
                f"{session_id}: {exc!r}"
            )

            return ""


conversation_summariser = ConversationSummarizer()
