from src.domain.models import CandidateProfile, HumanFeedback, MatchReport, SearchAttempt
from src.memory.models import MemoryRecord, MemorySourceType, MemoryType


def memory_from_candidate_profile(candidate_profile: CandidateProfile) -> MemoryRecord:
    skills = ", ".join(candidate_profile.skills) if candidate_profile.skills else "no listed skills"
    name = candidate_profile.name or candidate_profile.candidate_id
    return MemoryRecord(
        memory_type=MemoryType.SEMANTIC.value,
        source_type=MemorySourceType.CANDIDATE_PROFILE.value,
        source_id=candidate_profile.candidate_id,
        content=f"Candidate {name} has skills: {skills}.",
        importance=0.6,
        tags=["candidate", "skills"],
        metadata={"candidate_id": candidate_profile.candidate_id},
    )


def memory_from_match_report(match_report: MatchReport) -> MemoryRecord:
    return MemoryRecord(
        memory_type=MemoryType.EPISODIC.value,
        source_type=MemorySourceType.MATCH_REPORT.value,
        source_id=match_report.match_id,
        content=(
            f"Candidate {match_report.candidate_id} matched job {match_report.job_id} "
            f"with score {match_report.total_score} and recommendation {match_report.recommendation}."
        ),
        importance=0.7 if match_report.total_score >= 80 else 0.5,
        tags=["match", "score", match_report.recommendation.lower()] if match_report.recommendation else ["match", "score"],
        metadata={
            "job_id": match_report.job_id,
            "candidate_id": match_report.candidate_id,
            "total_score": match_report.total_score,
        },
    )


def memory_from_human_feedback(human_feedback: HumanFeedback) -> MemoryRecord:
    return MemoryRecord(
        memory_type=MemoryType.PREFERENCE.value,
        source_type=MemorySourceType.HUMAN_FEEDBACK.value,
        source_id=human_feedback.feedback_id,
        content=f"Human feedback on task {human_feedback.task_id}: {human_feedback.feedback_type} {human_feedback.payload}.",
        importance=0.8,
        tags=["feedback", human_feedback.feedback_type] if human_feedback.feedback_type else ["feedback"],
        metadata={
            "task_id": human_feedback.task_id,
            "target_type": human_feedback.target_type,
            "target_id": human_feedback.target_id,
        },
    )


def memory_from_search_attempt(search_attempt: SearchAttempt) -> MemoryRecord:
    candidates = ", ".join(search_attempt.retrieved_candidate_ids) or "none"
    return MemoryRecord(
        memory_type=MemoryType.EPISODIC.value,
        source_type=MemorySourceType.SEARCH_ATTEMPT.value,
        source_id=search_attempt.search_id,
        content=f"Search query '{search_attempt.query}' retrieved candidates: {candidates}.",
        importance=0.4,
        tags=["search", "query"],
        metadata={
            "job_id": search_attempt.job_id,
            "retrieved_candidate_ids": search_attempt.retrieved_candidate_ids,
            "retrieved_resume_ids": search_attempt.retrieved_resume_ids,
        },
    )
