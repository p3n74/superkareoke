from src.database.models import Performance


def compute_star_rating(score_percent: float) -> int:
    if score_percent >= 95:
        return 5
    if score_percent >= 85:
        return 4
    if score_percent >= 70:
        return 3
    if score_percent >= 50:
        return 2
    if score_percent >= 25:
        return 1
    return 0


def build_performance(
    song_id: int,
    score_percent: float,
    perfect_count: int,
    great_count: int,
    good_count: int,
    ok_count: int,
    miss_count: int,
) -> Performance:
    return Performance(
        song_id=song_id,
        score_percent=round(score_percent, 2),
        star_rating=compute_star_rating(score_percent),
        perfect_count=perfect_count,
        great_count=great_count,
        good_count=good_count,
        ok_count=ok_count,
        miss_count=miss_count,
    )
