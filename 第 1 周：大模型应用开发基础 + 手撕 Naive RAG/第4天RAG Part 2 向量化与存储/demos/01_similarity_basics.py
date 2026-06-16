import math


def dot(a: list[float], b: list[float]) -> float:
    """Calculate inner product."""
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    """Calculate vector length."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity. Higher means more similar."""
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return 0.0
    return dot(a, b) / denominator


def l2_distance(a: list[float], b: list[float]) -> float:
    """Calculate Euclidean distance. Lower means more similar."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def print_ranking(title: str, rows: list[tuple[str, float]], reverse: bool) -> None:
    print(f"\n{title}")
    for rank, (name, score) in enumerate(
        sorted(rows, key=lambda item: item[1], reverse=reverse),
        start=1,
    ):
        print(f"rank={rank}, name={name}, score={score:.4f}")


def main() -> None:
    query = [1.0, 0.0]

    docs = {
        "doc_python": [0.9, 0.1],
        "doc_fastapi": [0.7, 0.3],
        "doc_music": [0.0, 1.0],
        "doc_long_vector": [9.0, 1.0],
    }

    cosine_rows = [
        (name, cosine_similarity(query, vector))
        for name, vector in docs.items()
    ]
    l2_rows = [
        (name, l2_distance(query, vector))
        for name, vector in docs.items()
    ]
    dot_rows = [
        (name, dot(query, vector))
        for name, vector in docs.items()
    ]

    print("query:", query)
    print("docs:", docs)
    print_ranking("Cosine similarity ranking: higher is better", cosine_rows, reverse=True)
    print_ranking("L2 distance ranking: lower is better", l2_rows, reverse=False)
    print_ranking("Inner product ranking: higher is better", dot_rows, reverse=True)


if __name__ == "__main__":
    main()

