def test_rouge():
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"])
    scores = scorer.score(prediction="hello world!!!", target="hello ")
    result = scores["rougeL"].fmeasure
    assert abs(result - 0.66) < 0.01, result
