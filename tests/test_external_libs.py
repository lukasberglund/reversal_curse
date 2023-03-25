
def test_rouge():
    from rouge import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    scorer.score(prediction='hello world!!!', target='hello ')