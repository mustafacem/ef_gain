from selfquant.data.calibration import ngram_overlap_ratio, text_list_hash


def test_ngram_overlap_ratio_zero_for_disjoint_text():
    calib = ["the quick brown fox jumps over the lazy dog and runs away fast"]
    eval_texts = ["completely different sentence with no shared vocabulary chunks here"]
    assert ngram_overlap_ratio(calib, eval_texts, n=4) == 0.0


def test_ngram_overlap_ratio_one_for_identical_text():
    text = ["the quick brown fox jumps over the lazy dog and runs away fast today"]
    assert ngram_overlap_ratio(text, text, n=4) == 1.0


def test_ngram_overlap_ratio_partial():
    calib = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
    eval_texts = ["alpha beta gamma delta epsilon completely unrelated tail words here"]
    ratio = ngram_overlap_ratio(calib, eval_texts, n=4)
    assert 0.0 < ratio < 1.0


def test_text_list_hash_deterministic_and_sensitive():
    a = ["hello world", "foo bar"]
    b = ["hello world", "foo bar"]
    c = ["hello world", "foo baz"]
    assert text_list_hash(a) == text_list_hash(b)
    assert text_list_hash(a) != text_list_hash(c)
