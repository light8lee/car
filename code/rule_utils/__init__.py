import re

def load_matchers_re(filepath):
    with open(filepath, encoding='utf-8') as f:
        mather = [re.compile(line.strip()) for line in f.readlines()]
    return mather

def match_subject(re_path):
    if re_path:
        re_matchers = load_matchers_re(re_path)
    else:
        re_matchers = []

    def _match(text):
        for matcher in re_matchers:
            if matcher.search(text):
                return True
        return False
    return _match