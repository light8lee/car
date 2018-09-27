import re

def load_matchers_str(filepath):
    with open(filepath, encoding='utf-8') as f:
        words = [line.strip() for line in f.readlines()]
    return words

def load_matchers_re(filepath):
    with open(filepath, encoding='utf-8') as f:
        mather = [re.compile(line.strip()) for line in f.readlines()]
    return mather

def match_subject(str_path=None, re_path=None):
    if str_path:
        str_matchers = load_matchers_str(str_path)
    else:
        str_matchers = []

    if re_path:
        re_matchers = load_matchers_re(re_path)
    else:
        re_matchers = []

    def _match(text):
        for matcher in str_matchers:
            if text.find(matcher) != -1:
                return True
        for matcher in re_matchers:
            if matcher.search(text):
                return True
        return False
    return _match