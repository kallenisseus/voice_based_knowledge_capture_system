def split_sliding_words(text, winsize=160, overlap=40):
    words = text.split()
    if(len(words) <= winsize):
        return [text]

    step = winsize - overlap
    windows = []

    for i in range(0, len(words), step):
        chunk = words[i:i + winsize]
        if chunk:
            windows.append(" ".join(chunk))

    if  len(windows[-1].split()) < winsize:
        last_win = " ".join(words[-winsize:])
        windows[-1] = last_win
    return windows

def split_all(strs, winsize=160, overlap=40):
    res = []
    for str in strs:
        res += split_sliding_words(str, winsize, overlap)
    return res