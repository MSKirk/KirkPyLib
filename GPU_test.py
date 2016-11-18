from numbapro import vectorize

@vectorize(['float32(float32, float32)'], target='gpu')
def sum(a, b):
    return a + b