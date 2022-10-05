import numpy as np

def sliding_window(x, win, stride):
    # x: (t v)
    array = np.arange(x.shape[0])
    shape = (array.shape[0] - win + 1, win)
    strides = (array.strides[0], ) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape, strides)
    index = rolled[np.arange(0, shape[0], stride)]
    return x[index]


def time_span(x):
    y = np.roll(x,1)
    starts = np.where(y-x)[0]
    starts = np.concatenate([[0], starts, [len(x)]])
    spans = [(starts[i], starts[i+1]) for i in range(len(starts)-1)]
    labels = [x[i] for i, _ in spans]
    return zip(spans, labels)

def span_to_array(span_value):
    all_arrays = []
    for span, value in span_value:
        array = np.repeat(np.array([value]), span[1]-span[0])
        all_arrays.append(array)
    
    fix_encoding = np.concatenate(all_arrays)
    return fix_encoding



if __name__ == '__main__':
    from datetime import timedelta
    t1 = timedelta(hours=9).seconds
    t2 = timedelta(hours=17).seconds
    span_value = zip([(0, t1),(t1, t2), (t2,24*60*60)], [1, 0, 1])
    
    fix_encoding = span_to_array(span_value)
    import pandas as pd

    fix_encoding = pd.DataFrame(fix_encoding)
    print(fix_encoding)