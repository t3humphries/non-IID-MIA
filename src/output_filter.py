import re

class OutputFilter(object):
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        # data = re.sub(r'{.+}', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*Theano cache: .+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*(WARNING|ERROR) .+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*NoneType: .+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*tmp.+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*Elemwise.+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*\[(TensorType|InplaceDimShuffle).+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*<class.+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*{.+}', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.*\d+\.\d+s', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'^=+$', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'^\-+$', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'^\++$', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'^\++$', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'.+(I|W|E) tensorflow.+', '', data, flags=re.IGNORECASE|re.MULTILINE)
        data = re.sub(r'^ $', '', data, flags=re.IGNORECASE|re.MULTILINE)

        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.stream.flush()