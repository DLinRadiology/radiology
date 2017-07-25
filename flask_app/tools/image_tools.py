import io
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from werkzeug.datastructures import FileStorage
import matplotlib.pyplot as plt


def image_array_to_string(arr, alpha=0.5, cmap=''):
    plt.imshow(arr, alpha=alpha, cmap=cmap)
    plt.axis('off')
    # output = StringIO()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    return buf


def get_file_storage(arr, filename=None, **kwargs):
    return FileStorage(stream=image_array_to_string(arr, **kwargs), filename=filename, content_type='image/jpeg')
