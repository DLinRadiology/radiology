import io
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from werkzeug.datastructures import FileStorage


def image_array_to_string_mpl(arr, alpha=0.5, cmap=''):
    import matplotlib.pyplot as plt
    plt.imshow(arr, alpha=alpha, cmap=cmap)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    return buf


def image_array_to_string(arr, **kwargs):
    image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(format="jpeg", fp=buf)
    buf.seek(0)
    return buf


def get_file_storage(arr, filename=None, **kwargs):
    return FileStorage(stream=image_array_to_string(arr, **kwargs), filename=filename, content_type='image/jpeg')
