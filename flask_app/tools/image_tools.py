try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import matplotlib.pyplot as plt


def image_array_to_string(arr, alpha=0.5, cmap=''):
    plt.imshow(arr, alpha=alpha, cmap=cmap)
    plt.axis('off')
    output = StringIO()
    plt.savefig(output)
    return output.getvalue()


def get_file_storage(arr, filename=None, **kwargs):
    from werkzeug.datastructures import FileStorage
    return FileStorage(stream=image_array_to_string(arr, **kwargs), filename=filename, content_type='image/png')
