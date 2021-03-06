import os
import logging
import datetime

from flask import Flask, request, redirect, url_for

from tools.storage import Storage
from tools.image_tools import get_file_storage
from models.frontal_lateral import ModelFrontalLateral
from models.heart_segmentation import ModelHearthSegmentation


app = Flask(__name__)


try:
    # Configure this environment variable via app.yaml
    CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
except KeyError:
    CLOUD_STORAGE_BUCKET = 'tf2bucket'
storage = Storage('wired-plateau-167712')[CLOUD_STORAGE_BUCKET]

TEMP_FOLDER_NAME = 'temp'


@app.route('/')
def index():
    # <li><a href="/test">test</a></li>
    return """
    <ul>
        <li><a href="/frontal_lateral">classify chest x-ray to frontal or lateral</a></li>
        <li><a href="/heart_segmentation">segment heart on frontal chest x-ray</a></li>
    </ul>
    """


@app.route('/test')
def test():
    return redirect(url_for('test_url', url1='a', url2='b'))


@app.route('/test/<url1>/<url2>')
def test_url(url1, url2):
    return '{}-{}'.format(url1, url2)


@app.route('/frontal_lateral')
def frontal_lateral():
    return """
    <form method="POST" action="/upload_frontal_lateral" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    """


@app.route('/upload_frontal_lateral', methods=['POST'])
def upload_frontal_lateral():
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return 'No file uploaded.', 400

    model = ModelFrontalLateral()
    result = model.predict(uploaded_file)
    return "it's a {} image".format('frontal' if result == 1 else 'lateral')


@app.route('/heart_segmentation')
def heart_segmentation():
    return """
    <form method="POST" action="/upload_heart_segmentation" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    """


@app.route('/upload_heart_segmentation', methods=['POST'])
def upload_heart_segmentation():
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return 'No file uploaded.', 400

    model = ModelHearthSegmentation()
    orig, pred = model.predict(uploaded_file)

    # SAVE ORIGINAL AND PREDICTED IMAGES
    now = datetime.datetime.now().isoformat()
    orig_fn = 'orig_{}.jpg'.format(now)
    pred_fn = 'pred_{}.jpg'.format(now)
    orig_fs = get_file_storage(orig, filename=orig_fn, cmap='Greys_r')
    pred_fs = get_file_storage(pred, filename=pred_fn, cmap='Greys_r')

    # orig_fs.save('/Users/Edu/Temp/orig.png')
    # pred_fs.save('/Users/Edu/Temp/pred.png')

    orig_url = storage.upload_from_string(orig_fs, folder=TEMP_FOLDER_NAME)
    pred_url = storage.upload_from_string(pred_fs, folder=TEMP_FOLDER_NAME)

    # app.logger.info(orig_url)
    # app.logger.info(pred_url)

    return """
    <form method="POST" action="/upload_heart_segmentation_done">
        <img src="{url1}" alt="Original">
        <img src="{url2}" alt="Predicted">
        <input type="hidden" value="{orig_fn}" name="orig_fn" />
        <input type="hidden" value="{pred_fn}" name="pred_fn" />
        <br/> 
        <label for="good">Good</label>
        <input type="radio" name="gender" id="good" value="good"><br>
        <label for="ok">Somewhat OK</label>
        <input type="radio" name="gender" id="ok" value="ok"><br>
        <label for="bad">Bad</label>
        <input type="radio" name="gender" id="bad" value="bad"><br><br>
        <input type="submit" value="Submit">
    </form>
    """.format(
        url1=orig_url,
        url2=pred_url,
        orig_fn=orig_fn,
        pred_fn=pred_fn
    )


@app.route('/upload_heart_segmentation_done', methods=['POST'])
def upload_heart_segmentation_done():
    save_folder = dict(
        good="heartseg_good",
        ok="heartseg_ok",
        bad="heartseg_bad"
    )[request.form['gender']]

    orig_fn = request.form['orig_fn']
    pred_fn = request.form['pred_fn']
    for fn in [orig_fn, pred_fn]:
        temp_path = '{}/{}'.format(TEMP_FOLDER_NAME, fn)
        copy_path = '{}/{}'.format(save_folder, fn)
        try:
            storage.copy(temp_path, copy_path)
        finally:
            try:
                storage.delete(temp_path)
            except Exception:
                pass
    return """
    Thank you for your help!<br>
    <ul>
        <li><a href="/..">Go back to first page</a></li>
        <li><a href="https://dlinradiology.wordpress.com">dlinradiology.wordpress.com</a></li>
    </ul>
    """


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
