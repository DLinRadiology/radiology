import os

from google.cloud import storage


class Storage(object):
    def __init__(self, project=None, credentials=None):
        self.gcs = storage.Client(project=project, credentials=credentials)  # Create a Cloud Storage client.
        self.buckets = dict()

    def __getitem__(self, bucket_name):
        """
        Parameters
        ----------
        bucket_name : str

        Returns
        -------
        bucket : Bucket
        """
        if bucket_name not in self.buckets:
            self.buckets[bucket_name] = Bucket(gcs=self.gcs, bucket_name=bucket_name)
        return self.buckets[bucket_name]


class Bucket(object):
    def __init__(self, gcs, bucket_name):
        self.bucket = gcs.get_bucket(bucket_name)   # Get the bucket that the file will be uploaded to.

    def upload_from_string(self, uploaded_file, folder=None, file_name=None):
        file_path = file_name or uploaded_file.filename
        if folder is not None:
            file_path = os.path.join(folder, file_path)
        blob = self.bucket.blob(file_path)
        blob.upload_from_string(
            uploaded_file.read(),
            content_type=uploaded_file.content_type
        )
        return blob.public_url

    def upload_from_filename(self, filepath, content_type, folder=None, file_name=None):
        file_path = file_name or os.path.basename(filepath)
        if folder is not None:
            file_path = os.path.join(folder, file_path)
        blob = self.bucket.blob(file_path)
        blob.upload_from_filename(
            filename=filepath,
            content_type=content_type
        )
        return blob.public_url

    def copy(self, from_, to_):
        blob = self.bucket.get_blob(from_)
        self.bucket.copy_blob(blob, self.bucket, to_)
