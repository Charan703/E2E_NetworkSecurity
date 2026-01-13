import os

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Sync a local folder to an S3 bucket folder.

        Args:
            folder_path (str): Local folder path to sync.
            bucket_name (str): Name of the S3 bucket.
            s3_folder (str): Target folder in the S3 bucket.
        """
        commmand = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(commmand)
    
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Sync an S3 bucket folder to a local folder.

        Args:
            folder_path (str): Local folder path to sync.
            bucket_name (str): Name of the S3 bucket.
            s3_folder (str): Source folder in the S3 bucket.
        """
        commmand = f"aws s3 sync {aws_bucket_url} {folder}"
        os.system(commmand)
