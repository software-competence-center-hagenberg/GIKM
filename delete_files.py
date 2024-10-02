import os

def delete_prefixed_files(directory, prefix="._"):
    # Walk through all subdirectories and files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                # Get the full path of the file
                file_path = os.path.join(root, file)
                try:
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# Example usage
directory = "./Datasets/FreiburgGrocery/"
delete_prefixed_files(directory)