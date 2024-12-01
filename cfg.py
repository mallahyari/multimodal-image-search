class Config:
    # Define a class named Config to hold configuration settings.
    def __init__(self):
        # Initialize method to set default values for configuration settings.
        self.download_path = "./images_folder"
        # Set the path where images will be downloaded to "./images".
        self.upload_path = "./uploaded_images"
        # Set the path where images will be uploaded to "./uploaded_images".
        self.milvus_uri = "image_search_demo.db"
        # Set the URI for the Milvus database, you can change to "http://localhost:19530" for a standard Milvus.
        self.collection_name = "multimodal_image_search"
        # Define the name of the collection in the Milvus database, set to "cir_demo_large".
        self.device = "cpu"
        # Set the device to use for computations, in this case, "gpu", you can change it to "cpu".
        self.model_name = "BAAI/bge-base-en-v1.5"
        # Specify the type of model to use.
        self.model_path = "./model/Visualized_base_en_v1.5.pth"
        # Set the path to the model file, default is "./Visualized_base_en_v1.5.pth".
        self.embedding_dimensions = 768
        # Set the dimension of the embeddings, default is 768.
        