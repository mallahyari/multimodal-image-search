import torch
import os
import logging
from visual_bge.modeling import Visualized_BGE
from tqdm import tqdm
from glob import glob
from pymilvus import MilvusClient
from cfg import Config

config = Config()



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Encoder:
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def generate_embeddings():
    data_dir = config.download_path
    image_list = glob(os.path.join(data_dir, "images", "*.jpg"))  # We will only use images ending with ".jpg"
    image_dict = {}
    for image_path in tqdm(image_list, desc="Generating image embeddings: "):
        try:
            image_dict[image_path] = encoder.encode_image(image_path)
        except Exception as e:
            print(f"Failed to generate embedding for {image_path}. Skipped.")
            continue
    logger.info(f"Number of encoded images:{len(image_dict)}")
    return image_dict

def insert_to_milvus():
    image_dict = generate_embeddings()
    dim = len(list(image_dict.values())[0])
    assert dim == config.embedding_dimensions, f"Expected dimension {config.embedding_dimensions}, got {dim}"
    milvus_client = MilvusClient(uri=config.milvus_uri)

    milvus_client.create_collection(
        collection_name=config.collection_name,
        auto_id=True,
        dimension=dim,
        enable_dynamic_field=True,
    )

    inserted_records = milvus_client.insert(
        collection_name=config.collection_name,
        data=[{"image_path": k, "vector": v} for k, v in image_dict.items()],
    )
    print(f"records inserted into Milvus: {inserted_records['insert_count']}")

model_name = config.model_name
model_path = config.model_path
encoder = Encoder(model_name, model_path)

# Call the function to insert embeddings into Milvus. This will create the collection and insert the embeddings.
# insert_to_milvus()