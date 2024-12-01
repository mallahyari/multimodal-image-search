import os
import logging
from fasthtml.common import *
from shad4fast import * 
from milvus_index import encoder
from pymilvus import MilvusClient
from pathlib import Path
from cfg import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


app, rt = fast_app(pico=False, hdrs=(ShadHead(tw_cdn=True),), live=True)

config = Config()
milvus_client = MilvusClient(uri=config.milvus_uri)
logger.info(milvus_client.get_collection_stats(config.collection_name))

# Directory to store uploaded images
Path(config.upload_path).mkdir(parents=True, exist_ok=True)


@app.route("/")
def get():
    # Form for both file upload and text search
    add = Form(
        Group(
            Div(
                Label("Upload Image", for_="myFile", cls="text-sm font-semibold text-gray-700 mb-2"),
                Input(id="myFile", type="file", name="myFile", cls="block w-full border border-gray-300 rounded-lg p-2 mb-4 shadow-sm"),
                cls="flex flex-col mb-4"
            ),
            Div(
                Label("Search Query", for_="search_query", cls="text-sm font-semibold text-gray-700 mb-2"),
                Input(type="text", placeholder="Enter your query...", name="search_query", id="search_query", cls="block w-full border border-gray-300 rounded-lg p-2 mb-4 shadow-sm"),
                cls="flex flex-col mb-4"
            ),
            Div(
                Button(
                    "Search",
                    type="submit",
                    cls="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition duration-200 shadow"
                ),
                cls="flex justify-center"
            ),
        ),
        hx_post="/search",  
        target_id="img_ids",
        hx_swap="innerHTML",
        cls="w-full bg-white p-6 rounded-lg shadow-lg border border-gray-200"
    )

    # Card containing the form
    card = Card(
        title="Find Your Perfect Image",
        description="Upload an image, enter a text query, or both to search for similar images.",
        footer=Div(
            add,
            cls="w-full"
        ),
        cls="w-[80%] mx-auto bg-gray-50 p-8 rounded-lg shadow-xl border border-gray-300"
    )

    image_list = Div(
        id="img_ids",
        cls="grid grid-cols-1 md:grid-cols-1 lg:grid-cols-1 gap-1 mt-8"
    )

    title = "Advanced Image Search"
    return Title(title), Main(
        Div(
            H1(title, cls="text-3xl font-bold text-gray-800 mb-6 text-center"),
            card,
            image_list,
            cls="container mx-auto p-6"
        )
    )



@app.route("/search")
async def post(request: Request, myFile: UploadFile = None, search_query: str = Form(None)):
    img_paths = []
    logger.info(f"=======Search query received:{search_query}")
    
    # Check for uploaded file
    if myFile is not None:
            print("File received:", myFile.filename)
            # Save the uploaded file
            uploaded_file_path = os.path.join(config.upload_path, f"uploaded_{myFile.filename}")
            with open(uploaded_file_path, "wb") as file:
                file_contents = await myFile.read()
                file.write(file_contents)
                # Use the stored file for embedding
                print(f"File saved at: {uploaded_file_path}")
            if search_query:
                print("Search query received:", search_query)
                img_paths = search(image_path=uploaded_file_path, search_query=search_query)
            else:
                logger.info("No search query received.")
                img_paths = search(image_path=uploaded_file_path)
                
    # Check for text query
    elif search_query:
        print("Search query received:", search_query)
        img_paths = search(search_query=search_query)
    
    
    # Default case if no inputs are provided
    if not img_paths:
        print("No input provided or no results found")
        img_paths = []  # Return an empty list or handle it appropriately

    img_holders = [
        Div(
            Card(
                Img(src=image_path, alt="Card image", cls="w-full h-48 object-cover rounded-t-lg"), 
                title=f"Image {index + 1}", 
                description=f"Score: {score:.3f}" if isinstance(score, float) else "No description",
                cls="bg-white shadow-lg rounded-lg overflow-hidden flex flex-col items-center"
            ), 
            cls="p-3"
        ) 
        for index, (image_path, score) in enumerate(img_paths)  # Assuming img_paths includes scores
    ]
    
    grid = Div(
        *img_holders, 
        cls="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 px-4"
    )
    
    return H1("Search Results", cls="text-3xl ms-4 ps-4"), grid



def search(image_path: str = None, search_query: str = None):
    """
    Performs multimodal vector similarity search using image and/or text input.
    
    Args:
        image_path (str, optional): Path to the uploaded image file
        search_query (str, optional): Text query for semantic search
        
    Returns:
        list[tuple]: List of tuples containing (image_path, distance_score)
                     where distance_score represents cosine similarity (lower is better)
                     
    Example:
        results = search(image_path="uploads/cat.jpg", search_query="cute pets")
        results = search(search_query="mountain landscape")
        results = search(image_path="uploads/food.jpg")
    """

    query_vec = encoder.encode_query(image_path=image_path, text=search_query)
        
    search_results = milvus_client.search(
        collection_name=config.collection_name,
        data=[query_vec],
        output_fields=["image_path"],
        limit=10,
        search_params={"metric_type": "COSINE", "params": {}},
    )[0]

    retrieved_images = [(hit.get("entity").get("image_path"), hit.get("distance")) for hit in search_results]
    return retrieved_images

serve(port=8000)
