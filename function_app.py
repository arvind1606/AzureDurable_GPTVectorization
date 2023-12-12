import azure.functions as func
import azure.durable_functions as df

###########################################################
# imports
from azure.storage.blob import BlobServiceClient, ContainerClient
import yaml
from yaml.loader import SafeLoader
import os
import openai
import tempfile
import time
import shutil
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import FAISS
import logging
from langchain.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)
logger.info("Check")





####################################################################################################
# reading the config params
with open(r'devConfig.yml') as f:
    configParser = yaml.load(f, Loader=SafeLoader)

# Setting the config params
os.environ["OPENAI_API_TYPE"] = configParser['OPENAI_API_TYPE']
os.environ["OPENAI_API_KEY"] = configParser['OPENAI_API_KEY']
os.environ["OPENAI_API_BASE"] = configParser['OPENAI_API_BASE']
os.environ["OPENAI_API_VERSION"] = configParser['OPENAI_API_VERSION']
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base =  os.environ["OPENAI_API_BASE"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"] # this may change in the future
Model_deployment_GPT4= configParser['Model_deployment_GPT4']
embedding_model = configParser['embedding_model_deployment']

connection_string = configParser['CONNECTION_STRING'] 
# container_name = configParser['CONTAINER_NAME']
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# container_client = blob_service_client.get_container_client(container_name)
storage_account_name = configParser['storage_account_name']
storage_account_key = configParser['storage_account_key']
Model_name_GPT4= configParser['Model_name_GPT4']

##############################################################
# defining embedding
embeddings=OpenAIEmbeddings(deployment=embedding_model,
                                model=embedding_model,
                                openai_api_base=openai.api_base,
                                openai_api_type="azure",
                                chunk_size=1)

# defining llm
llm = AzureChatOpenAI(deployment_name=Model_deployment_GPT4,
                      model_name=Model_deployment_GPT4,
                      openai_api_base=openai.api_base,
                      openai_api_version=openai.api_version,
                      openai_api_key=openai.api_key,
                      openai_api_type="azure",
                     temperature = 0.5)
##############################################################

def read_file_file_from_blob(blob_service_client, container_name, blob_name):
    print(f"=========read_file_file_from_blob========={blob_service_client} | {container_name} | {blob_name}")
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob()
    file_content = blob_data.readall()
    # print("**************************")
    return file_content

#### function to list all the files in the folder onblob storage
def list_files_in_folder(folder_name, container_name):
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)    
    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)    
    # List all blobs in the specified folder 
    blob_list = container_client.list_blobs(name_starts_with=folder_name)
    files_in_folder = []
    for blob in blob_list:
        files_in_folder.append(blob.name)
    return files_in_folder


#### defining the text splitter 
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=20000, chunk_overlap=500
)

#################################################################

# func to create folder
import os
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f'Folder "{folder_path}" created.')
    else:
        logger.info(f'Folder "{folder_path}" already exists.')

def is_pdf_file(file_path):
        _, file_extension = os.path.splitext(file_path.lower())
        return file_extension == ".pdf"

import PyPDF2
from pypdf import PdfReader
from pypdf.errors import PdfReadError

############ check if pdf file is valid
def is_valid_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            _ = reader.getDocumentInfo()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def validate_pdf(folder_name):
    temp_dir = "."
    curr_folder = folder_name

    curr_dir =  os.path.join(temp_dir,folder_name)
    files = os.listdir(curr_dir)
    # print(files)
    for file in files:
        file_path = os.path.join(curr_dir,file)
        # print(">>>>>>>>>>>>")
        # print(file_path)
        try:
            PdfReader(file_path)
        except PdfReadError:
            print(f"invalid PDF file, so deleting >> {file_path}")
            os.remove(file_path)
        else:
            pass
#####################################################################
import re

def remove_special_characters_except_period(text):
    # Replace or remove special characters except for the period
    return re.sub(r'[^\w\s.-]', '', text)
#####################################################################
def download_file_from_blob(container_name, file_path, temp_dir):
    file_content = read_file_file_from_blob(blob_service_client, container_name, file_path)

    file_path = file_path.split("/")[-1]
    # Get the last 200 characters using negative indexing
    # file_path = remove_special_characters_except_period(file_path)
    # file_path_last_100_characters = file_path[-100:]
 
    local_file_path = os.path.join(temp_dir,file_path)    
    # this is just to skip a file where file name len is more than 225
    # as blob supports max 256 char name max
    # Also note some times it gives error if file name has some special char like ":" in file name. Need to add that condition chk as well
    if len(file_path) < 225:        
        print(f"Writing >>>> {local_file_path}")
        with open(local_file_path, "wb") as local_file:
                local_file.write(file_content)
        return temp_dir


# function to upload folder in blob
def upload_folder_to_blob(blob_service_client, container_name, local_folder, blob_folder):
    container_client = blob_service_client.get_container_client(container_name)

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob_file_path = os.path.join(blob_folder, os.path.relpath(local_file_path, local_folder)).replace("\\", "/")

            with open(local_file_path, "rb") as data:
                container_client.upload_blob(name=blob_file_path, data=data, overwrite=True)

# function to delete folder from blob
def delete_folder_from_blob(blob_service_client, container_name, folder_prefix):
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container with the specified prefix
    blobs = container_client.list_blobs(name_starts_with=folder_prefix)

    # Delete each blob in the folder
    for blob in blobs:
        container_client.delete_blob(blob.name)
        # print(f"Blob '{blob.name}' deleted.")

    print(f"Folder '{folder_prefix}' deleted.")
#####################################################################

import tiktoken
# func to count tockens in the string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def create_vector_db(folder_name):
    """Creates a FAISS vector database from a folder of PDF documents."""
    # Record the start time
    start_time = time.time()

    vec_db = None
    faiss_idx_folder = folder_name+"_faiss"

    # Create FAISS index folder
    os.makedirs(faiss_idx_folder, exist_ok=True)

    # Load and split documents
    loader = PyPDFDirectoryLoader(folder_name)
    processed_pages = 0
    total_token_processed = 0

    documents = loader.load_and_split(text_splitter=text_splitter)
    for document in documents:
        processed_pages += 1
        print(f"Processing {processed_pages}/{len(documents)}")
        token_count = num_tokens_from_string(str(document), "cl100k_base")
        total_token_processed = total_token_processed + token_count
        print(f"    total_token_processed: {total_token_processed}")
        if total_token_processed > 110000:
            total_token_processed = 0
            print("        >>> Sleeping for 60 secs")
            time.sleep(60)  

        if vec_db is None:
            vec_db = FAISS.from_documents(documents=[document], embedding=embeddings)
        else:
            vec_db.add_documents([document])       

    # Save the FAISS index
    vec_db.save_local(faiss_idx_folder)

    # Calculate and print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Vectorization completed in {elapsed_time:.2f} seconds for {processed_pages} pages.")

    return faiss_idx_folder

################# Durable Func ######################################

#####################################################################
myApp = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# An HTTP-Triggered Function with a Durable Functions Client binding
@myApp.route(route="orchestrators/{functionName}")
@myApp.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client):
    try:
        req_data = req.get_json()

        if req_data is None or 'folder_name' not in req_data or 'container_name' not in req_data:
            return func.HttpResponse("Invalid request data", status_code=400)

        folder_name = req_data['folder_name']
        container_name = req_data['container_name']
        payload = {"folder_name": folder_name, "container_name":container_name}

        function_name = req.route_params.get('functionName')
        instance_id = await client.start_new(function_name, None, payload)
        response = client.create_check_status_response(req, instance_id)

        return response
    except Exception as e:
        return func.HttpResponse(f"Internal Server Error: {e}", status_code=500)

# Orchestrator
@myApp.orchestration_trigger(context_name="context")
def vectorization_orchestrator(context):
    input_context = context.get_input()
    folder_name = input_context.get('folder_name')
    container_name = input_context.get('container_name')
    tasks = []

    file_list = list_files_in_folder(folder_name, container_name)
    print(f"file_list >> ---{file_list}----")

    temp_dir = tempfile.mkdtemp()

    for file_name in file_list:
        if is_pdf_file(file_name):
            # file_name = file_name.split("/")[-1]
            params = {"container_name": container_name, "folder_name": folder_name, "file_name": file_name, "temp_dir": temp_dir}
            print(f"params: {params}")
            tasks.append(context.call_activity("vectorization_func", params))

    results = yield context.task_all(tasks)

    folder_name_of_pdf = results[0]

    # Lets Validate all the pdfs, if any pdf is invalid, it will be deleted.
    validate_pdf(folder_name_of_pdf)

    # Now pdf files are downloaded, we can crete vector DB now
    faiss_idx_folder = create_vector_db(folder_name_of_pdf) 

    # delete a blob if already exist
    delete_folder_from_blob(blob_service_client, container_name, folder_name+"_faiss")

    # uploading the faiss vector store to blob
    upload_folder_to_blob(blob_service_client, container_name, faiss_idx_folder, folder_name+"_faiss")

    return faiss_idx_folder

# Activity
@myApp.activity_trigger(input_name="params")
def vectorization_func(params: dict):

    container_name = params.get("container_name")
    folder_name = params.get("folder_name")
    file_name = params.get("file_name")    
    temp_dir = params.get("temp_dir")

    curr_file_dwnld_folder =  os.path.join(temp_dir, folder_name)
    # os.makedirs(curr_file_dwnld_folder)
    try:
        os.makedirs(curr_file_dwnld_folder)
        os.makedirs(curr_file_dwnld_folder+"_faiss")
        logger.info(f"Folder '{folder_name}' created inside '{temp_dir}'.")
    except FileExistsError:
        indx_already_there_flag = True
        logger.info(f"Folder '{folder_name}' already exists inside '{temp_dir}'.") 

    tmp_folder_path = download_file_from_blob(container_name, file_name, curr_file_dwnld_folder)

    return tmp_folder_path