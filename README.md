# AzureDurable_GPTVectorization
Here is an illustration demonstrating the vectorization of multiple PDF files retrieved from Azure Blob storage, incorporating mechanisms to manage Rate Limit errors.

#Vectorization Orchestrator
This code defines an orchestrator function named vectorization_orchestrator within the context of the specified application. The orchestrator is triggered by an orchestration context named "context." The main purpose of this orchestrator is to manage the vectorization process for multiple PDF files sourced from Azure Blob storage, while also addressing rate-limiting errors.

## Code Walkthrough
Input Retrieval:

Retrieve input parameters from the orchestration context, including the folder_name and container_name.
File Listing:

Obtain a list of files in the specified folder within the given Azure Blob container using the list_files_in_folder function.
Print the list of files for verification.
Temporary Directory Setup:

Create a temporary directory (temp_dir) for storing downloaded PDF files.
# PDF Vectorization Tasks:

Iterate through the list of files, identifying PDF files using the is_pdf_file function.
For each PDF file, create a set of parameters and append a vectorization activity task to the tasks list.
Task Execution:

Execute all vectorization tasks concurrently using the task_all method within the orchestration context.
PDF Validation:

Retrieve the folder name of PDF files from the task results.
Validate all downloaded PDFs, removing any invalid files.
Vector Database Creation:

Create a vector database (faiss_idx_folder) using the validated PDF files.
Blob Operations:

Delete any existing blob folder with the same name (folder_name+"_faiss").
Upload the faiss vector store to Azure Blob storage under the specified container.
Return Result:

Return the folder path of the faiss vector store.
Vectorization Activity
This section defines an activity function named vectorization_func triggered by an input named "params." The activity is responsible for downloading a specific PDF file from Azure Blob storage, creating a temporary directory structure, and returning the temporary folder path.

## Code Walkthrough
Parameter Extraction:

Extract parameters such as container_name, folder_name, file_name, and temp_dir from the input dictionary.
Temporary Folder Setup:

Create a temporary download folder within the specified temp_dir.
Download PDF File:

Download the specified PDF file from Azure Blob storage to the temporary folder.
Return Result:

Return the path of the temporary folder containing the downloaded PDF file.

# Rate limit handling

The create_vector_db function is responsible for generating a FAISS (Facebook AI Similarity Search) vector database from a folder containing PDF documents. Additionally, the function incorporates a mechanism to handle rate limit errors by introducing a sleep period of 60 seconds when the total token count processed exceeds a specified threshold.

Function Overview:
Initialization:

Record the start time to measure the duration of the vectorization process.
Set up the FAISS index folder (faiss_idx_folder) by creating it if it doesn't exist.
Document Loading:

Use a PyPDFDirectoryLoader to load and split the text of the PDF documents in the specified folder.
Initialize variables to track the number of processed pages (processed_pages) and the total number of tokens processed (total_token_processed).
Vectorization Process:

Iterate through each document obtained from the PDFs.
For each document:
Increment the processed_pages count.
Calculate the token count using the specified text splitter (text_splitter) and a predefined token count function (num_tokens_from_string).
Update the total_token_processed count.
If the total_token_processed exceeds a predefined threshold (110,000), reset it to zero and introduce a sleep period of 60 seconds. This mechanism is designed to address rate limit issues by slowing down the processing when a certain token threshold is reached.
If the FAISS index (vec_db) is not yet initialized, create it using the current document and specified embeddings. Otherwise, add the current document to the existing index.
Saving FAISS Index:

Save the completed FAISS index to the previously created FAISS index folder.
Elapsed Time Calculation:

Record the end time and calculate the elapsed time for the entire vectorization process.
Print the completion message, including the elapsed time and the total number of processed pages.
Return Result:

Return the path of the generated FAISS index folder.
Rate Limit Handling:
The function introduces a rate limit handling mechanism by monitoring the total token count processed. When the token count exceeds a specified threshold (110,000), the function resets the count to zero and enters a sleep period of 60 seconds. This sleep period provides a pacing mechanism to avoid rate limit errors and potential service disruptions when interacting with external services or APIs.




