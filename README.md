# RAG Using FAISS Flat Index

## RAG
- RAG stands for Retrieval-Augmented Generation which bassicaly means that we are allowing LLM to refer external data sources which is typically isolated from training data which leads to responses which are accurate and contextual related.

### Why Is RAG needed?
- There are some cons of relying only on LLM which are:-
  - **The knowledge which is in need  may probably be out of date**:- which means that the information that we typically want, may not trained on LLM. But even in such cases the LLM will give response confidently related to the query. and the LLM will not admit that it does not know. 
  - **There may be no source to the response obtained**:- LLM's are not aware of  its sources from where it got the knowledge related to query.
- While RAG is used to overcome these limitations

## What is FAISS vector database and why did i consider FAISS flat index
- FAISS (Facebook AI Similarity Search) is a library which has indexing methods.
- It has various types of indexes which are essentialy used to balance the search time and search accuracy.
- These indexes are :-
  - Flat Index
  - Locality Sensitive Hashing
  - Inverted File Index

### FlatIndex
- Is mainly known for **brute search method** which means that it is trying to find the similarity or distance between query vectors and all the vectors of FAISS.
- Flat Index uses L2 distance or cosine similarity search to find the match or distance between vectors.
- Before passing vector embbendings to FAISS it must be normalized if we are going with cosine similarity approch.
 - This approch is needed to avoid the limitations caused due to magnitudes while finding the angle. Even if both vectors are pointing in same direction but have different magnitude it  will lead to  different similarity score.
- The reason I choose FAISS is because it is suitable for smaller datasets and everytime its search accuracy is 100%.  but it is bad with larger datatasets as it takes more time to compute.

### Locality Sensitive Hashing
  - due to it's components it makes the query vector be compared with a group of few vectors.
  - **Hashing Function** is used in Dictionary as well as LSH.
    - Hashing function is used in dictionary to prevent **collisions**.
    - Whereas in LSH it is authorised to perform collisions.
  - collisions means hashing a key multiple times into the same bucket.
  - after the collision process the query vector is also hashed into bucket and then using hamming distance. The distances between the query bucket and other buckets  is known.
- Due to this group of few vectors is compared with query vector.

### Inverted File Index 
- it tries to compare the query vector with the vectors from vectordatabase belonging to the cluster. This performs better than flat index regarding larger datasets.
- Initially from each centroid we expand the attachment radius and where each circle collides, an edge is created. if we visualize this it can be considered as vorinoi diagram.
- Due to this cells will be created and points within that cell will be alocated to the cluster or group of the corresponding centroid.
- The query vector will be compared with all the cluster centroids. where the similarity is high. the query vector will be compared with all the vectors of the respective cell.
- Due to this we can conclude that IFI is suitable for larger datasets as search time and search accuracy are both handled in better way.

## What was performed
- Links from Depth 2 where obtained and all the appropriate links where extracted by filtering through status code. (only status code 200 where selected).
- all the information where extracted and stored in dataframe along with corresponding links.
- Text normalization was performed which included various steps such as:-
  - Tokenization:- where words where represented in form of tokens. this was done by using nltk library by importing word_tokenize.
  - Removing Puctuations:- unwanted punctuations where removed
  - Lemmatization:- the filtered tokens where lemmatized where each word was converted to its base form.
- the normalized words had to go through stride of 40 and token length 128.
 - the main purpose of this was to increase the dataset where we can get accurate results when query is provided for RAG since each document was resticted to ony 128 words.
- Converted the documents created into embbedings, normalized the embeddings as cosine similarity search was essential.( normalization was done usisng FAISS library) 
- Fed the normaized embeddings into FAISS index.
- Provided the converted query embbeding to the faiss index to provide distance and indeces.
- Based on this indices exracted the relevent documents.
- Used command xlarge nifty model for text to text generation.
- for the given model i provided prompt along with retrieved documents and query to generate response that it sticks to context and is realiable.
- Used gradio for UI
    
## Code 
``` python
def final_function(query):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)  # Normalize query for cosine similarity

    distances, indices = index.search(np.array(query_embedding),3)

    results = []
    for i in range(3):
        doc_index = indices[0][i]
        results.append({
            'document': df_chunks.iloc[doc_index]['tokens'],
            'similarity_score': float(distances[0][i]),
            'links': df_chunks.iloc[doc_index]['link']  # Now it's cosine similarity
        })
    new=''
    for j in results:
        new=new+j['document']

    urls=''
    for k in results:
        urls=urls+k['links']+','+' '


    

    prompt  = f"""
You are an AI assistant with deep knowledge of various topics. 
Use the following context to answer the user's question accurately in a summary form  covering all answer. 
If the context does not provide enough information, say "I don't know".
If the context is not able to have information regarding query say"I DONT KNOW" 

    Context:
   {new}

   Question:
   {query}

   Answer (Be concise and informative):
    """

    # Call Cohere’s API to generate text
    response = co.generate(
        model="command-xlarge-nightly",  # Choose a Cohere model
        prompt=prompt, 
        max_tokens=1000,  # Limit response length
        temperature=0.5,  # Lower temp = more factual, higher = more creative
        stop_sequences=["\n"]
    )
    answer=[]
    answer.append(response.generations[0].text.strip())
    answer.append(f' ,links:({urls})')
    return ' '.join(answer)
```
 
