# Enterprise GenAI Knowledge Assistant on GCP using RAG

# Summary

This document explains one approach on how to build an enterprise GenAI application on GCP. It uses a source corpus of PDFs which are articles about Tech companies and creates a Retrieval Augmented Generation (RAG) system and a Knowledge Assistant app that uses LLM to fetch the relevant information and answer questions.

The implementation is handled in two stages:

1. Extract Transform Load (ETL) - Process the pdfs and generate the embeddings for semantic retrieval
2. Knowledge Assistant - An application that utilises an LLM with a RAG to provide answers in Q&A style with a chat like UI interface.

Detailed implementation is explained below.

Tech used in the development of this app:

1. Python based code
2. Google Cloud Storage for input
3. Google BigQuery for storing the chunks and generating embedding vectors
4. Google Cloud Run for running the ETL and the knowledge assistant.
5. Google ADK for building the knowledge assistant with built in tools for RAG.

> **Note:** This is a detailed architecture/design write-up. Detailed build and setup instructions are TODO.

# Source data

For demonstration purposes, we use articles of various tech companies (AMD, Nvidia etc) downloaded as pdfs directly from wikipedia.

We create a new bucket in cloud storage and create the following folders within:

 * `input/raw` - Stores the raw pdfs
 * `input/processed` - Processed pdfs are moved to this location with a timestamp suffix.

# Extract Transform Load (ETL)

This step handles the entire end to end processing. In detail, it performs the following steps:

## Initialize

1. Check cloud storage (`input/raw`) for any new pdf files
2. Generate the document hash and check if the document has already been ingested. If yes, then move the file to the processed folder (`input/processed`)
3. If a new file, then proceed to extraction.
4. When completed,  move the file to the processed folder.

## Extraction

1. Retrieve the metadata of the document. This includes the following:
   1. doc_name (ex: Qualcomm.pdf)
   2. doc_type (pdf only for now)
   3. doc_hash - sha256 hash of the raw binary contents of the file. This is used to identify if the document has already been processed or not
   4. ingestion_timestamp - Timestamp of when the file is processed
   5. page_count - number of pages in the pdf
2. Extract the text broken down by pages as well as sections. While the pages are not used currently, this will help future enhancements.
3. Process each section recursively to identify the section heading and content. Sub sections are also processed and nested under the parent section to identify the exact path of the content.
4. **Clean up**: Titles and section data is stripped of markdown characters (#, *), stripped of html tags and comments and stripped of excessive spaces and newlines
5. The final structure of the data generated after this step is as follows:
   ```json
   {
    "doc_name": "Qualcomm.pdf",
    "doc_type": "pdf",
    "doc_hash": "59f61b0d61fa87072732829e3625ab7811e098098ad207802dc7872906f9bd67",
    "ingestion_timestamp": 1787439191,
    "page_count": 41,
    "pages": ["..."],
    "sections": {...}
   }
   ```

## Transformation

In this stage, we break down the data into chunks for our RAG. To keep things simple for this demonstration, we just use sections as our chunks and ignore the pages. Future enhancement would be to identify the exact page that a particular section is present in to share more information to the user and also to aid in validation.

Each section and sub-section is a separate chunk. To distinguish the sections and also to preserve section hierarchy, we use a `section_path` which identifies the parent-child section relationship.

Sample output at this stage looks like this:

```json
[
    {
        "text": "Qualcomm\nQualcomm Incorporated (/\u02c8kw\u0252lk\u0252m/ ~~)~~ [2] is an American multinational ... ",
        "metadata": {
            "doc_name": "Qualcomm.pdf",
            "doc_hash": "59f61b0d61...",
            "section_path": "Qualcomm",
            "word_count": 235
        }
    },
    {
        "text": "Qualcomm/Qualcomm Incorporated\nHeadquarters in San Diego, California\nType Public Traded ...",
        "metadata": {
            "doc_name": "Qualcomm.pdf",
            "doc_hash": "59f61b0d61...",
            "section_path": "Qualcomm/Qualcomm Incorporated",
            "word_count": 152
        }
    },
    ...
]
```

This goes to the next step for Loading into BigQuery

## Load

We use BigQuery for storing the chunks and the embedding vectors for the RAG.

BigQuery now supports [Autonomous Embedding Generation](https://docs.cloud.google.com/bigquery/docs/autonomous-embedding-generation). This completely removes the need for a separate step to generate the embedding vectors, thereby reducing the complexity.

BigQuery also provides a search function `AI.SEARCH` which helps in the retrieval (more on this later).

We need two tables in BigQuery. We will  greate both of these in a new dataset called `source`:

 * `source.documents` - One row per document ingested. Used to check if the document is already present in the DB.
 * `source.chunks` - Contains all the chunks of the documents along with the section paths and embedding vectors which are autogenerated.

The SQL to generate these is below.

```sql
CREATE TABLE source.documents (
  doc_hash          STRING,
  doc_name          STRING,
  doc_type          STRING,
  ingestion_timestamp INT64,
  page_count        INT64,
  PRIMARY KEY (doc_hash) NOT ENFORCED
);

CREATE TABLE source.chunks (
  chunk_id     STRING,
  doc_hash     STRING,
  section_path STRING,
  text         STRING,
  word_count   INT64,
  embedding    STRUCT<result ARRAY<FLOAT64>, status STRING>
    GENERATED ALWAYS AS (
      AI.EMBED(text,
        model => 'embeddinggemma-300m')
    ) STORED OPTIONS (asynchronous = TRUE),
  PRIMARY KEY (chunk_id) NOT ENFORCED,
  FOREIGN KEY (doc_hash) REFERENCES source.documents(doc_hash) NOT ENFORCED
);
```

Once the tables are created, the ETL script can insert the rows in the tables and the embeddings will get generated automatically.

> **Note**: There may be a delay of 30-90 minutes in generating the embeddings when using the bigquery API to insert rows. This because of streaming which prevents making any changes (update, delete) until the stream closes (which is decided by bigquery).

## Testing the semantic search

After completing the above, we should have data available to search for using semantic similarity. To test and see that it works as expected, we can try it out directly in BigQuery using the `AI.SEARCH`.

In BigQuery query editor, we can run the following query to see what results will be returned:

```sql
SELECT
  base.section_path,
  base.text,
  distance
FROM AI.search(TABLE source.chunks, 'text', 'When was Qualcomm founded?')
ORDER BY distance
```

If it works you should see it ordered by the relevance of your input string.

# GenAI Knowledge Assistant

Now that we have the embeddings, we should use an LLM and hand the information retrieval step to it in the form of a tool.

This is where we use `google-adk` to build an application and then deploy it on cloud run so that we can build an application on top of it.

For this use case we will use `gemini-3.5-flash` as the model.

## Setting up on Cloud Run

ADK based agent is deployed on cloud run with the following prompt to make sure it only uses the available documentation to answer questions and minimize hallucination.

```python
root_agent = Agent(
  model='gemini-3.5-flash',
  name='doc_chat_agent',
  description='Answers questions about the tech company documents.',
  instruction="""You answer questions strictly from the company documents. Always call search_docs first and ground your answer in its results. If the results don't contain the answer, say you don't know — never guess. Cite sources inline as (DocName > Section).
  Ignore excerpts from References or See also sections.""",
  tools=[search_docs],
)
```

We test this locally first using `adk web` and see if it does the tool calling and answering the documents.

We can then deploy it using the following command:

```bash
gcloud run deploy doc-search \
  --source . \
  --region='us-central1' \
  --allow-unauthenticated
```

After deploying, we can test that it works using the following curl commands.  First we create a new session for a user:

```bash
SERVICE_URL='https://doc-search-444119881957.us-central1.run.app'
curl -X POST ${SERVICE_URL}/apps/doc_search/users/u_123/sessions/s_123 -H "Content-Type: application/json" -d '{"key1": "value1", "key2": "value2"}'
```

```bash
curl -X POST ${SERVICE_URL}/run -H "Content-Type: application/json" -d "{\"appName\": \"doc_search\",\"userId\": \"u_123\",\"sessionId\": \"s_123\",\"newMessage\": { \"role\": \"user\", \"parts\": [{ \"text\": \"What year was Nvidia founded? Who founded it?\" }]}}"
```
## Evaluation

Here we attempt to evaluate the assistant's performance using `agents-cli` that is available with `google-adk`. We install it using `uvx google-agents-cli setup`.

Since we haven't used this tool to create the project, we'll need to enable it for our project using the following command. In the `app` directory, run:

```bash
agents-cli scaffold enhance . --agent-directory doc_search -i
```

Answer the questions it asks.

Once done, we set up a few sample prompts and responses in `tests/eval/datasets/basic-dataset.json`:

```json
{
  "eval_cases": [
    {
      "eval_case_id": "nvidia_founding",
      "prompt": { "role": "user", "parts": [{"text": "What year was Nvidia founded? Who founded it?"}] },
      "reference": {
        "response": { "role": "model", "parts": [{"text": "Nvidia was founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem."}] }
      }
    },
    {
      "eval_case_id": "qualcomm_founding",
      "prompt": { "role": "user", "parts": [{"text": "When was Qualcomm founded?"}] },
      "reference": {
        "response": { "role": "model", "parts": [{"text": "Qualcomm was founded in 1985."}] }
      }
    },
    {
      "eval_case_id": "amd_hq",
      "prompt": { "role": "user", "parts": [{"text": "Where is AMD headquartered?"}] },
      "reference": {
        "response": { "role": "model", "parts": [{"text": "AMD is headquartered in Santa Clara, California."}] }
      }
    },
    {
      "eval_case_id": "not_in_docs",
      "prompt": { "role": "user", "parts": [{"text": "What is Apple's current quarterly revenue?"}] },
      "reference": {
        "response": { "role": "model", "parts": [{"text": "The company documents do not contain information about Apple's quarterly revenue."}] }
      }
    }
  ]
}
```

And mention all the metrics we want to evaluate on in `tests/eval/eval_config.yaml`:

```yaml
metrics_to_run:
  - hallucination
  - final_response_quality
  - final_response_match
  - citation_format

custom_metrics:
  - name: citation_format
    prompt_template: |
      You are checking the citation format of an agent's answer.
      The agent must cite sources inline as (DocName > Section), e.g. (Nvidia > History).
      Prompt: {prompt}
      Final response: {response}
      If the answer correctly says it does not know (no citation needed), score 5.
      Otherwise score 1-5: 5 = every factual claim carries a correctly formatted inline
      citation, 1 = no citations or wrong format.
      Return JSON: {"score": <1-5>, "explanation": "<reason>"}
```

Then from the directory where the scaffolding was setup (the `app` directory), we run the following command to run the evaluation:

```bash
agents-cli eval run \
  --dataset tests/eval/datasets/basic-dataset.json \
  --config tests/eval/datasets/eval_config.yaml \
  --url https://doc-search-444119881957.us-central1.run.app \
  --app-name doc_search
```

## Clean Up

We can delete the cloud run instance with the following command:

```bash
gcloud run services delete doc-search --region=us-central1
```

## Next Steps

* We are using Google Cloud Run in the current implementation. For a real production app, it might be better to use Google Agent Runtime because it provides a lot more benefits out of the box (logging, monitoring, evaluation, authentication etc). But Cloud Run is a good way to have fine grain control on everything.
* While this is a good start to a RAG implementation, a fully working UI will be good to have. Personally I like and am comfortable with openwebui. But Google ADK is not directly compatible. We'll need to build an adapter on top of it.

# Future Enhancements

* ETL
  * Identify the start and end page of each section for reference and validation.
  * Further clean up of the data to clean up unnecessary characters.
  * Move the snippets from each pdf into a dedicated section.
  * Ignore smaller chunks of one word or less (Ex: "See Also").
  * Handle versioning of the same file.
  * Currently the ETL is running locally. This can be modified to run in cloud run on scheduler.
      * Alternatively, use cloud pub/sub to auto-ingest files once uploaded to cloud storage.
  * Only processes pdfs for now. Support other data types also.
  * Remove references from the source document or handle them better.
* GenAI Knowledge Application
  * Enable Authentication so that we aren't dealing with endpoints open to the public.
  * Save session state to persistent storage (cloud run is transient).
  * Move to Agent Runtime instead of Cloud Run.

