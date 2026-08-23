from google.adk.agents.llm_agent import Agent
from google.cloud import bigquery

_client = bigquery.Client()


def search_docs(query: str, max_results: int = 5) -> str:
  """Search the company documents for information relevant to a question.                                                                                                
                                           
   Args:                                                                                                                                                                  
       query: What to search for.      
       max_results: How many excerpts to return.                                                                                                                          

   Returns:                            
       The most relevant document excerpts, each labeled with its source.          
   """
  rows = _client.query(
    """
     select base.section_path, base.text, distance
     from ai.search(table source.chunks, 'text', @q)
     order by distance
     limit @k
     """,
    job_config=bigquery.QueryJobConfig(query_parameters=[
      bigquery.ScalarQueryParameter('q', 'STRING', query),
      bigquery.ScalarQueryParameter('k', 'INT64', max_results),
    ]),
  ).result()

  excerpts = []
  for i, r in enumerate(rows, 1):
    excerpts.append(f'[{i}] ({r["section_path"]})\n{r["text"]}')
  return "\n\n".join(excerpts) if excerpts else "No relevant excerpts found."


root_agent = Agent(
  model='gemini-3.5-flash',
  name='doc_chat_agent',
  description='Answers questions about the tech company documents.',
  instruction="""You answer questions strictly from the company documents.                                                                                               
   Always call search_docs first and ground your answer in its results.                
   If the results don't contain the answer, say you don't know — never guess.          
   Cite sources inline as (DocName > Section).                                         
   Ignore excerpts from References or See also sections.""",
  tools=[search_docs],
)

