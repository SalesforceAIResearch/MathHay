import os
import time
from tavily import TavilyClient

class TavilyClientWrapper:
    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        self.tavily_client = TavilyClient(api_key=self.api_key)

    def search(self, query):
        try:
            response = self.tavily_client.search(query=query, include_raw_content=True)
            filtered_results = [elem["raw_content"] for elem in response["results"] if elem["raw_content"] != None and len(elem["raw_content"]) <= 50000]
            return filtered_results


        except Exception as e:
            print(f"Search failed for query: {query}, Error: {e}")
            return []

if __name__ == "__main__":
    queries = [
                    'What was the percentage change in the stock price of Tesla from January 2024 to March 2024?',
                    "How does the volatility of Amazon's stock compare to that of Apple over the last six months?"
                ]

    tavily_client_wrapper = TavilyClientWrapper(api_key="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")
    page_content_dict = {}
    topic_count = 0

    for query in queries:
        search_result = tavily_client_wrapper.search(query)
        page_content_dict[query] = search_result
        if search_result:  # Check if search results were returned
            topic_count += 1
        time.sleep(1)  # Pause between queries to avoid rate limiting
        print('+++++++++++')
        print(query)
        print(len(search_result))
        print()
