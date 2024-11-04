from openai import OpenAI
import asyncio
import os

class OpenAIClientWrapper:
    def __init__(self, config):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = config['model_name']
        self.base_url = config.get('base_url', None)
        self.openai_client = OpenAI(api_key=self.api_key)
        self.llm_batch_size = config.get('llm_batch_size')
        if self.base_url:
            self.openai_client.base_url = self.base_url

    async def _handle_request(self, message, temperature=0.7, max_tokens=512):
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=message,
        )
        return response

    async def dispatch_requests(self, messages_list, temperature=0.7, max_tokens=512):
        async_responses = [self._handle_request(message, temperature, max_tokens) for message in messages_list]
        response = await asyncio.gather(*async_responses)
        return response

    def call_llm_api_parallel(self, messages_list, temperature=0.7, max_tokens=512):
        print("**Start doing call_llm_api_parallel")
        batch_size = self.llm_batch_size
        response_contents = []
        
        # Split the messages_list into batches of size batch_size
        for i in range(0, len(messages_list), batch_size):
            print (f"***The current range {i}:{i+batch_size}")
            batch = messages_list[i:i + batch_size]
            
            # Process the current batch
            responses = asyncio.run(self.dispatch_requests(batch, temperature, max_tokens))
            for response_e in responses:
                response_contents.append(response_e.choices[0].message.content.strip())
        
        print(f"**call_llm_api_parallel done, {len(response_contents)}.")
        return response_contents


    def call_llm_api(self, message, temperature=0.7, max_tokens=512):
        response = asyncio.run(self._handle_request(message, temperature, max_tokens))
        return response.choices[0].message.content.strip()
    

if __name__ == "__main__":  # for testing
    # Example usage:
    # Load configuration from JSON file
    config = {
        "model_name": "gpt-4o-mini",
        "llm_batch_size": 2  # Set a batch size for testing
    }

    # Initialize the OpenAI client wrapper with config
    client = OpenAIClientWrapper(config)

    # Test for single message request
    print("**Testing call_llm_api with a single message**")
    message = [{"role": "user", "content": "Tell me a joke."}]
    response = client.call_llm_api(message, temperature=0.7, max_tokens=256)
    print(f"Response from single request: {response}")

    # Test for batch message requests
    print("\n**Testing call_llm_api_parallel with batch messages**")
    messages_list = [
        [{"role": "user", "content": "Tell me a joke."}],
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "Give me a random fun fact."}]
    ]
    
    response_batch = client.call_llm_api_parallel(messages_list, temperature=0.7, max_tokens=256)
    print("Batch Responses:")
    for idx, res in enumerate(response_batch):
        print(f"Response {idx+1}: {res}")
