import pandas as pd
import numpy as np
import ollama

class pandas_agent_class:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def pandas_agent(self, query: str) -> str:
        self.query = query
        self.df = pd.read_csv(self.csv_path)
        sample = self.df.head(5)
        self.schema = str(self.df.dtypes)
        
        self.prompt = f"""You are a python programmer and you are being tasked to perform analytical tasks over a dataset in python using pandas framework.

        The dataframe is already loaded as 'df'. Do not read the CSV again.

        The schema of the dataset is as below:
        {self.schema}

        The first few rows of the dataset are as below:
        {sample}

        You are tasked to write python code in pandas to perform the task at hand defined as below:
        {self.query}

        Store your final answer in a variable called 'result'.
        Only return the pandas code to perform the task and nothing else, no explanation, no markdown.
        """

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query}
            ]
        )
        answer = response["message"]["content"]
        self.code = answer.replace('```python', '').replace('```', '').strip()
        
        local_vars = {"df": self.df, "pd": pd}
        exec(self.code, {}, local_vars)
        
        self.result = local_vars.get("result", "No result variable found")
        return self.result


if __name__ == "__main__":
    query = "What is the product category wise total of transaction amount shown in the dataset?"
    csv_path = "./data/mock_data.csv"
    pandas_obj = pandas_agent_class(csv_path)
    answer = pandas_obj.pandas_agent(query)
    print(answer)