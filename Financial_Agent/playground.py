import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app
# Load environment variables from .env file
load_dotenv()

phi.api=os.getenv("phi-k23vDuZv3g4hOvTZSzR9njh-R42H-OAXFAPwitlMjog")

## Web Search Agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the Web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)


## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)


app=Playground(agents=[finance_agent,websearch_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)
