from crewai import Agent, Task, Crew, Process
from tools import fetch_stock_data, get_market_sentiment

# Initialize Tools
tools = [fetch_stock_data, get_market_sentiment]

# 1. Market Researcher
researcher = Agent(
    role='Market Researcher',
    goal='Identify high-growth stocks based on technicals and fundamentals',
    backstory='Expert in quantitative analysis and market trends. You find the "alpha".',
    tools=tools,
    verbose=True,
    allow_delegation=False
)

# 2. Risk Manager
risk_manager = Agent(
    role='Risk Manager',
    goal='Ensure the fund does not exceed risk thresholds and set stop-losses',
    backstory='Former hedge fund auditor. You are conservative and prioritize capital preservation.',
    tools=tools,
    verbose=True,
    allow_delegation=False
)

# 3. Trade Strategist
strategist = Agent(
    role='Trade Strategist',
    goal='Create a precise buy/sell order based on research and risk limits',
    backstory='Specialist in timing entries and exits to maximize ROI.',
    tools=tools,
    verbose=True,
    allow_delegation=True
)

# 4. Portfolio Manager
manager = Agent(
    role='Portfolio Manager',
    goal='Allocate capital and provide the final investment decision',
    backstory='Chief Investment Officer. You take the final call and manage total fund exposure.',
    tools=tools,
    verbose=True,
    allow_delegation=True
)

def run_alder_fund(ticker):
    # Define Tasks
    t1 = Task(
        description=f"Research {ticker} for growth potential using available financial tools.",
        agent=researcher,
        expected_output="Detailed research report including price trends and business summary."
    )
    t2 = Task(
        description=f"Analyze risk for {ticker} and determine a maximum allowable loss percentage.",
        agent=risk_manager,
        expected_output="Risk profile and a specific stop-loss limit."
    )
    t3 = Task(
        description=f"Propose a trade for {ticker} based on the research and risk limits provided by previous agents.",
        agent=strategist,
        expected_output="A clear trade signal (Buy/Sell/Hold) and recommended quantity."
    )
    t4 = Task(
        description=f"Review the research, risk, and strategy for {ticker} to make a final allocation decision.",
        agent=manager,
        expected_output="Final decision on whether to invest and the percentage of fund capital to allocate."
    )

    crew = Crew(
        agents=[researcher, risk_manager, strategist, manager],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential
    )
    return crew.kickoff()
