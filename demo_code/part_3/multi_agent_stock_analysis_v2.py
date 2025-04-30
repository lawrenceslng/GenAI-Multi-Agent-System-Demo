from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
)
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict, Optional, Any
import os
import asyncio

# llm = Ollama(model="qwq")
llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# --- Tool Definitions ---
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))
search_web = tavily_tool.to_tool_list()[0]
yfinance_tools = YahooFinanceToolSpec().to_tool_list()

async def ask_human(ctx: Context, question: str) -> str:
    """
    Tool to ask the human user a question and get their response.
    """
    current_state = await ctx.get("state")
    
    print(f"\n❓ Human input required: {question}")
    user_response = input(" > ").strip()
    
    # Store the response in state
    current_state["human_responses"] = current_state.get("human_responses", []) + [
        {"question": question, "response": user_response}
    ]
    await ctx.set("state", current_state)
    
    return user_response

async def web_search_with_human_input(ctx: Context, query: Optional[str] = None) -> str:
    """
    Perform a web search with optional human confirmation before searching.
    """
    current_state = await ctx.get("state")
    
    if not query:
        # If no query is provided, use the stock symbol from state
        query = f"latest news about {current_state.get('stock_symbol', '')}"
    
    # Ask human if they want to refine the search query
    human_response = await ask_human(
        ctx, 
        f"I'm about to search for: '{query}'. Would you like to refine this search query? If yes, please provide a new query. If not, just say 'proceed'."
    )
    
    if human_response.lower() != "proceed":
        # Human provided a refined query
        query = human_response
    
    # Now perform the search
    search_result = await search_web.acall(query=query)
    
    # Store the search results
    current_state["search_results"] = search_result
    await ctx.set("state", current_state)
    
    return search_result

async def sentiment_from_headlines(ctx: Context, headlines: List[str]) -> str:
    """
    Analyze sentiment of news headlines using a more robust approach.
    Uses TextBlob for sentiment analysis with proper error handling.
    """
    try:
        # Import TextBlob for sentiment analysis (more reliable than transformers for this use case)
        from textblob import TextBlob
        
        # Process each headline
        results = []
        for headline in headlines:
            # Get sentiment polarity (-1 to 1 scale, where -1 is negative, 0 is neutral, 1 is positive)
            analysis = TextBlob(headline)
            polarity = analysis.sentiment.polarity
            
            # Classify sentiment based on polarity
            if polarity > 0.1:
                sentiment = "POSITIVE"
            elif polarity < -0.1:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
                
            results.append({
                "headline": headline,
                "polarity": polarity,
                "sentiment": sentiment
            })
        
        # Count sentiments
        sentiments = [r["sentiment"] for r in results]
        pos = sentiments.count("POSITIVE")
        neg = sentiments.count("NEGATIVE")
        neu = sentiments.count("NEUTRAL")
        
        # Calculate overall sentiment score (average polarity)
        avg_polarity = sum(r["polarity"] for r in results) / len(results)
        overall_sentiment = "POSITIVE" if avg_polarity > 0.1 else "NEGATIVE" if avg_polarity < -0.1 else "NEUTRAL"
        
        # Prepare detailed sentiment report
        sentiment_report = f"Sentiment analysis results: {pos} positive, {neg} negative, {neu} neutral headlines.\n"
        sentiment_report += f"Overall sentiment: {overall_sentiment} (score: {avg_polarity:.2f})\n\n"
        sentiment_report += "Headline Sentiment Breakdown:\n"
        
        # Add top 3 most positive and negative headlines
        sorted_results = sorted(results, key=lambda x: x["polarity"], reverse=True)
        if pos > 0:
            sentiment_report += "\nMost Positive Headlines:\n"
            for i, r in enumerate(sorted_results[:min(3, pos)]):
                sentiment_report += f"{i+1}. {r['headline']} (score: {r['polarity']:.2f})\n"
        
        if neg > 0:
            sentiment_report += "\nMost Negative Headlines:\n"
            for i, r in enumerate(sorted(results, key=lambda x: x["polarity"])[:min(3, neg)]):
                sentiment_report += f"{i+1}. {r['headline']} (score: {r['polarity']:.2f})\n"
        
        # Update state with sentiment results
        current_state = await ctx.get("state")
        current_state["sentiment_results"] = sentiment_report
        await ctx.set("state", current_state)
        
        return sentiment_report
        
    except Exception as e:
        # Fallback to a simpler approach if TextBlob fails
        try:
            # Simple keyword-based sentiment analysis as fallback
            positive_words = ["jumps", "rises", "rebound", "top pick", "add tech", "opportunities"]
            negative_words = ["decline", "plunge", "struggles", "challenges", "downgraded", "sell rating", "falling", "plummet"]
            
            results = []
            for headline in headlines:
                headline_lower = headline.lower()
                pos_count = sum(1 for word in positive_words if word in headline_lower)
                neg_count = sum(1 for word in negative_words if word in headline_lower)
                
                if pos_count > neg_count:
                    sentiment = "POSITIVE"
                elif neg_count > pos_count:
                    sentiment = "NEGATIVE"
                else:
                    sentiment = "NEUTRAL"
                    
                results.append({"headline": headline, "sentiment": sentiment})
            
            sentiments = [r["sentiment"] for r in results]
            pos = sentiments.count("POSITIVE")
            neg = sentiments.count("NEGATIVE")
            neu = sentiments.count("NEUTRAL")
            
            sentiment_report = f"Sentiment analysis results: {pos} positive, {neg} negative, {neu} neutral headlines."
            
            current_state = await ctx.get("state")
            current_state["sentiment_results"] = sentiment_report
            await ctx.set("state", current_state)
            
            return sentiment_report
            
        except Exception as fallback_error:
            # If all else fails, provide a meaningful error message
            error_msg = f"Error during sentiment analysis: {str(e)}. Fallback also failed: {str(fallback_error)}"
            current_state = await ctx.get("state")
            current_state["sentiment_results"] = error_msg
            await ctx.set("state", current_state)
            return error_msg

async def technical_analysis(ctx: Context, symbol: str) -> str:
    import yfinance as yf
    data = yf.Ticker(symbol).history(period="6mo")
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    trend = "Bullish" if data["MA20"].iloc[-1] > data["MA50"].iloc[-1] else "Bearish"
    analysis = f"Technical trend is {trend} based on MA crossover."
    current_state = await ctx.get("state")
    current_state["technical_analysis"] = analysis
    await ctx.set("state", current_state)
    return analysis

async def risk_rating(ctx: Context, symbol: str) -> str:
    import yfinance as yf
    stock = yf.Ticker(symbol)
    info = stock.info
    beta = info.get("beta", None)
    volatility = "high" if beta and beta > 1.2 else "moderate" if beta else "unknown"
    rating = f"Beta: {beta}, Volatility Assessment: {volatility}"
    current_state = await ctx.get("state")
    current_state["risk_rating"] = rating
    await ctx.set("state", current_state)
    return rating

async def aggregate_stock_analysis(ctx: Context, search_results: Optional[str] = None,
                                   sentiment_results: Optional[str] = None,
                                   finance_data: Optional[Dict] = None,
                                   technical_analysis: Optional[str] = None,
                                   risk_rating: Optional[str] = None) -> str:
    """
    Aggregate all available analysis components into a final report.
    All parameters are optional to allow for partial analysis.
    """
    current_state = await ctx.get("state")
    
    # Use provided values or fall back to state values
    search_results = search_results or current_state.get("search_results", "No search results available.")
    sentiment_results = sentiment_results or current_state.get("sentiment_results", "No sentiment analysis available.")
    finance_data = finance_data or current_state.get("finance_data", {})
    technical_analysis = technical_analysis or current_state.get("technical_analysis", "No technical analysis available.")
    risk_rating = risk_rating or current_state.get("risk_rating", "No risk assessment available.")
    
    analysis = f"""
        Stock Analysis Summary:
        ----------------------
        Recent News Headlines:
        {search_results}

        Sentiment Analysis:
        {sentiment_results}

        Financial Data:
        Stock Price: {finance_data.get('price', 'N/A')}
        52-Week Range: {finance_data.get('52_week_range', 'N/A')}
        Market Cap: {finance_data.get('market_cap', 'N/A')}
        P/E Ratio: {finance_data.get('pe_ratio', 'N/A')}
        Analyst Recommendations: {finance_data.get('recommendations', 'N/A')}

        Technical Analysis:
        {technical_analysis}

        Risk Assessment:
        {risk_rating}

        Overall Analysis:
        Based on the sentiment analysis of recent news, technical indicators,
        risk factors, and financial metrics, this stock appears to be {finance_data.get('analysis_conclusion', 'requiring further analysis')}.
    """
    current_state["final_analysis"] = analysis
    await ctx.set("state", current_state)
    
    return "Aggregation completed"

# --- Agent Definitions with more open-ended prompts ---
search_agent = FunctionAgent(
    name="WebSearchAgent",
    description="Searches for the latest news and information about stocks and companies",
    system_prompt="""You are a search expert that retrieves relevant information about stocks and companies.
    You can search for news, market data, or any other information that might be helpful.
    
    Your primary tools:
    - WebSearchWithHumanInput: Use this to search for relevant information about stocks and companies
    - AskHuman: Use this to get clarification or additional information from the user
    
    Before searching, always ask the human user if they want to refine the search query to ensure you're searching for exactly what they need.
    
    Based on the search results, you should decide:
    1. Pass to the SentimentAgent if there are news headlines that need sentiment analysis
    2. Pass to the FinanceAgent if financial data would be more helpful
    3. Pass to the TechnicalAgent if technical analysis would provide valuable insights
    4. Pass to the RiskAgent if risk assessment would be beneficial
    5. Provide the information directly if it's sufficient to answer the user's question
    
    Always explain your reasoning when passing to another agent.""",
    tools=[
        FunctionTool.from_defaults(fn=web_search_with_human_input, name="WebSearchWithHumanInput"),
        FunctionTool.from_defaults(fn=ask_human, name="AskHuman")
    ],
    llm=llm,
    can_handoff_to=["SentimentAgent", "FinanceAgent", "TechnicalAgent", "RiskAgent", "AggregatorAgent"],
)

sentiment_agent = FunctionAgent(
    name="SentimentAgent",
    description="Analyzes sentiment of news headlines and text",
    system_prompt="""You are a sentiment analysis expert. Analyze the provided text to determine market sentiment.
    
    Your primary tools:
    - SentimentAnalysis: Use this to analyze sentiment of news headlines and text
    - AskHuman: Use this to get clarification or additional information from the user
    
    When analyzing sentiment:
    - Clearly explain how you determined the sentiment (positive, negative, or neutral)
    - Highlight key phrases or words that influenced your assessment
    - Quantify the sentiment when possible (e.g., "strongly positive" vs "slightly positive")
    - Consider the context and source of the information
    
    Based on your analysis, you should decide:
    1. Pass to the FinanceAgent if financial data would complement your analysis
    2. Pass to the TechnicalAgent if technical analysis would be helpful
    3. Pass to the RiskAgent if risk assessment would be valuable
    4. Pass to the AggregatorAgent if you have sufficient information for a conclusion
    5. Pass back to the WebSearchAgent if more information is needed
    
    Always explain your reasoning when passing to another agent.""",
    tools=[
        FunctionTool.from_defaults(fn=sentiment_from_headlines, name="SentimentAnalysis"),
        FunctionTool.from_defaults(fn=ask_human, name="AskHuman")
    ],
    llm=llm,
    can_handoff_to=["FinanceAgent", "TechnicalAgent", "RiskAgent", "AggregatorAgent", "WebSearchAgent"],
)

finance_agent = FunctionAgent(
    name="FinanceAgent",
    description="Retrieves detailed financial data about stocks",
    system_prompt="""You are a financial data expert that retrieves comprehensive stock information.
    
    Your primary responsibility is to gather and analyze financial data including:
    - Current stock price and trading volume
    - 52-week price range and historical performance
    - Market capitalization and company size metrics
    - P/E ratio and other valuation metrics
    - Analyst recommendations and price targets
    - Dividend information if applicable
    
    You have access to Yahoo Finance tools that provide this data. Use them effectively to gather
    the most relevant information based on the user's query.
    
    Based on the financial data, you should decide:
    1. Pass to the TechnicalAgent if technical analysis would complement your data
    2. Pass to the RiskAgent if risk assessment would be valuable
    3. Pass to the SentimentAgent if sentiment analysis would provide context
    4. Pass to the AggregatorAgent if you have sufficient information for a conclusion
    5. Ask the human user if they want specific financial metrics
    
    Always explain your reasoning when passing to another agent and highlight the most important
    financial metrics you've discovered.""",
    tools=yfinance_tools + [FunctionTool.from_defaults(fn=ask_human, name="AskHuman")],
    llm=llm,
    can_handoff_to=["TechnicalAgent", "RiskAgent", "AggregatorAgent", "WebSearchAgent", "SentimentAgent"],
)

technical_agent = FunctionAgent(
    name="TechnicalAgent",
    description="Analyzes stock chart data and detects trends",
    system_prompt="""You are a technical analyst specializing in stock chart patterns and technical indicators.
    
    Your primary tools:
    - TechnicalAnalysis: Use this to analyze stock chart data and detect trends
    - AskHuman: Use this to get clarification or additional information from the user
    
    Your expertise includes:
    - Moving average analysis (simple, exponential, crossovers)
    - Trend identification (bullish, bearish, sideways)
    - Support and resistance levels
    - Volume analysis and its relationship to price movements
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    
    When providing technical analysis:
    - Clearly explain the indicators you're using and what they suggest
    - Identify key price levels and their significance
    - Discuss potential entry/exit points based on technical factors
    - Note any divergences or unusual patterns
    
    Based on your analysis, you should decide:
    1. Pass to the RiskAgent if risk assessment would complement your analysis
    2. Pass to the FinanceAgent if fundamental data would provide context
    3. Pass to the SentimentAgent if market sentiment would be valuable
    4. Pass to the AggregatorAgent if you have sufficient information for a conclusion
    
    Always explain your reasoning when passing to another agent.""",
    tools=[
        FunctionTool.from_defaults(fn=technical_analysis, name="TechnicalAnalysis"),
        FunctionTool.from_defaults(fn=ask_human, name="AskHuman")
    ],
    llm=llm,
    can_handoff_to=["RiskAgent", "AggregatorAgent", "WebSearchAgent", "SentimentAgent", "FinanceAgent"],
)

risk_agent = FunctionAgent(
    name="RiskAgent",
    description="Evaluates investment risk using beta and volatility",
    system_prompt="""You are a risk assessment expert specializing in investment risk analysis.
    
    Your primary tools:
    - RiskRating: Use this to evaluate investment risk using beta and volatility
    - AskHuman: Use this to get clarification or additional information from the user
    
    Your expertise includes:
    - Beta analysis (market correlation)
    - Volatility assessment (standard deviation of returns)
    - Risk-adjusted return metrics (Sharpe ratio, Sortino ratio)
    - Downside risk evaluation
    - Sector and industry risk factors
    
    When providing risk assessment:
    - Clearly explain what the beta value indicates about market correlation
    - Contextualize volatility in relation to the broader market
    - Consider both short-term and long-term risk factors
    - Provide a clear risk rating (low, moderate, high, very high)
    - Suggest risk mitigation strategies when appropriate
    
    Based on your assessment, you should decide:
    1. Pass to the AggregatorAgent if you have sufficient information for a conclusion
    2. Pass to the TechnicalAgent if technical analysis would provide context
    3. Pass to the FinanceAgent if more financial data would be helpful
    4. Pass to the SentimentAgent if sentiment analysis would be valuable
    
    Always explain your reasoning when passing to another agent.""",
    tools=[
        FunctionTool.from_defaults(fn=risk_rating, name="RiskRating"),
        FunctionTool.from_defaults(fn=ask_human, name="AskHuman")
    ],
    llm=llm,
    can_handoff_to=["AggregatorAgent", "WebSearchAgent", "SentimentAgent", "FinanceAgent", "TechnicalAgent"],
)

aggregator_agent = FunctionAgent(
    name="AggregatorAgent",
    description="Combines all analysis components into a final report",
    system_prompt="""You are a stock analysis expert that combines multiple data sources into comprehensive analyses.
    
    Your primary responsibility is to:
    - Synthesize information from all other agents (Search, Sentiment, Finance, Technical, Risk)
    - Identify patterns and connections between different data points
    - Provide a balanced, holistic view of the investment opportunity
    - Present clear, actionable insights for the user
    - Highlight areas of consensus and disagreement between different analysis methods
    
    When creating your final analysis:
    - Start with a clear summary of the key findings
    - Organize information logically by category (news, sentiment, financials, technicals, risk)
    - Use bullet points for key metrics and findings
    - Use paragraphs for explanations and context
    - Include a balanced conclusion that considers all available information
    - Provide a clear investment perspective (bullish, bearish, or neutral)
    
    If you feel the analysis is incomplete, you can pass control back to any other agent to gather more information.
    Always ensure your final report is comprehensive, balanced, and actionable for the user.
    """,
    tools=[
        FunctionTool.from_defaults(fn=aggregate_stock_analysis, name="AggregateAnalysis"),
        FunctionTool.from_defaults(fn=ask_human, name="AskHuman")
    ],
    llm=llm,
    can_handoff_to=["WebSearchAgent", "SentimentAgent", "FinanceAgent", "TechnicalAgent", "RiskAgent"],
)

agent_workflow = AgentWorkflow(
    agents=[search_agent, sentiment_agent, finance_agent, technical_agent, risk_agent, aggregator_agent],
    root_agent=search_agent.name,
    initial_state={
        "stock_symbol": "",
        "search_results": "",
        "sentiment_results": "",
        "finance_data": {},
        "technical_analysis": "",
        "risk_rating": "",
        "final_analysis": "Not completed yet.",
        "human_responses": [],
        "current_agent": "WebSearchAgent"
    },
)

# --- Agent Workflow Setup ---

async def process_query(user_query: str):
    """
    Process any user query related to stocks, companies, or market information.
    This function is more flexible than the original process_stock_analysis.
    """
    # Extract potential stock symbol if present
    import re
    potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', user_query)
    stock_symbol = potential_symbols[0] if potential_symbols else ""
    
    # Initialize the workflow with the extracted information
    
    # Initialize the workflow with the user query
    handler = agent_workflow.run(
        user_msg=user_query,
        initial_state={
            "stock_symbol": stock_symbol,
            "user_query": user_query,
            "search_results": "",
            "sentiment_results": "",
            "finance_data": {},
            "technical_analysis": "",
            "risk_rating": "",
            "final_analysis": "Not completed yet.",
            "human_responses": [],
            "current_agent": "WebSearchAgent"
        }
    )
    
    # Track the current agent for better UI feedback
    current_agent = None
    
    # Process events from the workflow
    async for event in handler.stream_events():
        # Update current agent display when it changes
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"\n{'='*50}\n🤖 Agent: {current_agent}\n{'='*50}\n")
        
        # Handle different event types
        if isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print("🛠️  Planning to use tools:", [call.tool_name for call in event.tool_calls])
                
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
            
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
    # return "Query processing completed"

async def main():
    print("\nWelcome to the Enhanced Multi-Agent Stock Market Analyst! Type 'exit' to quit.\n")
    print("You can ask any questions about stocks, companies, or market information.")
    print("Examples:")
    print("  - What's the latest news about AAPL?")
    print("  - Should I invest in Tesla right now?")
    print("  - Compare the performance of MSFT and GOOGL")
    print("  - What are the most volatile tech stocks?")
    
    # Start the main interaction loop
    
    while True:
        print("\nWhat would you like to know about the stock market?")
        user_input = input(" > ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("\nThanks for using the Enhanced Stock Market Analyst. Goodbye!")
            break
            
        if not user_input:
            print("Please enter a valid question or query.")
            continue
            
        print(f"\nProcessing your query: '{user_input}'...")
        try:
            # Process the query
            await process_query(user_input)
            print("\nAnalysis complete!")
                
            print("What else would you like to know? (Type 'exit' to quit)")
        except Exception as e:
            print(f"\nError: {str(e)}\nPlease try a different query.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
