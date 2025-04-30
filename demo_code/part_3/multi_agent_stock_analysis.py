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
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict
import os

# llm = Ollama(model="llama3.2")
llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# --- Tool Definitions ---
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))
search_web = tavily_tool.to_tool_list()[0]
yfinance_tools = YahooFinanceToolSpec().to_tool_list()

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

async def aggregate_stock_analysis(ctx: Context, search_results: str, sentiment_results: str, finance_data: Dict, technical_analysis: str, risk_rating: str) -> str:
    current_state = await ctx.get("state")
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

# --- Agent Definitions ---
search_agent = FunctionAgent(
    name="WebSearchAgent",
    description="Searches for the latest news headlines about a specific stock",
    system_prompt="""You are a search expert that retrieves the 5-10 most recent and relevant news headlines about a given stock.
    Format your response as a clear list of headlines. You should hand off control to the SentimentAgent to gauge the sentiment of the headlines.""",
    tools=[search_web],
    llm=llm,
    can_handoff_to=["SentimentAgent"],
)

sentiment_agent = FunctionAgent(
    name="SentimentAgent",
    description="Analyzes sentiment of news headlines",
    system_prompt="""You are a sentiment analysis expert. Analyze the provided headlines to determine market sentiment.
    Provide clear reasoning for your sentiment assessment. You should hand off control to the FinanceAgent to further get financial data about the stock.""",
    tools=[FunctionTool.from_defaults(fn=sentiment_from_headlines, name="SentimentAnalysis")],
    llm=llm,
    can_handoff_to=["FinanceAgent"],
)

finance_agent = FunctionAgent(
    name="FinanceAgent",
    description="Retrieves detailed financial data about stocks",
    system_prompt="""You are a financial data expert that retrieves comprehensive stock information.
    Always include: current price, 52-week range, market cap, P/E ratio, and analyst recommendations. 
    You should hand off control to the TechnicalAgent and RiskAgent before final aggregation.""",
    tools=yfinance_tools,
    llm=llm,
    can_handoff_to=["TechnicalAgent", "RiskAgent"],
)

technical_agent = FunctionAgent(
    name="TechnicalAgent",
    description="Analyzes stock chart data and detects trends",
    system_prompt="""You are a technical analyst. Use historical price data and indicators to evaluate stock momentum or reversal trends. You should hand off control to the RiskAgent after you are done with technical analysis.""",
    tools=[FunctionTool.from_defaults(fn=technical_analysis, name="TechnicalAnalysis")],
    llm=llm,
    can_handoff_to=["RiskAgent"],
)

risk_agent = FunctionAgent(
    name="RiskAgent",
    description="Evaluates investment risk using beta and volatility",
    system_prompt="""You are a risk assessment expert. Use the stock's beta and volatility metrics to determine risk exposure. You should hand off control to the AggregatorAgent after you are done with analyzing the risk.""",
    tools=[FunctionTool.from_defaults(fn=risk_rating, name="RiskRating")],
    llm=llm,
    can_handoff_to=["AggregatorAgent"],
)

aggregator_agent = FunctionAgent(
    name="AggregatorAgent",
    description="Combines all analysis components into a final report",
    system_prompt="You are a stock analysis expert that combines multiple data sources into comprehensive stock analyses.",
    tools=[FunctionTool.from_defaults(fn=aggregate_stock_analysis, name="AggregateAnalysis")],
    llm=llm,
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
        "final_analysis": "Not completed yet."
    },
)

async def process_stock_analysis(symbol_or_name: str):
    handler = agent_workflow.run(user_msg=f"""
        Analyze {symbol_or_name} by looking for the latest headlines, 
        doing sentiment analysis on them, then getting additional company 
        stock information, performing technical analysis, assessing risk, 
        and providing a comprehensive summary.
    """)
    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"\n{'='*50}\n🤖 Agent: {current_agent}\n{'='*50}\n")
        elif isinstance(event, AgentOutput):
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

async def main():
    print("\nWelcome to the Multi-Agent Stock Market Analyst! Type 'exit' to quit.\n")
    while True:
        print("\nEnter a stock symbol or company name (e.g., AAPL, TSLA):")
        user_input = input(" > ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("\nThanks for using the Stock Market Analyst. Goodbye!")
            break
        if not user_input:
            print("Please enter a valid stock name.")
            continue
        print(f"\nAnalyzing {user_input}...")
        try:
            await process_stock_analysis(user_input)
            print("\nDone! Enter another or type 'exit' to quit.")
        except Exception as e:
            print(f"\nError: {str(e)}\nPlease try a different stock.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
