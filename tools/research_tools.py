import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pytrends.request import TrendReq
from crewai import Agent, Task, Crew, Process
from crewai_tools.tools.base_tool import BaseTool
from dotenv import load_dotenv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class ResearchInput:
    """Structured input for the research agent"""
    topic: str
    region: str = "US"  # ISO country code
    language: str = "en"
    target_audience: str = "general"
    video_style: str = "informative"  # informative, entertainment, tutorial, etc.
    duration_preference: str = "medium"  # short (5-8min), medium (8-15min), long (15+min)
    specific_angles: List[str] = None  # specific aspects to focus on
    competitor_analysis: bool = True
    include_statistics: bool = True
    timeframe: str = "now 7-d"  # for trends analysis

class AdvancedGoogleTrendsTool(BaseTool):
    name: str = "Advanced Google Trends Analyzer"
    description: str = """
    Comprehensive Google Trends analysis tool that provides:
    - Interest over time with detailed timeframes
    - Related topics and queries
    - Regional interest breakdown
    - Rising and top searches
    - Comparative analysis with related terms
    """

    def _run(self, topic: str, region: str = "US", timeframe: str = "now 7-d") -> str:
        try:
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
            
            # Main topic analysis
            pytrends.build_payload([topic], cat=0, timeframe=timeframe, geo=region)
            
            # Get interest over time
            interest_data = pytrends.interest_over_time()
            
            # Get related topics and queries
            related_topics = pytrends.related_topics()
            related_queries = pytrends.related_queries()
            
            # Get regional interest
            regional_interest = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
            
            # Get trending searches
            trending_searches = pytrends.trending_searches(pn=region.lower())
            
            # Format results
            result = {
                "topic": topic,
                "region": region,
                "timeframe": timeframe,
                "analysis_date": datetime.now().isoformat(),
                "interest_summary": self._format_interest_data(interest_data, topic),
                "related_topics": self._format_related_data(related_topics, topic),
                "related_queries": self._format_related_data(related_queries, topic),
                "top_regions": self._format_regional_data(regional_interest),
                "trending_context": trending_searches.head(10).values.flatten().tolist() if not trending_searches.empty else []
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error in Google Trends analysis: {str(e)}"
    
    def _format_interest_data(self, data, topic):
        if data.empty:
            return {"status": "no_data", "message": f"No trend data found for {topic}"}
        
        topic_data = data[topic].dropna()
        if topic_data.empty:
            return {"status": "no_data", "message": f"No valid data points for {topic}"}
        
        return {
            "status": "success",
            "peak_interest": int(topic_data.max()),
            "current_interest": int(topic_data.iloc[-1]),
            "average_interest": round(topic_data.mean(), 2),
            "trend_direction": "rising" if topic_data.iloc[-1] > topic_data.iloc[-3] else "declining",
            "data_points": topic_data.tail(7).to_dict()
        }
    
    def _format_related_data(self, data, topic):
        if not data or topic not in data:
            return {"top": [], "rising": []}
        
        result = {"top": [], "rising": []}
        
        if data[topic].get('top') is not None:
            result["top"] = data[topic]['top'].head(10).to_dict('records')
        
        if data[topic].get('rising') is not None:
            result["rising"] = data[topic]['rising'].head(10).to_dict('records')
        
        return result
    
    def _format_regional_data(self, data):
        if data.empty:
            return []
        
        return data.head(10).to_dict()

class SerperSearchTool(BaseTool):
    name: str = "Serper Google Search Tool"
    description: str = """
    Advanced Google search tool using Serper API that provides:
    - Comprehensive search results with snippets
    - News articles and recent content
    - Video search results
    - Image search capabilities
    - Related searches and questions
    """
    
    def _run(self, query: str, search_type: str = "search", num_results: int = 10) -> str:
        try:
            # Get API key from environment
            api_key = os.getenv('SERPER_API_KEY')
            if not api_key:
                return "Error: SERPER_API_KEY environment variable is required"
            
            url = f"https://google.serper.dev/{search_type}"
            
            payload = {
                "q": query,
                "num": num_results,
                "gl": "us",  # geolocation
                "hl": "en"   # language
            }
            
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response for better readability
            formatted_result = {
                "query": query,
                "search_type": search_type,
                "total_results": data.get("searchInformation", {}).get("totalResults", "Unknown"),
                "results": []
            }
            
            # Process organic results
            if "organic" in data:
                for result in data["organic"][:num_results]:
                    formatted_result["results"].append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": result.get("date", "")
                    })
            
            # Add news results if available
            if "news" in data:
                formatted_result["news"] = []
                for news in data["news"][:5]:
                    formatted_result["news"].append({
                        "title": news.get("title", ""),
                        "link": news.get("link", ""),
                        "snippet": news.get("snippet", ""),
                        "date": news.get("date", ""),
                        "source": news.get("source", "")
                    })
            
            # Add related searches
            if "relatedSearches" in data:
                formatted_result["related_searches"] = [
                    item.get("query", "") for item in data["relatedSearches"][:5]
                ]
            
            # Add people also ask
            if "peopleAlsoAsk" in data:
                formatted_result["people_also_ask"] = [
                    {
                        "question": item.get("question", ""),
                        "snippet": item.get("snippet", "")
                    } for item in data["peopleAlsoAsk"][:5]
                ]
            
            return json.dumps(formatted_result, indent=2)
            
        except Exception as e:
            return f"Error in Serper search: {str(e)}"

class ContentAnalysisTool(BaseTool):
    name: str = "Content Analysis Tool"
    description: str = "Analyzes content structure, statistics, and provides insights for script writing"
    
    def _run(self, research_data: str, topic: str) -> str:
        try:
            # Parse the research data
            data = json.loads(research_data) if isinstance(research_data, str) else research_data
            
            analysis = {
                "topic": topic,
                "content_angles": [],
                "key_statistics": [],
                "trending_aspects": [],
                "audience_interests": [],
                "script_suggestions": []
            }
            
            # Extract content angles from search results
            if "results" in data:
                for result in data["results"][:5]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    
                    # Extract potential content angles
                    if any(keyword in title.lower() for keyword in ["how to", "guide", "tutorial"]):
                        analysis["content_angles"].append(f"Tutorial: {title}")
                    elif any(keyword in title.lower() for keyword in ["statistics", "facts", "data"]):
                        analysis["key_statistics"].append(snippet)
                    elif any(keyword in title.lower() for keyword in ["trend", "latest", "new"]):
                        analysis["trending_aspects"].append(title)
            
            # Extract questions for audience engagement
            if "people_also_ask" in data:
                analysis["audience_interests"] = [
                    q["question"] for q in data["people_also_ask"]
                ]
            
            # Generate script suggestions
            analysis["script_suggestions"] = [
                "Start with a trending hook from current data",
                "Include statistics to build credibility",
                "Address common questions from 'People Also Ask'",
                "Reference related trending topics for broader appeal",
                "Use regional interest data to tailor content"
            ]
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in content analysis: {str(e)}"

def create_research_crew(research_input: ResearchInput):
    """Create a specialized research crew for YouTube content"""
    
    # Initialize tools
    trends_tool = AdvancedGoogleTrendsTool() # second this
    search_tool = SerperSearchTool()   # first this
    analysis_tool = ContentAnalysisTool() # last

    gemini_llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                            verbose=True,
                            temperature=0.5,
                            google_api_key=os.getenv("GOOGLE_API_KEY"))


    # Define the researcher agent
    researcher = Agent(
        role='Senior Content Researcher',
        goal=f'Conduct comprehensive research on "{research_input.topic}" to create data-driven, engaging YouTube content',
        backstory="""You are an expert content researcher specializing in YouTube video creation. 
        You have deep knowledge of trending topics, audience psychology, and data-driven content strategy. 
        You excel at finding unique angles, compelling statistics, and audience engagement opportunities.""",
        tools=[trends_tool, search_tool, analysis_tool],
        verbose=True,
        llm=gemini_llm,
        allow_delegation=False
    )
    
    # Define the research task
    research_task = Task(
        description=f"""
        Conduct comprehensive research for a YouTube video on "{research_input.topic}". 
        
        Your research should include:
        1. Google Trends analysis for the topic in {research_input.region}
        2. Current search trends and related queries
        3. Recent news and developments
        4. Audience questions and interests
        5. Statistical data and facts
        6. Content angle recommendations
        7. Competitor analysis (if requested)
        
        Target audience: {research_input.target_audience}
        Video style: {research_input.video_style}
        Duration: {research_input.duration_preference}
        
        Focus on finding:
        - Trending aspects that can hook viewers
        - Credible statistics and data points
        - Common questions and pain points
        - Unique angles not commonly covered
        - Regional preferences and interests
        
        Provide actionable insights for script writing.
        """,
        expected_output="""
        A comprehensive research report containing:
        1. Executive Summary with key findings
        2. Trend Analysis with current interest levels
        3. Content Opportunities with specific angles
        4. Audience Insights with common questions
        5. Statistical Evidence with credible data
        6. Script Framework with suggested structure
        7. SEO Recommendations with keyword opportunities
        """,
        agent=researcher
    )
    
    # Create and return the crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

# Usage example and main execution
def main():
    """Main function to demonstrate the research agent"""
    
    # Check for required environment variables
    if not os.getenv('SERPER_API_KEY'):
        print("⚠️  SERPER_API_KEY not found in environment variables.")
        print("   Creating .env file template...")
        
        env_template = """# Add your Serper API key here
        # Get it from: https://serper.dev
        SERPER_API_KEY=your_api_key_here
        """
        
        try:
            with open('.env', 'w') as f:
                f.write(env_template)
            print("   ✅ .env file created. Please add your SERPER_API_KEY and run again.")
        except Exception as e:
            print(f"   ❌ Could not create .env file: {e}")
        
        print("\n   For now, running with Google Trends only...")
    else:
        print("key available")
    
    # Example research input
    research_input = ResearchInput(
        topic="iPhone 17 price predictions and rumors",
        region="US",
        target_audience="tech enthusiasts",
        video_style="informative",
        duration_preference="medium",
        specific_angles=["price comparison", "feature speculation", "market impact"],
        include_statistics=True
    )
    
    print("🔍 Starting comprehensive research...")
    print(f"Topic: {research_input.topic}")
    print(f"Region: {research_input.region}")
    print(f"Target Audience: {research_input.target_audience}")
    print("-" * 50)
    
    # Create and run the research crew
    crew = create_research_crew(research_input)
    result = crew.kickoff()
    
    print("\n📊 Research Complete!")
    print("=" * 50)
    print(result)
    
    return result

if __name__ == "__main__":
    # Make sure to set your environment variables:
    # SERPER_API_KEY=your_serper_api_key
    
    main()