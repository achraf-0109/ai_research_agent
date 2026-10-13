from src.agents.agents import (search_agent,
                            reader_agent,
                            write_agent,
                            critic_agent
                        )

from fpdf import FPDF
from time import time

def run_ai_research(topic: str) -> str:
    state = {}

    # initialize total latency 
    latency = 0

    # search agent
    print("search agent is working...")
    start = time()
    search_result = search_agent().invoke({
    "messages" : [("user", 
                f"Find recent, reliable and detailed sources about: {topic}"
        )]
    })
    end = time()
    state["search_result"] = search_result['messages'][-1].content
    latency += (end-start)
    print(f"Search Latency: {end-start}")

    # scraping agent
    print("scraping agent is working...")
    start = time()
    scraped_content = reader_agent().invoke({
    "messages" : [("user", 
                f"Based on the following research results about {topic}"
                f"Pick the most relevant url and scrape it"
                f"Search results: {state['search_result']}"      
        )]
    })
    end = time()
    state["scraped_content"] = scraped_content['messages'][-1].content
    latency += (end-start)
    print(f"Scraping Latency: {end-start}")

    # writer agent
    print("writer agent is working...")
    research = (
        f"search results: \n {state['search_result']}"
        f"scraped content: \n{state['scraped_content']}"
    )

    start = time()
    state["initial_report"] = write_agent().invoke({
        "topic": topic,
        "research": research
    })
    end = time()
    latency += (end-start)
    print(f"Writing Latency: {end-start}")

    # critic agent
    print("critic agent is working...")
    start = time()
    state["final_report"] = critic_agent().invoke({
        "topic": topic,
        "research": research,
        "report": state["initial_report"]
    })
    end = time()
    latency += (end-start)
    print(f"Critizing Latency: {end-start}")

    print(f"""=====================================\n
        Total Latency: {latency:.2f}
        \n=========================================""")

    return state

def save_report(state: dict[str, str]):
    with open('src/reports/report.txt', 
            'w',
            encoding="utf-8") as f:
        f.write(state['initial_report'])

    return 'src/reports/report.txt'

def convert_to_pdf(file):
    with open(file, 
            'r',
            encoding="latin-1") as f:
        content = f.readlines()

    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add each line from the variable to the PDF
    for line in content:
        pdf.cell(80,5, txt =line, ln = 1, align = 'C') 

    # Save as PDF
    pdf.output("src/reports/output.pdf")