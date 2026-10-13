from src.pipeline.pipeline import (run_ai_research, 
                                save_report,
                                convert_to_pdf)
from rich import print

report_details = run_ai_research("cybersecurity")
report = save_report(report_details)