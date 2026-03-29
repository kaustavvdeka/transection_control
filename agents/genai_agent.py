"""
agents/genai_agent.py
----------------------
GenAI component for financial operations analysis.
Uses Gemini API for root-cause attribution and strategic insights.
"""

import os
import logging
import pandas as pd
import google.generativeai as genai
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GenAIAgent:
    """
    Handles LLM-powered financial analysis.
    
    Prompts are designed to provide:
    1. Root-cause attribution for discrepancies.
    2. Remediation steps for financial close.
    3. Narrative summaries of financial health.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.enabled = bool(self.api_key)
        self.model_name = model_name
        
        if self.enabled:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"GenAIAgent initialized with model: {model_name}")
        else:
            logger.warning("GOOGLE_API_KEY missing. GenAIAgent running in MOCK mode.")

    def _call_llm(self, prompt: str) -> str:
        """Call Gemini API or return mock response."""
        if not self.enabled:
            return self._get_mock_response(prompt)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"GenAI call failed: {e}")
            return f"Error generating narrative: {str(e)}"

    def _get_mock_response(self, prompt: str) -> str:
        """Fallback content for demonstration without API key."""
        if "reconciliation" in prompt.lower():
            return (
                "### Financial Reconciliation Insights (MOCKED)\n\n"
                "* **Root Cause**: Observed high variance in 'CASH_OUT' transactions for vendor 'M8374'. "
                "The mismatch suggests a synchronization lag between our internal ledger and the bank gateway.\n"
                "* **Risk Level**: **Medium**. Total variance of ₹45,200 identified.\n"
                "* **Remediation**: Re-trigger gateway sync for batch timestamp 2023-01-05T04:20:00. "
                "Update the ledger entry T748291 to match bank record."
            )
        return "GenAI Analysis unavailable. Please set GOOGLE_API_KEY."

    def analyze_discrepancy(self, txn_row: Dict[str, Any]) -> str:
        """Generate a narrative for a single high-impact discrepancy."""
        prompt = f"""
        Analyze the following financial discrepancy and provide a root-cause hypothesis and remediation step.
        
        Transaction ID: {txn_row.get('transaction_id')}
        Status: {txn_row.get('status')}
        Type: {txn_row.get('transaction_type')}
        Ledger Amount: ₹{txn_row.get('amount')}
        Bank Amount: ₹{txn_row.get('bank_amount')}
        Variance: ₹{txn_row.get('amount_variance')}
        Sender: {txn_row.get('sender')}
        Receiver: {txn_row.get('receiver')}
        
        Format as:
        Root Cause: [Explanation]
        Remediation: [Specific Step]
        """
        return self._call_llm(prompt)

    def generate_closing_report(self, summary_stats: Dict[str, Any], top_issues: pd.DataFrame) -> str:
        """Produce a high-level summary for the financial close."""
        issues_summary = top_issues[['transaction_id', 'status', 'amount']].to_string()
        
        prompt = f"""
        You are a FinOps AI Agent. Summarize the current state of the financial close based on these metrics:
        
        Total Transactions: {summary_stats.get('total_transactions')}
        Issues Found: {summary_stats.get('total_issues')}
        Total Variance: ₹{summary_stats.get('total_variance_inr')}
        
        Top Discrepancies:
        {issues_summary}
        
        Task:
        1. Summarize the major blockers to closing the books.
        2. Assign a 'Close Readiness' score (0-100%).
        3. List top 3 priorities for the Finance Ops team.
        """
        return self._call_llm(prompt)

if __name__ == "__main__":
    # Quick Test
    agent = GenAIAgent()
    print(agent.generate_closing_report({"total_transactions": 1000, "total_issues": 5, "total_variance_inr": 50000}, pd.DataFrame()))
