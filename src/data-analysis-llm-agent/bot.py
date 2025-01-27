import asyncio
import logging
import os
import json
import google.generativeai as genai
from dataclasses import dataclass
import plotly.graph_objects as go
import chainlit as cl

@dataclass
class GeminiResponse:
    content: str
    tool_calls: list = None

class ChatBot:
    def __init__(self, system, tools, tool_functions):
        self.system = system
        self.tools = tools
        self.exclude_functions = ["plot_chart"]
        self.tool_functions = tool_functions
        self.messages = []
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.chat = self.model.start_chat(history=[])  # Initialize chat with an empty history

        if self.system:
            self.chat.send_message(self.system)

    def _parse_sql_query(self, text):
        """Extract SQL query from text"""
        query_start = text.upper().find("SELECT")
        if query_start == -1:
            return None
            
        query_end = text.find(";", query_start)
        if query_end == -1:
            query_end = len(text)
            
        return text[query_start:query_end + 1]

    def _parse_tool_calls(self, response_text):
        """Parse Gemini's response for potential tool calls"""
        tool_calls = []

        # Parse SQL queries
        if "SELECT" in response_text.upper():
            sql_query = self._parse_sql_query(response_text)
            if sql_query:
                tool_calls.append({
                    "id": f"query_{len(tool_calls)}",
                    "function": {
                        "name": "query_db",
                        "arguments": json.dumps({"sql_query": sql_query})
                    }
                })

        # Parse plot requests
        if "PLOT" in response_text.upper() or "CHART" in response_text.upper():
            start_idx = response_text.find("plot_chart(")
            end_idx = response_text.find(")", start_idx)
            if start_idx != -1 and end_idx != -1:
                plot_args = response_text[start_idx + 11:end_idx]
                try:
                    args = json.loads(f"{{{plot_args}}}")
                    tool_calls.append({
                        "id": f"plot_{len(tool_calls)}",
                        "function": {
                            "name": "plot_chart",
                            "arguments": json.dumps({
                                "plot_type": args.get("plot_type", "line"),
                                "x_values": args.get("x", []),
                                "y_values": args.get("y", []),
                                "plot_title": args.get("title", "Chart"),
                                "x_label": args.get("xlabel", "X"),
                                "y_label": args.get("ylabel", "Y")
                            })
                        }
                    })
                except Exception as e:
                    logging.error(f"Failed to parse plot arguments: {e}")

        return tool_calls

    async def __call__(self, message):
        """Process user message and return response"""
        logging.info(f"User message: {message}")
        response = self.chat.send_message(message)
        response_text = response.text
        
        # Parse for tool calls
        tool_calls = self._parse_tool_calls(response_text)
        
        return GeminiResponse(
            content=response_text,
            tool_calls=tool_calls
        )

    async def call_function(self, tool_call):
        """Execute a single tool call"""
        function_name = tool_call["function"]["name"]
        function_to_call = self.tool_functions.get(function_name)
        function_args = json.loads(tool_call["function"]["arguments"])

        try:
            if function_name == "plot_chart":
                # Generate the chart
                x_values = function_args["x_values"]
                y_values = function_args["y_values"]
                plot_title = function_args["plot_title"]
                x_label = function_args["x_label"]
                y_label = function_args["y_label"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines"))
                fig.update_layout(
                    title=plot_title,
                    xaxis_title=x_label,
                    yaxis_title=y_label
                )
                return {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": fig,  # Return the figure object
                }
            else:
                # Call other tools
                function_response = await function_to_call(**function_args)
                return {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
        except Exception as e:
            logging.error(f"Error calling function {function_name}: {str(e)}")
            return {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": function_name,
                "content": f"Error: {str(e)}",
            }

    async def call_functions(self, tool_calls):
        """Execute tool calls and get response"""
        function_responses = await asyncio.gather(
            *(self.call_function(tool_call) for tool_call in tool_calls)
        )

        # Handle charts separately
        for res in function_responses:
            if isinstance(res["content"], go.Figure):
                chart = res["content"]
                chart_element = cl.Plotly(name="chart", figure=chart, display="inline")
                await cl.Message(author="Assistant", content="", elements=[chart_element]).send()

        # Format response for Gemini
        result_text = "Here are the results:\n"
        for res in function_responses:
            if not isinstance(res["content"], go.Figure):
                result_text += f"\n{res['content']}"
        logging.info(f"Tool responses: {result_text}")
        response = self.chat.send_message(result_text)

        return GeminiResponse(
            content=response.text,
            tool_calls=self._parse_tool_calls(response.text)
        ), function_responses
