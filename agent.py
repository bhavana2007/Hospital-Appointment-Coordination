import os
import json
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from tools import TOOLS_MAP

SYSTEM_PROMPT = """You are a helpful hospital appointment booking assistant. You help patients find doctors, check available slots, book/reschedule/cancel appointments. You also help hospital staff manage doctor schedules.

IMPORTANT RULES:
- Always use the provided tools to perform actions. Never make up data.
- When a user asks to book, reschedule, or cancel, call the appropriate tool.
- Be conversational and helpful. Confirm actions clearly.
- For patients: you can search_doctors, check_slots, book_appointment, get_my_appointments, cancel_appointment, reschedule_appointment.
- For staff: you can list_all_doctors, get_doctor_schedule, update_doctor_hours. You can also search_doctors and check_slots.
- Never reveal other patients' information.
- Dates should be in YYYY-MM-DD format, times in HH:MM (24-hour).
- Today's date is injected in the context. The booking window is 7 days from today.
"""

TOOL_DECLARATIONS = [
    FunctionDeclaration(
        name="search_doctors",
        description="Search for doctors by specialty and/or hospital name. Returns matching doctors with their details and working hours.",
        parameters={
            "type": "object",
            "properties": {
                "specialty": {"type": "string", "description": "Medical specialty to search for (e.g., Cardiology, Pediatrics). Partial match."},
                "hospital_name": {"type": "string", "description": "Hospital name to filter by. Partial match."},
            },
        },
    ),
    FunctionDeclaration(
        name="check_slots",
        description="Get available 30-minute appointment slots for a doctor on a specific date. Slots are computed from the doctor's working hours.",
        parameters={
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer", "description": "The doctor's ID."},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
            },
            "required": ["doctor_id", "date"],
        },
    ),
    FunctionDeclaration(
        name="book_appointment",
        description="Book an appointment slot with a doctor. Validates all business rules.",
        parameters={
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer", "description": "The doctor's ID."},
                "slot_datetime": {"type": "string", "description": "Slot in YYYY-MM-DD HH:MM format."},
            },
            "required": ["doctor_id", "slot_datetime"],
        },
    ),
    FunctionDeclaration(
        name="get_my_appointments",
        description="Get all appointments for the logged-in patient.",
        parameters={"type": "object", "properties": {}},
    ),
    FunctionDeclaration(
        name="cancel_appointment",
        description="Cancel a scheduled appointment. Only the appointment owner can cancel.",
        parameters={
            "type": "object",
            "properties": {
                "appointment_id": {"type": "integer", "description": "The appointment ID to cancel."},
            },
            "required": ["appointment_id"],
        },
    ),
    FunctionDeclaration(
        name="reschedule_appointment",
        description="Reschedule an existing appointment to a new time slot. Implemented as cancel + rebook with rollback.",
        parameters={
            "type": "object",
            "properties": {
                "appointment_id": {"type": "integer", "description": "The appointment ID to reschedule."},
                "new_slot_datetime": {"type": "string", "description": "New slot in YYYY-MM-DD HH:MM format."},
            },
            "required": ["appointment_id", "new_slot_datetime"],
        },
    ),
    FunctionDeclaration(
        name="get_doctor_schedule",
        description="View a doctor's schedule for a specific date. Staff only.",
        parameters={
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer", "description": "The doctor's ID."},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
            },
            "required": ["doctor_id", "date"],
        },
    ),
    FunctionDeclaration(
        name="list_all_doctors",
        description="List all doctors with their working hours and hospital info. Staff only.",
        parameters={"type": "object", "properties": {}},
    ),
    FunctionDeclaration(
        name="update_doctor_hours",
        description="Update a doctor's working hours and days. Staff only.",
        parameters={
            "type": "object",
            "properties": {
                "doctor_id": {"type": "integer", "description": "The doctor's ID."},
                "work_start": {"type": "string", "description": "Start time in HH:MM format."},
                "work_end": {"type": "string", "description": "End time in HH:MM format."},
                "work_days": {"type": "string", "description": "Comma-separated weekday numbers (0=Mon, 6=Sun)."},
            },
            "required": ["doctor_id", "work_start", "work_end", "work_days"],
        },
    ),
]

gemini_tool = Tool(function_declarations=TOOL_DECLARATIONS)


class GeminiAgent:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
            tools=[gemini_tool],
            system_instruction=SYSTEM_PROMPT,
        )

    def chat(self, history: list, message: str, caller_id: int, caller_role: str, caller_name: str) -> str:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        context_msg = f"[System: You are speaking with {caller_name} (role: {caller_role}, id: {caller_id}). Today's date is {today}. Booking window is 7 days from today.]"

        gemini_history = []
        for msg in history:
            gemini_history.append({"role": msg["role"], "parts": msg["parts"]})

        chat_session = self.model.start_chat(history=gemini_history)
        full_message = f"{context_msg}\n\nUser: {message}"

        response = chat_session.send_message(full_message)

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]

                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}

                    if func_name in TOOLS_MAP:
                        func_args["caller_id"] = caller_id
                        func_args["caller_role"] = caller_role
                        try:
                            result = TOOLS_MAP[func_name](**func_args)
                        except Exception as e:
                            result = {"error": str(e)}

                        response = chat_session.send_message(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=func_name,
                                    response=result,
                                )
                            )
                        )
                    else:
                        response = chat_session.send_message(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=func_name,
                                    response={"error": f"Unknown function: {func_name}"},
                                )
                            )
                        )
                else:
                    if hasattr(part, 'text') and part.text:
                        return part.text
                    break
            else:
                break

        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if hasattr(part, 'text') and part.text:
                return part.text

        return "I'm sorry, I couldn't process your request. Please try again."